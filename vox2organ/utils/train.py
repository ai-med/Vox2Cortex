
""" Training procedure """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import re
from copy import deepcopy

import json
import wandb
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CyclicLR

import logger
from params.default import hps_to_wandb
from data.dataset_split_handler import dataset_split_handler
from models.model_handler import ModelHandler
from utils.template import MeshTemplate, TEMPLATE_SPECS
from utils.utils import (
    score_is_better,
    load_checkpoint,
    save_checkpoint,
    grad_norm,
)
from utils.losses import ChamferAndNormalsLoss
from utils.mesh import MeshesOfMeshes
from utils.distributed import (
    cleanup,
    setup_ddp,
    prepare_dataloader,
)
from utils.evaluate import ModelEvaluator
from utils.losses import (
    all_linear_loss_combine,
)
from utils.model_names import (
    INTERMEDIATE_MODEL_NAME,
    BEST_MODEL_NAME,
    FINAL_MODEL_NAME
)


log = logger.get_std_logger(__name__)


def _init_optimizer(model, optim_class, optim_state_dict=None, **optim_params):
    """ Initialize an optimizer for training depending on whether a
    separate lr for a graph network is specified.
    """
    if optim_params.get('graph_lr', None) is not None:
        # Separate learning rates for voxel and graph network
        graph_lr = optim_params['graph_lr']
        optim_params_new = optim_params.copy()
        del optim_params_new['graph_lr']
        try:
            optim = optim_class([
                {'params': model.voxel_net.parameters()},
                {'params': model.graph_net.parameters(), 'lr': graph_lr},
            ], **optim_params_new)
        except AttributeError:
            # DDP model
            optim = optim_class([
                {'params': model.module.voxel_net.parameters()},
                {'params': model.module.graph_net.parameters(), 'lr': graph_lr},
            ], **optim_params_new)
    else:
        if 'graph_lr' in optim_params:
            del optim_params['graph_lr']
        # All parameters updated with the same lr
        optim = optim_class(
            model.parameters(), **optim_params
        )

    # If a pre-trained model is resumed, the learning rates
    # might have been changed with lr scheduling, so in this case, load
    # the current learning rate from the checkpoint here:
    if optim_state_dict is not None:
        optim.load_state_dict(optim_state_dict)

    optim.zero_grad()

    return optim


def _init_lr_scheduler(
    scheduler_class,
    optim,
    scheduler_state_dict=None,
    **scheduler_params
):
    """ Initialize a lr scheduler
    """
    if scheduler_class == CyclicLR:
        # Parameters of CycliclLR as recommended by Leslie Smith
        lr_scheduler = CyclicLR(
            optim,
            base_lr=[p['lr'] for p in optim.param_groups],
            max_lr=[4 * p['lr'] for p in optim.param_groups],
            **scheduler_params
        )

    if scheduler_state_dict is not None:
        lr_scheduler.load_state_dict(scheduler_state_dict)

    return lr_scheduler


class Solver():
    """
    Solver class for optimizing the weights of neural networks. It supports
                 distributed training on multiple GPUs.

    :param rank: The rank/device to train on
    :param world size: Number of devices to train on in total
    :param mesh_template: The template mesh used for training.
    :param torch.optim optimizer_class: The optimizer to use, e.g. Adam.
    :param dict optim_params: The parameters for the optimizer. If empty,
    default values are used.
    :param evaluator: Evaluator for the optimized model.
    :param list voxel_loss_func: A list of loss functions to apply for the 3D voxel
    prediction.
    :param list voxel_loss_func_weights: A list of the same length of 'voxel_loss_func'
    with weights for the losses.
    :param list mesh_loss_func: A list of loss functions to apply for the mesh
    prediction.
    :param list mesh_loss_func_weights: A list of the same length of 'mesh_loss_func'
    with weights for the losses.
    :param str save_path: The path where results and stats are saved.
    :param log_every: Log the stats every n iterations.
    :param str device: The device(s) for execution. It can be a single device,
    e.g., "cuda:0" or a list of multiple devices for DDP, e.g.,
    ["cuda:0", "cuda:1"]
    :param str main_eval_metric: The main evaluation metric according to which
    the best model is determined.
    :param int accumulate_n_gradients: Gradient accumulation of n gradients.
    :param bool mixed_precision: Whether or not to use automatic mixed
    precision.
    :param clip_gradient: Clip gradient at this norm if specified (not False)
    :param lr_scheduler_params: Parameters for the lr scheduler
    :param n_epochs: Number of training epochs
    :param freeze_pre_trained: Whether pretrained weights should be freezed
    :param eval_every: Evaluate every n epochs on validation set
    :param save_models: Whether to save models every 'eval_every' epoch
    :param scheduler_class: The class to use as a scheduler
    :param batch_size: Batch size for training

    """

    def __init__(
        self,
        rank,
        world_size,
        mesh_template: MeshTemplate,
        optimizer_class,
        optim_params,
        evaluator,
        voxel_loss_func,
        voxel_loss_func_weights,
        mesh_loss_func,
        mesh_loss_func_weights,
        save_path,
        log_every,
        device,
        main_eval_metric,
        accumulate_n_gradients,
        mixed_precision,
        clip_gradient,
        lr_scheduler_params,
        n_epochs,
        freeze_pre_trained,
        eval_every,
        scheduler_class,
        batch_size,
        resample_targets,
        **kwargs
    ):

        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.patience = lr_scheduler_params.pop('patience')
        self.gamma_plateau = lr_scheduler_params.pop('gamma_plateau')
        self.lr_scheduler_params = lr_scheduler_params
        self.scheduler_class = scheduler_class
        self.n_epochs = n_epochs
        self.eval_every = eval_every
        self.resample_targets = resample_targets
        self.freeze_pre_trained = freeze_pre_trained
        self.optim_class = optimizer_class
        self.optim_params = optim_params
        self.optim = None # defined for each training separately
        self.scaler = GradScaler() # for mixed precision
        self.evaluator = evaluator
        self.voxel_loss_func = voxel_loss_func
        self.voxel_loss_func_weights = voxel_loss_func_weights
        assert len(voxel_loss_func) == len(voxel_loss_func_weights),\
                "Number of weights must be equal to number of 3D seg. losses."

        self.mesh_loss_func = mesh_loss_func
        self.mesh_loss_func_weights = mesh_loss_func_weights
        self.clip_gradient = clip_gradient
        if any(isinstance(lf, ChamferAndNormalsLoss)
               for lf in self.mesh_loss_func):
            assert len(mesh_loss_func) + 1 == len(mesh_loss_func_weights),\
                    "Number of weights must be equal to number of mesh losses."
        else:
            assert len(mesh_loss_func) == len(mesh_loss_func_weights),\
                    "Number of weights must be equal to number of mesh losses."

        self.save_path = save_path
        self.log_every = log_every
        self.devices = [device] if isinstance(device, str) else device
        self.distributed = world_size > 1
        # Log only in rank 0 or always if training is not distributed
        self.log_active = not self.distributed or rank == 0
        self.main_eval_metric = main_eval_metric
        self.accumulate_ngrad = accumulate_n_gradients
        self.mixed_precision = mixed_precision
        self.template = mesh_template
        self.use_wandb = kwargs.get('use_wandb', False)
        self.save_models = kwargs.get('save_models', True)

    @logger.measure_time
    def training_step(self, model, data, iteration):
        """ One training step.

        :param model: Current pytorch model.
        :param data: The minibatch.
        :param iteration: The training iteration (used for logging)
        :returns: The overall (weighted) loss.
        """

        # Loss
        loss_total = self.compute_loss(model, data, iteration)

        if self.mixed_precision:
            self.scaler.scale(loss_total).backward()
        else:
            loss_total.backward()

        # Log gradient norm and optionally clip gradient
        parameters = model.module.parameters() if (
            isinstance(model, DDP)
        ) else model.parameters()
        grad = grad_norm(parameters)
        log.debug("Gradient norm: %.5f (rank %d)", grad, self.rank)
        if logger.wandb_is_active():
            wandb.log({"Gradient_norm": grad}, step=iteration)

        if self.clip_gradient:
            clip_grad_norm_(parameters, self.clip_gradient)

        # Accumulate gradients
        if iteration % self.accumulate_ngrad == 0:
            if self.mixed_precision:
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                self.optim.step()

            self.optim.zero_grad()
            log.debug("Rank %d updated parameters.", self.rank)

        return loss_total

    @logger.measure_time
    def compute_loss(self, model, data, iteration) -> torch.tensor:
        # Chop data
        (x_img,
         y,
         points,
         normals,
         curvs,
         parcs) = data

        input_meshes = self.template.create_template_batch_size(
            x_img.shape[0], self.rank
        )

        log.debug(
            "%d reference points in ground truth", points.shape[-2]
        )

        # Predict
        with autocast(self.mixed_precision):
            pred = model(x_img.to(self.rank), input_meshes)

        model_class = model.module.__class__ if (
            isinstance(model, DDP)
        ) else model.__class__

        # Mesh and voxel prediction
        pred_meshes = model_class.pred_to_pred_meshes(pred) # MeshesXD
        raw_voxel_pred = model_class.pred_to_raw_voxel_pred(pred)

        # Log
        logger.write_img_if_debug(
            x_img.cpu().squeeze().numpy(),
            os.path.join(
                logger.get_log_dir(),
                "voxel_input_img_train.nii.gz"
            )
        )
        logger.write_img_if_debug(
            y.cpu().squeeze().numpy(),
            os.path.join(
                logger.get_log_dir(),
                "voxel_target_img_train.nii.gz"
            )
        )

        # Magnitude of displacement vectors: mean per deformation step
        try:
            disps = pred_meshes.verts_features_packed()[:, -3:]
            disps = disps.view((model.deform_stages + 1, -1, 3))
        except AttributeError:
            # Some models don't return the displacements
            disps = None

        if (
            iteration % self.log_every == 0 and
            self.log_active and
            disps is not None
        ):
            d_mean = disps.detach().cpu().norm(dim=-1).mean(dim=-1)
            d_log = {f"deltaV_{i}": di for i, di in enumerate(d_mean)}
            for k, v in d_log.items():
                log.info("%s: %.5f (rank %d)", k, v, self.rank)
            if logger.wandb_is_active():
                wandb.log(d_log, step=iteration)

        losses = {}
        with autocast(self.mixed_precision):
            losses, loss_total = all_linear_loss_combine(
                # Voxel pred
                self.voxel_loss_func,
                torch.tensor(self.voxel_loss_func_weights, device=self.rank),
                raw_voxel_pred,
                y.to(self.rank),
                # Mesh pred
                self.mesh_loss_func,
                torch.tensor(self.mesh_loss_func_weights, device=self.rank),
                pred_meshes,
                [points.to(self.rank),
                 normals.to(self.rank),
                 curvs.to(self.rank),
                 parcs.to(self.rank)]
            )

        losses['TotalLoss'] = loss_total

        # Log
        if iteration % self.log_every == 0:
            det_losses = {k: v.detach() for k, v in losses.items()}
            for k, v in det_losses.items():
                log.info("%s: %.5f (rank %d)", k, v, self.rank)
            if logger.wandb_is_active() and self.log_active:
                wandb.log(det_losses, step=iteration)

        return loss_total

    def train(
        self,
        model: torch.nn.Module,
        training_set: torch.utils.data.Dataset,
        start_epoch: int,
        optim_state_dict: dict,
        scheduler_state_dict: dict,
    ):
        """
        Training procedure
        """

        # Init wandb logging
        if self.use_wandb and self.log_active:
            logger.init_wandb(**hps_to_wandb(logger.get_experiment_hps()))
            log.info("Wandb active: %r", logger.wandb_is_active())


        # Init best values
        best_val_score = None
        best_epoch = 0
        best_state = None
        main_val_score = None


        log.info("Training on device %d", self.rank)

        # Optimizer
        self.optim = _init_optimizer(
            model,
            self.optim_class,
            optim_state_dict=optim_state_dict,
            **self.optim_params
        )

        # Lr scheduling
        # In distributed training, the batch size is the per-rank batch
        # size x the world size since every rank has its own lr scheduler
        # Attention: This assumes that every rank has equal batch size!!!
        scheduler_step_size_up = int( # For CyclicLR
            np.ceil(len(training_set)/float(self.batch_size * self.world_size))
        ) * 10
        lr_scheduler = _init_lr_scheduler(
            self.scheduler_class,
            self.optim,
            scheduler_state_dict=scheduler_state_dict,
            step_size_up=scheduler_step_size_up,
            **self.lr_scheduler_params,
        )

        # Training loader potentially distributed
        training_loader = prepare_dataloader(
            rank=self.rank,
            world_size=self.world_size,
            batch_size=self.batch_size,
            dataset=training_set,
            shuffle=True
        )
        log.info(
            "Created training loader of length %d (rank %d)",
            len(training_loader),
            self.rank
        )

        # Logging every epoch
        log_was_epoch = False
        if self.log_every == 'epoch':
            log_was_epoch = True
            self.log_every = len(training_loader)

        epochs_file = os.path.join(self.save_path, "models_to_epochs.json")
        models_to_epochs = {}

        # Compute the iteration for a potentially resumed training
        iteration = (start_epoch - 1) * len(training_loader) + 1

        for epoch in range(start_epoch, self.n_epochs+1):

            # Tell distributed sampler which epoch it is
            try:
                training_loader.sampler.set_epoch(epoch)
            except AttributeError:
                # Training loader may not have a sampler
                pass

            # Potentially freeze parameters and set all others to train
            if isinstance(model, DDP):
                model.module.train(self.freeze_pre_trained)
            else:
                model.train(self.freeze_pre_trained)

            # Iterate over training dataloader
            for iter_in_epoch, data in enumerate(training_loader):
                # Log
                if iteration % self.log_every == 0 and self.log_active:
                    log.info("Iteration: %d", iteration)
                    log.info("Epoch: %d", epoch)
                    for i, lr in enumerate(lr_scheduler.get_last_lr()):
                        log.info("Lr %d: %.10f", i, lr)
                    if logger.wandb_is_active():
                        wandb.log({"epoch": epoch}, step=iteration)
                        for i, lr in enumerate(lr_scheduler.get_last_lr()):
                            wandb.log({f"lr_{i}": lr}, step=iteration)

                # Training step
                self.training_step(model, data, iteration)

                # Launch scheduler step
                lr_scheduler.step()

                iteration += 1

            # Validate only in certain epochs and only in rank 0 process
            # In theory, it would be possible to validate also in parallel.
            # However, we cannot create a distributed validation loader at the
            # moment since we do not use "get_item_from_index" at
            # validation/test time.
            if (epoch % self.eval_every == 0 or
                epoch >= self.n_epochs - self.eval_every or  # Evaluate more often at the end
                epoch == start_epoch) and self.log_active:

                model.eval()
                val_results = self.evaluator.evaluate(
                    model, epoch, torch.device(self.rank), save_predictions=5
                )
                # Log
                val_logs = logger.df_to_wandb_log(val_results)
                for k, v in val_logs.items():
                    log.info("%s: %.5f", k, v)
                if logger.wandb_is_active():
                    wandb.log({"val": val_logs}, step=iteration-1)

                # Save model of current epoch: Takes up a lot of memory and is
                # only recommended for debugging
                # save_checkpoint(
                    # {
                        # 'start_epoch': epoch + 1,
                        # 'state_dict': model.module.state_dict() if (
                            # isinstance(model, DDP)
                        # ) else model.state_dict(),
                        # 'optimizer': self.optim.state_dict(),
                        # 'scheduler': lr_scheduler.state_dict(),
                    # },
                    # os.path.join(self.save_path, f"epoch_{epoch}.pt")
                # )

                # Main validation score
                summary = val_results.groupby('Metric').mean().reset_index()
                main_val_score = summary[
                    summary['Metric'] == self.main_eval_metric
                ]['Value'].item()
                if score_is_better(
                    best_val_score, main_val_score, self.main_eval_metric
                )[0]:
                    best_val_score = main_val_score
                    if logger.wandb_is_active():
                        for k, v in val_logs.items():
                            wandb.run.summary["val." + k] = v
                    best_state = deepcopy(model.state_dict())
                    best_epoch = epoch
                    if self.save_models:
                        save_checkpoint(
                            {
                                'start_epoch': epoch + 1,
                                'state_dict': model.module.state_dict() if (
                                    isinstance(model, DDP)
                                ) else model.state_dict(),
                                'optimizer': self.optim.state_dict(),
                                'scheduler': lr_scheduler.state_dict(),
                            },
                            os.path.join(self.save_path, BEST_MODEL_NAME)
                        )
                        models_to_epochs[BEST_MODEL_NAME] = best_epoch

            # Save intermediate model after each epoch
            if self.save_models and self.log_active:
                model.eval()
                save_checkpoint({
                    'start_epoch': epoch + 1,
                    'state_dict': model.module.state_dict() if (
                        isinstance(model, DDP)
                    ) else model.state_dict(),
                    'optimizer': self.optim.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                }, save_path=os.path.join(self.save_path, INTERMEDIATE_MODEL_NAME))

                models_to_epochs[INTERMEDIATE_MODEL_NAME] = epoch
                with open(epochs_file, 'w') as f:
                    json.dump(models_to_epochs, f)
                log.debug(
                    "Saved intermediate model from epoch %d.", epoch
                )

            if self.distributed:
                # print(f"Rank {self.rank} waiting at barrier.")
                dist.barrier()
                # print(f"Rank {self.rank} passed barrier.")

            # Resample targets and reduce lr
            if epoch - best_epoch >= self.patience and self.resample_targets and self.log_active:
                training_set.create_training_targets(remove_meshes=False)
                for g in self.optim.param_groups:
                    g['lr'] = g['lr'] * self.gamma_plateau
                lr_scheduler = _init_lr_scheduler(
                    self.scheduler_class,
                    self.optim,
                    scheduler_state_dict=scheduler_state_dict,
                    step_size_up=scheduler_step_size_up,
                    **self.lr_scheduler_params,
                )

        # Save final model
        if self.save_models and self.log_active:
            model.eval()
            save_checkpoint({
                'start_epoch': epoch + 1,
                'state_dict': model.module.state_dict() if (
                    isinstance(model, DDP)
                ) else model.state_dict(),
                'optimizer': self.optim.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
            }, save_path=os.path.join(self.save_path, INTERMEDIATE_MODEL_NAME))
            models_to_epochs[FINAL_MODEL_NAME] = epoch
            if best_state is not None:
                log.info("Best model in epoch %d", best_epoch)

            # Save epochs corresponding to models
            with open(epochs_file, 'w') as f:
                json.dump(models_to_epochs, f)

            log.info("Saved models at %s", self.save_path)

            if log_was_epoch:
                self.log_every = 'epoch'

        if self.log_active and self.use_wandb:
            logger.finish_wandb_run()

        # Return last main validation score
        return main_val_score


def distributed_training(
    rank,
    world_size,
    master_port: int,
    model,
    training_set,
    start_epoch,
    optim_state_dict,
    scheduler_state_dict,
    solver_params: dict,
    hps: dict
):
    """ Distributed training with one solver per rank.
    """
    setup_ddp(rank, world_size, master_port)
    log.debug("Setup distributed training in rank %d", rank)

    # Reinit the experiment here in the respective process. If this is not
    # done, the relation to the experiment is lost in DDP processes.
    _,  _ = logger.init_experiment(
        experiment_base_dir=hps['EXPERIMENT_BASE_DIR'],
        experiment_name=hps['EXPERIMENT_NAME'],
        prefix=hps['EXP_PREFIX'],
        exist_ok=True,
        create=False, # Should already exist
        log_time=hps['TIME_LOGGING'],
        log_level=hps['LOGLEVEL'],
    )

    # Wrap model with DDP
    model.float().to(rank)
    log.debug("Model id at rank %d: %s", rank, id(model))
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        # Setting 'find_unused_parameters=True' leads to an error, don't
        # know why see also
        # https://discuss.pytorch.org/t/finding-the-cause-of-runtimeerror-expected-to-mark-a-variable-ready-only-once/124428/5
        # Update: For training with pretrained models,
        # find_unused_parameters should be set to true, which, however, is
        # not possible at the moment...
        find_unused_parameters=False
    )

    # Solver responsible for model training
    solver = Solver(
        rank=rank,
        world_size=world_size,
        **solver_params
    )

    solver.train(
        model,
        training_set,
        start_epoch,
        optim_state_dict,
        scheduler_state_dict
    )

    cleanup()



def training_routine(hps: dict, resume=False):
    """
    A full training routine.

    :param dict hps: Hyperparameters to use.
    :param str experiment_name (optional): The name of the experiment
    directory. If None, a name is created automatically.
    :param loglevel: The loglevel of the standard logger to use.
    :param resume: If true, a previous training is resumed.
    :return: The name of the experiment.
    """

    experiment_name = hps['EXPERIMENT_NAME']

    # Save parameters if a new training is started and check for equality of
    # resumed training parameters otherwise
    logger.save_params(hps, write=(not resume))
    # Sanity check if all parameters are equal to the stored ones
    differ = logger.check_configs_equal(hps)
    for k, v_differ in differ.items():
        # Changed this into a warning, as some hyperparams are ok to change
        log.warning(
            f"Hyperparameter {k} is not equal to the experiment"
            f" that should be resumed. Values are '{v_differ[0]}' (stored),"
            f" '{v_differ[1]}' (new)."
        )

    # Lower case param names as input to constructors/functions
    hps_lower = dict((k.lower(), v) for k, v in hps.items())
    model_config = dict((k.lower(), v) for k, v in hps['MODEL_CONFIG'].items())

    log.info("Start training '%s'...", hps['EXPERIMENT_NAME'])

    ###### Load data ######
    log.info("Loading dataset %s...", hps['DATASET'])
    training_set, validation_set, _ = dataset_split_handler[hps['DATASET']](
        save_dir=logger.get_experiment_dir(),
        load_only=('train', 'validation'),
        check_dir=logger.get_log_dir(),
        **hps_lower
    )
    # Only store relevant targets if they are not resampled during training
    training_set.create_training_targets(remove_meshes=not hps['RESAMPLE_TARGETS'])
    log.info("%d training files.", len(training_set))
    log.info("%d validation files.", len(validation_set))
    log.info(
        "Minimum number of vertices in training set: %d.",
        training_set.n_min_vertices
    )

    ##### Load template ######
    log.info("Loading template %s...", hps['MESH_TEMPLATE_ID'])
    trans_affine = validation_set.get_data_element(0)[
        'trans_affine_label'
    ]
    # All meshes should have the same transformation matrix
    assert all(
        np.allclose(
            trans_affine,
            validation_set.get_data_element(i)[
                'trans_affine_label'
            ]
        )
        for i in range(len(validation_set))
    )
    template = MeshTemplate(
        mesh_label_names=list(training_set.mesh_label_names.keys()),
        trans_affine=trans_affine,
        **TEMPLATE_SPECS[hps['MESH_TEMPLATE_ID']]
    )

    ###### Training ######
    log.info("Initializing model %s...", hps['ARCHITECTURE'])
    model = ModelHandler[hps['ARCHITECTURE']].value(
        ndims=hps['NDIMS'],
        n_v_classes=hps['N_V_CLASSES'],
        n_m_classes=hps['N_M_CLASSES'],
        patch_size=hps['PATCH_SIZE'],
        **model_config
    )
    log.info("%d parameters in the model.", model.count_parameters())

    # Path of a previous experiment that might be used as a starting point for
    # training
    try:
        previous_exp_path = os.path.join(
            hps['EXPERIMENT_BASE_DIR'], hps['PREVIOUS_EXPERIMENT_NAME']
        )
    except TypeError:
        previous_exp_path = None

    # Init weights, scheduler state etc.
    (model,
     optim_state_dict,
     scheduler_state_dict,
     start_epoch) = _init_training_params(
         logger.get_experiment_dir(),
         resume,
         hps['PRE_TRAINED_MODEL_PATH'],
         model,
         previous_exp_path
    )

    # Evaluation during training on validation set
    evaluator = ModelEvaluator(
        eval_dataset=validation_set,
        mesh_template=template,
        save_dir=logger.get_eval_dir(),
        **hps_lower
    )

    world_size = len(hps['DEVICE'])
    solver_params = {
        'evaluator': evaluator,
        'save_path': logger.get_experiment_dir(),
        'mesh_template': template,
        **hps_lower,
    }
    solver_and_train_params = (
        world_size,
        hps['MASTER_PORT'],
        # Train params
        deepcopy(model),
        training_set,
        start_epoch,
        optim_state_dict,
        scheduler_state_dict,
        # Solver params
        solver_params,
        # All params
        hps
    )

    # Distributed or single-GPU training
    if world_size > 1:
        mp.spawn(
            distributed_training,
            args=solver_and_train_params,
            nprocs=world_size,
        )
    else:
        rank = int(hps['DEVICE'][0].split(":")[1])
        model = model.float().to(rank)
        solver = Solver(
            rank=rank,
            world_size=world_size,
            **solver_params
        )
        solver.train(
            model,
            training_set,
            start_epoch,
            optim_state_dict,
            scheduler_state_dict
        )

    log.info("Training finished.")

    return experiment_name


def _init_training_params(
    experiment_dir,
    resume,
    pretrained_path,
    model,
    previous_exp_path=None
):
    """ Load a model as a pretrained starting point or start a new training.

    returns: model, optim_state_dict, scheduler_state_dict, start_epoch
    """
    optim_state_dict, scheduler_state_dict, start_epoch = None, None, 1

    # First read pretrained model in order to freeze the correct parameters and
    # then optionally overwrite with trained weights from aborted training
    # (which might now be resumed).
    # ! It is important to first read the pretrained model in order to know
    # which parameters to freeze (potentially)!
    if pretrained_path is not None:
        # Load a pre-trained (sub-) model
        try:
            model.load_part(pretrained_path)
        except:
            # Try to infer name of pretrained model from previous experiment
            log.warning(
                "Model could not be loaded, trying with previous model"
                " directory."
            )
            pretrained_path = os.path.join(
                previous_exp_path,
                INTERMEDIATE_MODEL_NAME
            )
            model.load_part(pretrained_path)

    if resume:
        # Load state and epoch
        model_path = os.path.join(experiment_dir, INTERMEDIATE_MODEL_NAME)
        log.info("Loading model weights from %s...", model_path)
        (model,
         optim_state_dict,
         scheduler_state_dict,
         start_epoch) = load_checkpoint(model, model_path, 'cpu')
        epochs_file = os.path.join(experiment_dir, "models_to_epochs.json")
        with open(epochs_file, 'r') as f:
            models_to_epochs = json.load(f)
        if start_epoch == -1:
            start_epoch = models_to_epochs[INTERMEDIATE_MODEL_NAME] + 1
        log.info("Resuming training from epoch %d", start_epoch)

    return model, optim_state_dict, scheduler_state_dict, start_epoch
