
""" Utilities for distributed training. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import datetime

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def cleanup():
    """ Cleanup ddp process group. """
    dist.destroy_process_group()


def setup_ddp(rank, world_size, master_port=29500):
    """ Setup for ddp training.
    Attention: Multiple DDP trainings on the same server
    cannot use the same master port!
    """
    # Set environment vairables for distributed data parallel, see
    # https://pytorch.org/docs/stable/notes/ddp.html
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    # Needed to set timeout in dist.init_process_group
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    # Init process for ddp, nccl most recommended according to
    # https://pytorch.org/docs/1.9.0/generated/torch.nn.parallel.DistributedDataParallel.html
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        # Max. blocking 3h to ensure that validation on a single rank does not
        # lead to a crash
        timeout=datetime.timedelta(minutes=180),
    )


def prepare_dataloader(
    rank,
    world_size,
    batch_size,
    dataset,
    shuffle,
    pin_memory=False,
    num_workers=0
):
    """ Prepare dataloader for distributed training.

    As mentioned by
    https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51,
    pin_memory=False and num_workers=0 is recommended for distributed training.
    """
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        # shuffle is mutually exclusive with sampler (already contained in
        # sampler)
        shuffle = False
    else:
        sampler = None

    if batch_size > len(dataset):
        raise ValueError(
            "Batch size should not be larger than the size of the dataset."
        )

    # Dataloader with potentially distributed sampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=True,
        sampler=sampler,
        shuffle=shuffle,
    )

    return dataloader
