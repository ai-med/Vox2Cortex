
""" Logging a single experiment. This module is designed to handle a single
experiment at a time."""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import time
import logging
import functools
import copy
import collections
import inspect
import json

import torch
import wandb
import pandas as pd
import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt

LOG_FILE_NAME = "v2o.log"
PARAM_FILE_NAME = "params.json"
TIME_FILE_NAME = "times.txt"
EVAL_DIR_NAME = "validation"

file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
)
console_formatter = logging.Formatter(
    "[%(levelname)s] %(message)s"
)
time_logger = None

_wandb_run = None
_debug = True
_log_dir = "logs/"
_log_file = os.path.join(_log_dir, LOG_FILE_NAME)
_experiment_base_dir = "../experiments"
_experiment_name = ""
_root_logger_level = logging.DEBUG
_hps_log = {}
_wandb_user = None


if not os.path.isdir(os.path.join(_experiment_base_dir, _log_dir)):
    os.makedirs(os.path.join(_experiment_base_dir, _log_dir))


def wandb_is_active():
    return _wandb_run is not None


def wandb_test_summary(
    split: str,
    results: pd.DataFrame,
    user: str,
    project: str,
    exp_name: str
):
    """ Send test results to wandb """
    api = wandb.Api()
    run = api.run(f"{user}/{project}/{exp_name}")

    log_results = df_to_wandb_log(results)
    for k, v in log_results.items():
        run.summary[split + "." + k] = v

    run.summary.update()


def df_to_wandb_log(data: pd.DataFrame):
    """ Convert results DataFrame to dict that can be logged with wandb
    """

    val_logs = {}

    # Results per organ
    data_summary = data.groupby(
        ['Metric', 'Tissue']
    ).mean()
    data_summary = data_summary.reset_index()
    for _, row in data_summary.iterrows():
        key = "_".join([row['Tissue'], row['Metric']])
        val_logs[key] = row['Value']
    # Mean over organs
    data_summary = data.groupby(['Metric']).mean()
    data_summary = data_summary.reset_index()
    for _, row in data_summary.iterrows():
        key = "mean_" + row['Metric']
        val_logs[key] = row['Value']

    return val_logs


def string_dict(d: dict):
    """
    Make dict jsonable.

    :param dict d: The dict that should be made serializable/writable.
    :returns: The dict with objects converted to their names.
    """
    u = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            # Dicts
            u[k] = string_dict(u.get(k, {}))
        elif isinstance(v, collections.abc.MutableSequence):
            # Lists
            u[k] = string_list(u.get(k, []))
        elif inspect.isclass(v) or inspect.isfunction(v):
            # Class or function
            u[k] = v.__name__
        elif isinstance(v, tuple):
            # Tuple
            u[k] = string_list(u.get(k, []))
        elif not is_jsonable(v):
            # Non-trivial objects
            u[k] = str(v)
    return u


def string_list(l: list):
    """
    Make list jsonable.

    :param list l: The list that should be made serializable/writable.
    :returns: The list with objects converted to their names.
    """
    u = []
    for e in l:
        if inspect.isclass(e) or inspect.isfunction(e):
            # Class or function
            u.append(e.__name__)
        elif isinstance(e, collections.abc.MutableSequence):
            # List
            u.append(string_list(list(e)))
        elif isinstance(e, tuple):
            # Tuple
            u.append(string_list(e))
        elif not is_jsonable(e):
            # Non-trivial objects
            u.append(str(e))
        else:
            u.append(e)
    return u


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def get_experiment_dir():
    """ Get the current experiment directory.
    """
    return os.path.join(_experiment_base_dir, _experiment_name)

def get_eval_dir():
    eval_dir = os.path.join(get_experiment_dir(), EVAL_DIR_NAME)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    return eval_dir

def set_eval_dir_name(name: str):
    """ Set the name for the eval directory.
    """
    global EVAL_DIR_NAME
    EVAL_DIR_NAME = name

def get_params_file():
    """ Return the path to the params file of the experiment.
    """
    return os.path.join(get_experiment_dir(), PARAM_FILE_NAME)


def get_std_logfile():
    """ The main log file for an experiment.
    """
    return os.path.join(get_log_dir(), LOG_FILE_NAME)


def get_experiment_hps():
    """ Get the experiment parameters.
    """
    with open(get_params_file(), 'r') as f:
        current_hps = json.load(f)
    return current_hps


def save_params(hps: dict, write=True):
    """ Save the parameter dict.
    """
    global _hps_log
    hps_to_write = string_dict(hps)
    _hps_log = hps_to_write
    if write:
        param_file = get_params_file()
        with open(param_file, 'w') as f:
            json.dump(hps_to_write, f)


def check_configs_equal(hps: dict):
    """ Check if a config (hps) is equal to the current experiment config and
    return the parameter names and values that differ. The returned dict has
    form:
        key: (stored value, hps_value)
    """
    differ = {}

    # Current experiment config
    with open(get_params_file(), 'r') as f:
        current_hps = json.load(f)

    # Check if configs are equal
    hps_new = string_dict(hps)
    for k_file, v_file in current_hps.items():
        if hps_new[k_file] != v_file:
            differ[k_file] = (v_file, hps_new[k_file])

    return differ


def init_experiment(
    experiment_base_dir: str=None,
    experiment_name: str=None,
    prefix: str=None,
    exist_ok=False,
    create=True,
    log_time=False,
    log_level="DEBUG",
):
    """ Create experiment directory and set the experiment name automatically
    if not specified. If 'experiment_name' is 'debug', an existing directory
    might be overwritten.

    returns a new experiment name if 'experiment_name' is None and a new loglevel
    if 'experiment_name' is 'debug'
    """
    global _experiment_base_dir
    global _experiment_name
    global time_logger

    # Base directory for experiment dirs
    if experiment_base_dir is not None:
        _experiment_base_dir = experiment_base_dir

    # Specific experiment dir
    if experiment_name is not None:
        _experiment_name = experiment_name
    else:
        # Automatically enumerate experiments exp_i
        ids_exist = []
        for n in os.listdir(_experiment_base_dir):
            try:
                ids_exist.append(int(n.split("_")[-1]))
            except ValueError:
                pass
        if len(ids_exist) > 0:
            new_id = np.max(ids_exist) + 1
        else:
            new_id = 1

        experiment_name = prefix + str(new_id)

    # Save name of current experiment
    _experiment_name = experiment_name

    # Create directories
    log_dir = get_log_dir()
    if create:
        os.makedirs(log_dir, exist_ok=exist_ok)
    elif not exist_ok and os.path.isdir(log_dir):
        raise RuntimeError(f"{log_dir} already exists.")

    if log_time:
        time_logger = get_time_logger()

    global _debug
    global _root_logger_level
    if experiment_name == "debug":
        _debug = True
        _root_logger_level = "DEBUG"
    else:
        _root_logger_level = log_level
        _debug = False

    logging.root.setLevel(_root_logger_level)

    return experiment_name, _root_logger_level


def reset_root_logger_level():
    global _root_logger_level
    _root_logger_level = get_experiment_hps()['LOGLEVEL']
    logging.root.setLevel(_root_logger_level)


def get_log_level():
    return _root_logger_level


def set_log_dir(log_dir: str):
    global _log_dir
    global _log_file
    _log_dir = log_dir
    _log_file = os.path.join(_log_dir, LOG_FILE_NAME)


def get_console_handler():
    """ Logging to console.
    """
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.name = "ConsoleHandler"

    return console_handler


def get_file_handler():
    """ Logging to file.
    """
    file_handler = logging.FileHandler(get_std_logfile(), mode='a')
    file_handler.setFormatter(file_formatter)
    file_handler.name = "FileHandler"

    return file_handler


def add_handlers(logger, handler_list: list):
    """ Add handlers to a logger if they do not yet exist.
    """
    existing_handler_names = []
    for existing_handler in logger.handlers:
        existing_handler_names.append(existing_handler.name)

    for new_handler in handler_list:
        if new_handler.name not in existing_handler_names:
            logger.addHandler(new_handler)


def get_std_logger(logger_name):
    """ Generate a std logger for console and file output. It is
    best-practice to create a child logger per module.
    """
    logger = logging.getLogger(logger_name)
    console_handler = get_console_handler()
    file_handler = get_file_handler()
    add_handlers(logger, [console_handler, file_handler])
    logger.propagate = False

    return logger


def get_time_logger():
    """ A logger specifically for logging time measurements.
    """
    log = logging.getLogger("TIME")
    for handler in log.handlers[:]:
        log.removeHandler(handler)
    log.setLevel('DEBUG')
    time_file = os.path.join(get_log_dir(), TIME_FILE_NAME)
    fileHandler = logging.FileHandler(time_file, mode='a')
    log.addHandler(fileHandler)

    return log


def init_wandb_run(**wandb_args):
    """ Init a wandb run.

    :params wandb_args witll be passed to wandb.init()
    """
    global _wandb_run

    # Finish run if it already exists
    finish_wandb_run()

    # New run
    _wandb_run = wandb.init(**wandb_args)


def set_loglevel(level):
    """ Set root log level which is inherited by all child loggers.
    """
    logging.root.setLevel(level)


def get_log_dir():
    return os.path.join(get_experiment_dir(), _log_dir)


def init_wandb(**wandb_args):
    """ Initialization for logging with wandb
    """
    global _wandb_run

    _wandb_run = wandb.init(**wandb_args)


def finish_wandb_run():
    global _wandb_run
    if _wandb_run is not None:
        _wandb_run.finish()
        _wandb_run = None
    else:
        pass


def write_array_if_debug(data_1, data_2):
    """ Write data if debug mode is on.
    """
    file_1 = "../misc/array_1.npy"
    file_2 = "../misc/array_2.npy"
    if _debug:
        np.save(file_1, data_1)
        np.save(file_2, data_2)


def write_img_if_debug(img: np.ndarray, path: str):
    """ Write data if debug mode is on.
    """
    if _debug:
        img = nib.Nifti1Image(img, np.eye(4))
        nib.save(img, path)


def write_scatter_plot_if_debug(points, path: str):
    """ Write a screenshot of a 3d scatter plot """
    if isinstance(points, torch.Tensor):
        points = points.cpu()
    if _debug:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0],
                   points[:,1],
                   points[:,2])
        plt.savefig(path)
        plt.close()


def debug():
    return _debug


def measure_time(func):
    """ Decorator for time measurement """
    @functools.wraps(func)
    def time_wrapper(*args, **kwargs):
        if time_logger is not None:
            tic = time.perf_counter()
            return_value = func(*args, **kwargs)
            toc = time.perf_counter()
            time_elapsed = toc - tic
            time_logger.info(
                "Function %s takes %.5f s", func.__qualname__, time_elapsed
            )
        else:
            return_value = func(*args, **kwargs)

        return return_value

    return time_wrapper
