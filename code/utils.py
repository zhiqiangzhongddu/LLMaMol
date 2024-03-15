from typing import List, Optional
import os
import random
import numpy as np
import pandas as pd
import argparse
import logging
from datetime import datetime
import time
import pytz
import torch
from pathlib import Path, PurePath
from scipy import spatial


project_root_path = PurePath(__file__).parent.parent


def is_notebook() -> bool:
    from IPython import get_ipython
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)

    except NameError:
        return False  # Probably standard Python interpreter


def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_output_folder(args):
    folder = "../output/{}/{}_{}".format(
        args.dataset, args.model, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    os.makedirs(folder)
    return folder


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def init_path(dir_or_file):
    path = os.path.dirname(dir_or_file) + '/'

    if not os.path.exists(path):
        mkdir_p(path)

    return dir_or_file


def get_root_logger(folder):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if folder:
        handler = logging.FileHandler(os.path.join(folder, "log.txt"))
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def save_info(args, folder, result):
    data = args.dataset
    split = args.split
    model = args.model
    date = folder.split("_")[-1]

    RES_FILE = '../output/res_%s.csv' % data
    df_res = pd.DataFrame.from_dict(
        {'model': [model],
         'split': [split],
         'date': [date],
         'performance': [result],
         }
    )
    if os.path.exists(RES_FILE):
        df_res.to_csv(RES_FILE, mode='a', header=False, index=False)
    else:
        df_res.to_csv(RES_FILE, header=True, index=False)


def ogbg_statistic(dataset):
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']

    print("Num. Graphs: {}, Avg. Nodes: {:.1f}, Avg. Edges: {:.1f}.".format(
        len(dataset), np.mean([data.num_nodes for data in dataset]), np.mean([data.num_edges for data in dataset])
    ))
    print("Num. Train: {}, Num. Valid: {}, Num. Test: {}.".format(
        train_idx.size()[0], val_idx.size()[0], test_idx.size()[0]
    ))
    print("Task Type: {}, Num. Tasks: {}, Num. Classe: {}.".format(
        dataset.task_type, dataset.num_tasks, dataset.num_classes
    ))


# * ============================= Time Related =============================


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Europe/Paris', t_format='%m-%d %H:%M:%S'):
    return datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print('Start running {} at {}'.format(
            func.__name__, get_cur_time()
        ))
        ret = func(*args, **kw)
        print(
            'Finished running {} at {}, running time = {}.'.format(
                func.__name__, get_cur_time(), time2str(time.time() - start_time)
            ))
        return ret

    return wrapper
