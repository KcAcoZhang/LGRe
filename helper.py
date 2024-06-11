import torch
import logging, logging.config
import sys
import json
import numpy as np

np.set_printoptions(precision=4)


def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object

    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read

    Returns
    -------
    A logger object which writes to both file and stdout

    """
    config_dict = json.load(open(config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


def get_param(shape):
    param = torch.nn.Parameter(torch.Tensor(*shape))
    torch.nn.init.xavier_normal_(param.data)
    return param
