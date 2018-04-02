""" Utility """

import datetime
import logging
import os

def init_logger(log_dir, file_name, level):
    """ Initialise logger """
    ensure_dir(log_dir)
    log_file = os.path.join(log_dir, file_name+'_'+get_now_str()+'.log')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=level,
        format='%(asctime)s [%(levelname)s][%(message)s]'
    )

def get_now_str():
    """ get current time in %Y-%m-%d_%H-%M format """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M")

def ensure_dir(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)