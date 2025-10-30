import torch
import numpy as np
import datetime
import os

def print_log(*values, log=None, end='\n'):
    print(*values, end=end)

    if log:
        if isinstance(log, str):
            log = open(log, 'a')
        print(*values, file=log, end=end)
        log.flush()

def get_graph_dict(data):
    if data=='G56':
        dict = torch.load('data/G56/graph_info.pt')
    elif data=='G60':
        dict = torch.load('data/G60/graph_info.pt')
    else:
        print("Error choice in graph loading!")
        return {}
    return dict

def make_log():
    log_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    log_path = f'logs/G56'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f'G56-{log_time}.log')
    log = open(log, 'a')
    log.seek(0)
    log.truncate()