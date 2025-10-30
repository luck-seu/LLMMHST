import random
import torch
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
from graph_algo import normalize_adj_mx
import matplotlib.pyplot as plt
import sys
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)


    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def get_metadata():
    hg = HeteroData()
    hg['hw'].x = torch.randn(130, 3)
    hg['para'].x = torch.randn(13, 3)
    fout, fin = gen_in_out_edge(130, 13)
    hg['hw', 'fout', 'para'].edge_index = fout # [2, num_edges]
    hg['para', 'fin', 'hw'].edge_index = fin
    hg['para', 'connect1', 'para'].edge_index = gen_edge(13)
    hg['hw', 'connect2', 'hw'].edge_index = gen_edge(130)
    return hg.metadata()

def get_gso():
    adj_mx = np.load('jh_rn_adj.npy')
    adj_mx = adj_mx - np.eye(130)

    gso = normalize_adj_mx(adj_mx, 'scalap')[0]
    gso = torch.tensor(gso).to(device)
    return gso

class HeterogeneousDataset(Dataset):
    def __init__(self, hw_x, hw_y, para_x, para_y):
        self.hw_x = hw_x
        self.hw_y = hw_y
        self.para_x = para_x
        self.para_y = para_y

    def __len__(self):
        return len(self.hw_x)

    def __getitem__(self, index):
        hw_x, hw_y, para_x, para_y = self.hw_x[index], self.hw_y[index], self.para_x[index], self.para_y[index]
        hg = HeteroData()
        hg['hw'].x = torch.FloatTensor(hw_x).permute(1,0,2)
        hg['para'].x = torch.FloatTensor(para_x).permute(1,0,2)
        fout, fin = gen_in_out_edge(130, 13)
        hg['hw', 'fout', 'para'].edge_index = fout # [2, num_edges]
        hg['para', 'fin', 'hw'].edge_index = fin
        hg['para', 'connect1', 'para'].edge_index = gen_edge(13)
        hg['hw', 'connect2', 'hw'].edge_index = gen_edge(130)
        return hg, torch.FloatTensor(hw_x), torch.FloatTensor(hw_y), torch.FloatTensor(para_x)
    
def gen_edge(n):
    nodes = torch.arange(n) 
    edge_index = []
    if n == 10:
        t = 3
    else:
        t = 3
    for i in range(n):
        for j in range(n):
            if abs(i - j) <= t:
                edge_index.append([nodes[i], nodes[j]])
    edge_index = torch.tensor(edge_index).t().contiguous()
    return edge_index

def gen_in_out_edge(n1, n2):
    nodes1 = list(np.array([12,22,26,29,41,44,52,68,73,80,87,97,103]))
    nodes2 = list(range(0,n2))
    fout = torch.tensor([nodes1, nodes2])
    fin = torch.tensor([nodes2, nodes1])
    return fout, fin

def load_dataset(mode="train"):
    file_path = 'data/G60/'
    hw_ptr = np.load(file_path+'G60_high.npz',allow_pickle=True)
    para_ptr = np.load(file_path+'G60_para.npz',allow_pickle=True)

    hw_train_x = hw_ptr['hw_train_x']
    hw_train_y = hw_ptr['hw_train_y']
    hw_val_x = hw_ptr['hw_val_x']
    hw_val_y = hw_ptr['hw_val_y']
    hw_test_x = hw_ptr['hw_test_x']
    hw_test_y = hw_ptr['hw_test_y']
    para_train_x = para_ptr['para_train_x']
    para_train_y = para_ptr['para_train_y']
    para_val_x = para_ptr['para_val_x']
    para_val_y = para_ptr['para_val_y']
    para_test_x = para_ptr['para_test_x']
    para_test_y = para_ptr['para_test_y']

    # scaler
    hw_scaler = StandardScaler(mean=hw_ptr['mean'], std=hw_ptr['std'])
    para_scaler = StandardScaler(mean=para_ptr['mean'], std=para_ptr['std'])
    
    dataset = None
    if mode == "train":
        dataset = HeterogeneousDataset(hw_train_x, hw_train_y, para_train_x, para_train_y)
    elif mode == "val":
        dataset =  HeterogeneousDataset(hw_val_x, hw_val_y, para_val_x, para_val_y)
    else:
        dataset = HeterogeneousDataset(hw_test_x, hw_test_y, para_test_x, para_test_y)
    return dataset, hw_scaler, para_scaler
