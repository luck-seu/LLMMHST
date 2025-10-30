import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_G60, Dataset_G56
from models.STLLMGen import Dual
from utils.metrics import metric

from matplotlib import rcParams

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="G56_high", help="data path")
    parser.add_argument("--channel", type=int, default=64, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=100, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=144, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=144, help="out_len")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.1, help="dropout rate of neural network layers")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=2, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=2, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--model_path", type=str, default='best_model.pth', help="path to the saved model")
    return parser.parse_args()

def load_data(args):
    data_map = {
        'G60_high': Dataset_G60,
        'G56_high': Dataset_G56,
    }
    data_class = data_map.get(args.data_path, Dataset_G56)
    test_set = data_class(flag='test', scale=True, size=[args.seq_len, 0, args.pred_len], data_path=args.data_path)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=10)
    return test_loader

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_loader = load_data(args)
    
    model = Dual(
        device=device,
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head
    )

    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    print("开始测试...")

    test_outputs = []

    with torch.no_grad():
        for iter, (x, y, x_mark, y_mark, embeddings) in enumerate(test_loader):
            testx = torch.Tensor(x).to(device)
            testx_mark = torch.Tensor(x_mark).to(device)
            test_embedding = torch.Tensor(embeddings).to(device)
            
            preds = model(testx, testx_mark, test_embedding)
            test_outputs.append(preds)

    test_pre = torch.cat(test_outputs, dim=0)

    test_set = test_loader.dataset
    selected = test_pre[:20]

    selected_denorm = test_set.inverse_transform(selected.cpu().numpy())
    flattened_denorm = selected_denorm.reshape(-1, test_pre.shape[2])

    file_path = "data/G56_high_pre.csv"
    df = pd.read_csv(file_path)
    node_columns = [col for col in df.columns if col != 'date']
    node_data = df[node_columns].values
    num_samples = node_data.shape[0] - args.seq_len + 1
    sliding_data = np.zeros((num_samples, args.seq_len, args.num_nodes))
    for i in range(num_samples):
        sliding_data[i] = node_data[i:i+args.seq_len, :]
    final_data = sliding_data.reshape(num_samples * args.seq_len, args.num_nodes)

    combined_data = np.concatenate([flattened_denorm, final_data[:args.seq_len*6]], axis=0)
    print("拼接后的数据形状:", combined_data.shape)

    np.savez('../data/G56/G56_llm.npz', data=combined_data)

if __name__ == "__main__":
    main()

