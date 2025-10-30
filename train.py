from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
import pytorch_lightning as pl
import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch_geometric.loader import DataLoader
from model import LLMMUST
from utils import print_log
from HWDataset import get_metadata, load_dataset

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pl.seed_everything(295)

parser = ArgumentParser()
parser.add_argument('--seq_len', type=int, default=6, help="Time input length.")
parser.add_argument('--horizen', type=int, default=6, help="Time output length.")
parser.add_argument('--in_channels', type=int, default=3, help="The dimension of inputs.")
parser.add_argument('--out_channels', type=int, default=1, help="The dimension of outputs.")
parser.add_argument('--data', type=str, default='G56', help="Which data to use.")
parser.add_argument('--hidden_dim', type=int, default=64, help="The hidden dimension of models.")
parser.add_argument('--temporal_dim', type=int, default=32, help="Time embedding dimension.")
parser.add_argument('--align_dim', type=int, default=512, help="The align dimension of models.")
parser.add_argument('--num_heads', type=int, default=4, help='The num heads of attention.')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout prob.')
parser.add_argument('--num_layers', type=int, default=1, help="Num of Transformer encoder layers.")
parser.add_argument('--only_gwnet', action='store_true', help="Whether use the pretrained gwnet.")
parser.add_argument('--only_thgt', action='store_true', help="Whether use the thgt to prompt.")
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
parser.add_argument('--batch_size', type=int, default=8, help="Batch size.")
parser.add_argument('--clip_val', type=float, default=5, help="Gradient clipping values. ")
parser.add_argument('--total_epochs', type=int, default=30, help="Max epochs of model training.")
parser.add_argument('--lr_decay_step', type=int, default=100, help="Learning rate decay step size.")
parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help="Learning rate decay rate.")
parser.add_argument('--gpu_lst', nargs='+', type=int, help="Which gpu to use.")
parser.add_argument('--accumulate_grad_batches', type=int, default=8, help="Gradient accumulation steps.")
args = parser.parse_args()

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.cuda.empty_cache()

def train():
    train_dataset, hw_scaler, _ = load_dataset("train")
    train_dataset = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    valid_dataset, _, _ = load_dataset("val")
    valid_dataset = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_dataset, _, _ = load_dataset("test")
    test_dataset = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    pre_HSTG = LLMMUST(in_channels=[[3],[64,64,64],[args.seq_len]], trainmode='base', batch=args.batch_size, seq_len=args.seq_len, horizen=args.horizen, scaler=hw_scaler, num_nodes=[100,10], metadata=get_metadata(), lr=args.lr,
                       weight_decay=args.weight_decay, lr_decay_step=args.lr_decay_step, lr_decay_gamma=args.lr_decay_gamma, is_large_label=False)
    print("agrs: ", args)
    print(pre_HSTG)

    checkpoint_callback = ModelCheckpoint(monitor="validation_epoch_average",
                                          filename='G56_res' + '-{epoch:03d}-{validation_epoch_average:.4f}',
                                          save_top_k=5,
                                          mode='min',
                                          save_last=True)

    early_stop_callback = EarlyStopping(
        monitor='validation_epoch_average', 
        min_delta=0,  
        patience=20, 
        verbose=True, 
        mode='min'
        )

    trainer = pl.Trainer(callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='epoch'), early_stop_callback],
                        gradient_clip_val=args.clip_val, 
                        max_epochs=args.total_epochs, 
                        devices=[0], 
                        accelerator='gpu',
                        accumulate_grad_batches=args.accumulate_grad_batches,
                        log_every_n_steps=1
                        )
    trainer.fit(pre_HSTG, train_dataloaders=train_dataset, val_dataloaders=valid_dataset)
    res = trainer.test(pre_HSTG, test_dataset, ckpt_path='best')
    res = trainer.test(pre_HSTG, test_dataset)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    log_path = f'logs/{args.data}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f'{args.data}-{log_time}.log')
    log = open(log, 'a')
    log.seek(0)
    log.truncate()

    print_log(res, log=log)

if __name__ == "__main__":
    if args.data == 'G56':
        from HWDataset import get_metadata, load_dataset
    elif args.data == 'G60':
        from JHDataset import get_metadata, load_dataset
    train()
