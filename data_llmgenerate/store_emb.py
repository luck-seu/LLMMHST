import torch
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader
from data_provider.data_loader_save import Dataset_G60, Dataset_G56
from storage.gen_prompt_emb import GenPromptEmb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    parser.add_argument("--data_path", type=str, default="G56_high")
    parser.add_argument("--num_nodes", type=int, default=100)
    parser.add_argument("--input_len", type=int, default=24*6)
    parser.add_argument("--output_len", type=int, default=24*6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=2048)
    parser.add_argument("--l_layers", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--divide", type=str, default="val")
    parser.add_argument("--num_workers", type=int, default=min(10, os.cpu_count()))
    return parser.parse_args()

def get_dataset(data_path, flag, input_len, output_len):
    datasets = {
        'G60': Dataset_G60,
        'G56': Dataset_G56,
    }
    dataset_class = datasets.get(data_path, Dataset_G56)
    return dataset_class(flag=flag, size=[input_len, 0, output_len], data_path=data_path)

def save_embeddings(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_set = get_dataset(args.data_path, 'train', args.input_len, args.output_len)
    test_set = get_dataset(args.data_path, 'test', args.input_len, args.output_len)
    val_set = get_dataset(args.data_path, 'val', args.input_len, args.output_len)

    data_loader = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'test': DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    }[args.divide]

    gen_prompt_emb = GenPromptEmb(
        device=device, 
        input_len=args.input_len,
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        layer=args.l_layers,
        divide=args.divide
    ).to(device)

    save_path = f"Embeddings/{args.data_path}/{args.divide}/"
    os.makedirs(save_path, exist_ok=True)

    emb_time_path = f"Results/emb_logs/"
    os.makedirs(emb_time_path, exist_ok=True)

    for i, (x, y, x_mark, y_mark) in enumerate(data_loader):
        embeddings = gen_prompt_emb.generate_embeddings(x.to(device), x_mark.to(device))
        file_path = f"{save_path}{i}.h5"
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('embeddings', data = embeddings.cpu().numpy())

if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings(args)
    t2 = time.time()
    print(f"Total time spent: {(t2 - t1)/60:.4f} minutes")