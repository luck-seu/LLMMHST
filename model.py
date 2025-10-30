import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import math
from torch_geometric.loader import DataLoader
from metrics import masked_rmse, masked_mae, masked_mape
from utils import get_graph_dict
from typing import Tuple
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from HWDataset import load_dataset
# load hetero graph
graph_dict = get_graph_dict('G56')

node_type = graph_dict['node_type'].to(device)
edge_path_type = graph_dict['edge_path_type'].to(device)
edge_path_len = graph_dict['edge_path_len'].to(device)
mask = graph_dict['mask'].to(device)
edge_path_type_r = graph_dict['edge_path_type_r'].to(device)
edge_path_len_r = graph_dict['edge_path_len_r'].to(device)
mask_r = graph_dict['mask_r'].to(device)

def rmse(y_pred, y_true):
    mask_value = torch.tensor(0)
    if y_true.min() < 1:
        mask_value = y_true.min()
    return masked_rmse(y_pred, y_true, mask_value)

def mae(y_pred, y_true):
    mask_value = torch.tensor(0)
    if y_true.min() < 1:
        mask_value = y_true.min()
    return masked_mae(y_pred, y_true, mask_value)

def mape(y_pred, y_true):
    mask_value = torch.tensor(0)
    if y_true.min() < 1:
        mask_value = y_true.min()
    return masked_mape(y_pred, y_true, mask_value)

class GTU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GTU, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.conv = CausalConv2d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=(1, kernel_size),
            enable_padding=True  
        )
        
        self.w_conv = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))
        self.align = Align(in_channels, out_channels)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) 

        x_conv = self.conv(x) 
        x_p = x_conv[:, :self.out_channels, :, :] 
        x_q = x_conv[:, -self.out_channels:, :, :] 

        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q)) 
        w = torch.sigmoid(self.w_conv(x)).mean(-1).unsqueeze(-1)  
        return w * x_gtu + (1 - w) * x_gtu

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, node_num, timestep = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, node_num, timestep]).to(x)], dim=1)
        else:
            x = x
        return x

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim, 1)

    def forward(self, outputs):
        attn_weights = F.softmax(self.attn(outputs), dim=1)
        weighted = torch.mul(outputs, attn_weights)
        representations = weighted.sum(dim=1)
        return representations

class MSWT(nn.Module):
    def __init__(self, in_channels, d, num_scales=3):
        super(MSWT, self).__init__()
        self.d = d
        self.num_scales = num_scales

        self.gtu_scales = nn.ModuleList([GTU(in_channels, d, scale) for scale in range(1, num_scales + 1)])

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d, nhead=1, dim_feedforward=4 * d)
            for _ in range(num_scales)
        ])
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d, num_heads=1)

    def forward(self, node_features):
        B, N, T, in_channels = node_features.shape
        assert in_channels == self.gtu_scales[0].conv.in_channels, \
            f"Input channels {in_channels} do not match GTU's in_channels"

        outputs = []
        for gtu in self.gtu_scales:
            scale_output = gtu(node_features) 
            outputs.append(scale_output)

        transformer_outputs = []
        for i, output in enumerate(outputs):
            reshaped_output = output.permute(2, 0, 1, 3).reshape(T, B*N, self.d)
            transformed = self.transformer_layers[i](reshaped_output)
            transformed = transformed.reshape(T, B, N, self.d).permute(1, 2, 0, 3)
            transformer_outputs.append(transformed)
        
        stacked_output = torch.cat(transformer_outputs, dim=1) 

        stacked_output = stacked_output.permute(2, 1, 0, 3)
        stacked_output = stacked_output.reshape(T, -1, self.d)
        attn_output, _ = self.multihead_attn(stacked_output, stacked_output, stacked_output)
        attn_output = attn_output.reshape(T, self.num_scales, N, B, self.d)
        attn_output = attn_output.permute(3, 2, 1, 0, 4) 

        final_output = attn_output.mean(dim=2) 

        return final_output

class CHGAN(nn.Module):
    def __init__(self, ntype, d, etype, num_heads, lambda_decay=0.5):
        super(CHGAN, self).__init__()

        self.ntype = ntype
        self.d = d
        self.etype = etype
        self.num_heads = num_heads
        self.lambda_decay = lambda_decay
        self.d_head = d // num_heads 
        
        assert d % num_heads == 0, "d must be divisible by num_heads"

        self.Q = nn.Parameter(torch.randn(ntype, num_heads, self.d_head, self.d_head))
        self.K = nn.Parameter(torch.randn(ntype, num_heads, self.d_head, self.d_head))
        self.V = nn.Parameter(torch.randn(ntype, num_heads, self.d_head, self.d_head))

        self.E = nn.Embedding(etype, d, padding_idx=0)  
        self.edge_bias_linear = nn.Linear(d, 1)

        self.output_linear = nn.Linear(d * 2, d)  
        
    def forward_attention(self, node_features, node_type, edge_path_type, edge_path_len, mask):
        """
        Attention calculation (shared between forward and reverse)
        """
        N, T, _ = node_features.shape 
        device = node_features.device
        node_type = node_type.to(device)
        edge_path_type = edge_path_type.to(device)
        edge_path_len = edge_path_len.to(device)
        mask = mask.to(device)

        Q_mat = self.Q[node_type]
        K_mat = self.K[node_type] 
        V_mat = self.V[node_type] 

        node_features_reshaped = node_features.view(N, T, self.num_heads, self.d_head)  
        node_features_reshaped = node_features_reshaped.permute(0, 2, 1, 3) 
        Q = torch.einsum('nhtd,nhdc->nhtc', node_features_reshaped, Q_mat)  
        K = torch.einsum('nhtd,nhdc->nhtc', node_features_reshaped, K_mat) 
        V = torch.einsum('nhtd,nhdc->nhtc', node_features_reshaped, V_mat) 

        Q = Q.permute(1, 2, 0, 3)
        K = K.permute(1, 2, 0, 3)
        V = V.permute(1, 2, 0, 3)

        a = torch.einsum('hnik,hnjk->hnij', Q, K) / (self.d_head ** 0.5)

        E_features = self.E(edge_path_type) 
        edge_bias = E_features.mean(dim=-2)
        edge_bias = self.edge_bias_linear(edge_bias).squeeze(-1)
        a += edge_bias.unsqueeze(0)

        decay = torch.exp(self.lambda_decay * (edge_path_len - 1)) 
        a = a * decay.unsqueeze(0) 

        a = a.masked_fill(mask.unsqueeze(0) == 0, float('-inf'))
        a = F.softmax(a, dim=-1) 

        output = torch.einsum('hnij,hnjk->hnik', a, V)  
        output = output.permute(2, 1, 0, 3).contiguous()  
        output = output.view(N, T, -1) 
        return output

    def forward(self, node_features):
        device = node_features.device
        self.Q = self.Q.to(device)
        self.K = self.K.to(device)
        self.V = self.V.to(device)
        self.E = self.E.to(device)

        output_f = self.forward_attention(
            node_features, node_type, 
            edge_path_type, edge_path_len, mask
        )

        output_r = self.forward_attention(
            node_features, node_type,
            edge_path_type_r, edge_path_len_r, mask_r
        )

        output = torch.cat([output_f, output_r], dim=-1) 

        output = self.output_linear(output) 

        output += node_features  

        return output


class MutliScaleDiffGTU(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MutliScaleDiffGTU, self).__init__()

        self.gtu1 = GTU(in_channels, out_channels, 1)
        self.gtu2 = GTU(in_channels, out_channels, 2)
        self.gtu3 = GTU(in_channels, out_channels, 3)
        self.pooling = nn.MaxPool2d(kernel_size=(1,5), ceil_mode=False)
        

    def forward(self, hw_x):
        x = torch.cat([self.gtu1(hw_x),
                        self.gtu2(hw_x),
                        self.gtu3(hw_x)],dim=-1) 
        x = self.pooling(x)
        return x

class MSDTHGTEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, num_nodes, metadata):
        super(MSDTHGTEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.num_nodes = num_nodes

        self.mswt1 = MSWT(in_channels=in_channels, d=out_channels)
        self.chgan = CHGAN(ntype=2, d=out_channels, etype=3, num_heads=4)
        self.mswt2 = MSWT(in_channels=out_channels, d=out_channels)

        self.relu = nn.ReLU()
        self.ln1 = nn.LayerNorm([num_nodes[0], out_channels])
        self.ln2 = nn.LayerNorm([num_nodes[1], out_channels])
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x_dict, edge_index_dict, hw_x):
        init_x_dict = {k:v for k,v in x_dict.items()}

        for key in x_dict:
            x = x_dict[key].reshape(hw_x.size(0), -1, self.seq_len, self.in_channels)
            x_dict[key] = self.mswt1(x).reshape(-1, self.seq_len, self.out_channels)
        x_values = []
        for k in x_dict:
            x_values.append(x_dict[k].reshape(hw_x.size(0), -1, self.seq_len, self.out_channels))
        x = torch.cat(x_values, dim=1)

        new_x = torch.zeros((x.size(0), x.size(1), x.size(2), self.out_channels))
        for b in range(x.size(0)):
            x_b = x[b]
            x_b = self.chgan(x_b.reshape(-1, self.seq_len, self.out_channels))
            new_x[b:b+1] = x_b
        x = new_x.to(device)

        start_idx = 0
        for key in x_dict:
            end_idx = start_idx + x_dict[key].size(0)//hw_x.size(0)
            x_dict[key] = x[:, start_idx:end_idx, :, :].reshape(hw_x.size(0), -1, self.seq_len, self.out_channels).to(device)

            x_dict[key] = self.mswt2(x_dict[key]).reshape(-1, self.seq_len, self.out_channels)
            
            start_idx = end_idx
            if x_dict[key].size() == init_x_dict[key].size():
                x_dict[key] = x_dict[key] + init_x_dict[key]
            h = x_dict[key].reshape(hw_x.size(0), -1, self.seq_len, self.out_channels).to(device)
            if h.size(1) == self.num_nodes[0]:
                h = self.ln1(h.permute(0,2,1,3)).permute(0,2,1,3)
            else:
                h = self.ln2(h.permute(0,2,1,3)).permute(0,2,1,3)
            x_dict[key] = self.dropout(h).reshape(-1, self.seq_len, self.out_channels)

        return x_dict


class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1)) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, groups=groups, bias=bias)

    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)
        return result

class LLMMUST(pl.LightningModule):
    def __init__(self, in_channels, trainmode, batch, seq_len, horizen, scaler, num_nodes, metadata,
                lr, weight_decay, lr_decay_step, lr_decay_gamma, is_large_label):
        super(LLMMUST, self).__init__()
        self.trainmode = trainmode
        self.batch_size = batch
        self.out_channel = in_channels[-2][-1]
        self.seq_len = seq_len
        self.horizen = horizen
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.scalar = scaler
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_step = lr_decay_step
        self.lr_decay_gamma = lr_decay_gamma
        self.is_large_label = is_large_label
        self.save_hyperparameters()

        modules = [MSDTHGTEncoder(in_channels[0][0],in_channels[1][0],self.seq_len*2, num_nodes=num_nodes, metadata=metadata)]
        modules.extend([MSDTHGTEncoder(in_channels[1][i],in_channels[1][i+1],self.seq_len*2, num_nodes=num_nodes,metadata=metadata) for i in range(len(in_channels[1])-1)])
        self.stg_blocks = nn.ModuleList(modules)
        
        self.out_linear = nn.Conv2d(in_channels[-1][-1]*(len(in_channels[1])), 128, kernel_size=(1, in_channels[-2][-1]))
        self.final_fc = nn.Linear(128, in_channels[-1][-1])
        self.ln = nn.Sequential(nn.ReLU(), nn.LayerNorm(128), nn.Dropout(0.3))
        self.projection = nn.Linear(2 * self.seq_len, self.seq_len)

        self.series_size = seq_len
        self.series_tensor_hw = self.init_weight((1, self.num_nodes[0], self.series_size, in_channels[0][0]))
        self.series_tensor_para = self.init_weight((1, self.num_nodes[1], self.series_size, in_channels[0][0]))

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_step_res = []
        self.test_step_res_all = []

    def init_weight(self, size: Tuple):
        weight = torch.empty(size, requires_grad=True)
        weight = torch.nn.init.kaiming_normal_(weight)
        return torch.nn.Parameter(weight, requires_grad=True)
    
    def to_device(self, data, device):
        if isinstance(data, dict):
            return {key: self.to_device(value, device) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(item, device) for item in data)
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        return data
    
    def forward(self, x_dict, edge_index_dict, hw_x):

        B = hw_x.shape[0]
        series_tensor_hw = self.series_tensor_hw.expand(B, -1, -1, -1).reshape(-1, self.series_size, self.in_channels[0][0])
        series_tensor_para = self.series_tensor_para.expand(B, -1, -1, -1).reshape(-1, self.series_size, self.in_channels[0][0])
        
        # encoder1
        x_dict_1 = x_dict.copy()
        x_dict_1['hw'] = torch.cat((x_dict_1['hw'], series_tensor_hw), dim=1)
        x_dict_1['para'] = torch.cat((x_dict_1['para'], series_tensor_para), dim=1)
        # encoder2
        x_dict_2 = x_dict.copy()
        augmentor2 = SequenceAugmentor(noise_std=0.05)
        x_dict_2['hw'][:, :, 0] = augmentor2.augment(x_dict_2['hw'][:, :, 0])
        x_dict_2['para'][:,:,0] = augmentor2.augment(x_dict_2['para'][:, :, 0])
        x_dict_2['hw'] = torch.cat((x_dict_2['hw'], series_tensor_hw), dim=1)
        x_dict_2['para'] = torch.cat((x_dict_2['para'], series_tensor_para), dim=1)
        # encoder3
        x_dict_3 = x_dict.copy()
        augmentor3 = SequenceAugmentor(noise_std=0.2)
        if self.trainmode == 'base':
            x_dict_3['hw'][:, :, 0] = augmentor3.augment(x_dict_3['hw'][:, :, 0])
            x_dict_3['para'][:, :, 0] = augmentor3.augment(x_dict_3['para'][:, :, 0])
        else:
            train_dataset, hw_scaler, _ = load_dataset("train")
            train_dataset = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16, pin_memory=True)
            train_data = next(iter(train_dataset))
            hgs, hw_x, _, _ = train_data
            x_dict_3['hw'][:,:,0] = hgs.x_dict['hw'][:hw_x.shape[0]*self.num_nodes[0],:,0]
            x_dict_3['para'][:,:,0] = hgs.x_dict['para'][:hw_x.shape[0]*self.num_nodes[1],:,0]
        x_dict_3['hw'] = torch.cat((x_dict_3['hw'], series_tensor_hw), dim=1)
        x_dict_3['para'] = torch.cat((x_dict_3['para'], series_tensor_para), dim=1)

        need_concat_1 = []
        need_concat_2 = []
        need_concat_3 = []

        for encoder in self.stg_blocks:
            x_dict_1 = encoder(x_dict_1, edge_index_dict, hw_x)
            x_dict_2 = encoder(x_dict_2, edge_index_dict, hw_x)
            x_dict_3 = encoder(x_dict_3, edge_index_dict, hw_x)

            need_concat_1.append(x_dict_1['hw'].reshape(hw_x.size(0), -1, self.seq_len*2, self.out_channel)) 
            need_concat_2.append(x_dict_2['hw'].reshape(hw_x.size(0), -1, self.seq_len*2, self.out_channel))
            need_concat_3.append(x_dict_3['hw'].reshape(hw_x.size(0), -1, self.seq_len*2, self.out_channel))

        need_concat_1 = [self.projection(x.permute(0,1,3,2)).permute(0,1,3,2) for x in need_concat_1] 
        need_concat_2 = [self.projection(x.permute(0,1,3,2)).permute(0,1,3,2) for x in need_concat_2]
        need_concat_3 = [self.projection(x.permute(0,1,3,2)).permute(0,1,3,2) for x in need_concat_3]
           
        final_x_1 = torch.cat(need_concat_1, dim=-2).permute(0, 2, 1, 3) 
        final_x_2 = torch.cat(need_concat_2, dim=-2).permute(0, 2, 1, 3)
        final_x_3 = torch.cat(need_concat_3, dim=-2).permute(0, 2, 1, 3)

        out1 = self.out_linear(final_x_1)[:, :, :, -1].permute(0, 2, 1)
        out2 = self.out_linear(final_x_2)[:, :, :, -1].permute(0, 2, 1)
        out3 = self.out_linear(final_x_3)[:, :, :, -1].permute(0, 2, 1)

        out1 = self.ln(out1)
        out2 = self.ln(out2)
        out3 = self.ln(out3)
        pre = self.final_fc(out1).permute(0,2,1)
        return pre, out2, out3
    
    def training_step(self, batch, batch_idx):
        hgs, hw_x, hw_y, para_x = batch
        pre, out2, out3 = self(hgs.x_dict, hgs.edge_index_dict, hw_x)

        pre, label = self._inverse_transform([pre, hw_y[:,:,:,0]], self.scalar)

        out2 = out2.reshape(hw_x.shape[0], -1)
        out3 = out3.reshape(hw_x.shape[0], -1)
        loss1 = torch.mean(torch.abs(pre - label)) 
        loss2 = self.SimCLRLoss(out2, out3, hw_x.shape[0])
        loss = 0.8*loss1 + 0.2*loss2
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        hgs, hw_x, hw_y, _ = batch
        pre, out2, out3 = self(hgs.x_dict, hgs.edge_index_dict, hw_x)
        pre, label = self._inverse_transform([pre, hw_y[:,:,:,0]], self.scalar)

        out2 = out2.reshape(hw_x.shape[0], -1)
        out3 = out3.reshape(hw_x.shape[0], -1)

        loss1 = rmse(pre, label)
        loss2 = self.SimCLRLoss(out2, out3, hw_x.shape[0])
        loss = 0.8*loss1 + 0.2*loss2
        self.validation_step_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", epoch_average)
        self.validation_step_outputs.clear()  
    
    def test_step(self, batch, batch_idx):
        hgs, hw_x, hw_y, _ = batch
        pre, _, _ = self(hgs.x_dict, hgs.edge_index_dict, hw_x)
        pre, label = self._inverse_transform([pre, hw_y[:,:,:,0]], self.scalar)
        self.test_step_res_all.append({'y_pre': pre, 'label': label})

        loss = rmse(pre, label)
        
        self.test_step_outputs.append(loss)
        self.test_step_res.append({'y_pre': pre, 'label': label})
        return loss

    def on_test_epoch_end(self):
        epoch_average = torch.stack(self.test_step_outputs).mean()
        y_pre = torch.cat([o['y_pre'] for o in self.test_step_res])
        tgt = torch.cat([o['label'] for o in self.test_step_res])

        loss = 0
        loss_log = ''
        mae_met = 0
        mape_met = 0
        base = 1
        for i in range(self.horizen):
            cur_loss = rmse(y_pre[:,i,:], tgt[:,i,:])
            loss = loss + cur_loss
            mae_met = mae_met + mae(y_pre[:,i,:], tgt[:,i,:])
            mape_met =  mape_met + mape(y_pre[:,i,:], tgt[:,i,:])
            loss_log += 'horizen: {}, meanloss: {}, mae: {}, mape: {}\n'.format(i, loss/(i+1)*base, mae_met/(i+1)*base, mape_met/(i+1)*base)

        self.test_step_outputs.clear() 
        self.test_step_res.clear()
        self.test_step_res_all.clear()

        self.log_dict({"test_epoch_average": epoch_average,
                        "horizen_mean_loss": loss / self.horizen})
        self.log_dict({"test_mae\n": mae_met / self.horizen, 
                        "test_mapes\n": mape_met / self.horizen})
        print("test_horizen_loss\n", loss_log)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(
                                    optimizer,
                                    step_size=self.lr_decay_step,
                                    gamma=self.lr_decay_gamma),
                    'interval': 'epoch'}
        return [optimizer], [lr_scheduler]
    

    def _inverse_transform(self, tensors, scaler):
        def inv(tensor):
            return scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)
        
    def add_gaussian_noise(self, tensor, mean=0, std=1):
        noise = torch.randn(tensor.size()) * std + mean
        noise = noise.to(tensor.device)
        noisy_tensor = tensor + noise
        return noisy_tensor
    
    def SimCLRLoss(self,out_1,out_2,batch_size,temperature=500):
    
        out = torch.cat([out_1, out_2], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool() 
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1) 

        a = torch.sum(out_1 * out_2, dim=-1) / temperature
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)  
        return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        
class TemporalEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(TemporalEmbedding, self).__init__()
        self.input_embdding = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.input_embdding(x.permute(0,3,2,1))
        
class SequenceAugmentor:
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std 

    def flip(self, sequence, m, max_value):
        seq_length = sequence.size()[0]
        indices = torch.randperm(seq_length)[:m].to(sequence.device)  
        sequence[indices] = max_value - sequence[indices]
        return sequence

    def mask(self, sequence, m):
        seq_length = sequence.size()[0]
        indices = torch.randperm(seq_length)[:m].to(sequence.device) 
        sequence[indices] = 0
        return sequence

    def replace_with_noise(self, sequence, m):
        seq_length = sequence.size()[0]
        indices = torch.randperm(seq_length)[:m].to(sequence.device) 
        noise = torch.normal(0.0, self.noise_std, size=(m,)).to(sequence.device)
        sequence[indices] = noise
        return sequence
    
    def shift(self, sequence, m):
        seq_length = sequence.size()[0]
        shifted_sequence = torch.zeros_like(sequence) 
        shifted_sequence[:seq_length - m] = sequence[m:]
        return shifted_sequence
    
    def add_noise(self, sequence, strength=1.0):
        noise = torch.normal(0.0, strength * self.noise_std, size=(sequence.size()[0],)).to(sequence.device)
        return sequence + noise

    def augment_sequence(self, sequence, max_value):
        seq_length = sequence.size()[0]
        m = int(0.25 * seq_length)
        ops = [
            lambda seq: self.flip(seq, m, max_value),
            lambda seq: self.mask(seq, m),
            lambda seq: self.shift(seq, m),
            lambda seq: self.add_noise(seq)
        ]
        chosen_ops = torch.randperm(len(ops))[:2] 
        for idx in chosen_ops:
            sequence = ops[idx.item()](sequence)
        return sequence

    def augment(self, sequences):
        augmented_sequences = []
        max_value = torch.max(sequences)
        for sequence in sequences:
            augmented_sequences.append(self.augment_sequence(sequence,  max_value))
        return torch.stack(augmented_sequences)