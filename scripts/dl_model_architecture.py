# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from torch.nn.functional import softmax, relu, selu, leaky_relu, elu, max_pool1d, batch_norm
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import global_max_pool as gmp
import torch.nn.functional as F
import torch.nn.init as init
import time
import torch
import torch.nn as nn
import torch.optim as optim
import inspect
import random
import math


class NN_Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, hid_dims, dropout):
        super().__init__()
        
        #Initializing 
        self.out_dim = out_dim

        self.dropout = nn.Dropout(dropout)
        
        self.relu = nn.ReLU()
        
        new_hid_dims = [input_dim]+hid_dims+[out_dim]
        
        self.hid_dims = new_hid_dims
        
        N_hid_dim = len(new_hid_dims)
        
        self.fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(new_hid_dims[i],new_hid_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(new_hid_dims[i+1])
            )
            for i in range(0,(N_hid_dim-1))
        ])
                    
    def forward(self, cell_src, batch_size):
        
        #src = [batch_size, cell_src_input_dim]
        
        x = cell_src

        for fc in self.fcs:
            x = fc(x)
            
        #Output will be [batch_size, output_dim]                   
        
        return x


class GATNet(torch.nn.Module):
    def __init__(self, input_dim, n_heads, hid_dim, out_dim, dropout):

        super(GATNet, self).__init__()

        # SMILES graph branch
        self.dropout = nn.Dropout(dropout)
        self.conv1 = GATConv(input_dim, input_dim, heads=n_heads, dropout=dropout)
        self.conv2 = GATConv(input_dim*n_heads, input_dim*n_heads, heads=n_heads, dropout=dropout)
        self.conv3 = GATConv(input_dim*n_heads*n_heads, input_dim*n_heads*n_heads, dropout=dropout)
        self.fc_g1 = torch.nn.Linear(input_dim*n_heads*n_heads, hid_dim)
        self.bn2 = nn.BatchNorm1d(hid_dim)
        self.fc_g2 = torch.nn.Linear(hid_dim, out_dim)
        self.out_dim = out_dim


    def forward(self, smiles_src):

        # get graph input
        x, edge_index, batch = smiles_src.x, smiles_src.edge_index, smiles_src.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = gmp(x, batch)       # global max pooling

        x1 = F.relu(self.fc_g1(x))
        x1 = self.dropout(x1)
        x2 = F.relu(self.fc_g2(x1))
        return(x2)


class GATNet_Enhanced(torch.nn.Module):
    def __init__(self, input_dim, n_heads, hid_dim, out_dim, dropout, n_gat_layers=3):
        super(GATNet_Enhanced, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.n_gat_layers = n_gat_layers

        # First GAT layer
        self.conv1 = GATConv(input_dim, input_dim, heads=n_heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(input_dim * n_heads)

        # Middle GAT layers (variable)
        self.convs_middle = nn.ModuleList()
        self.bns_middle = nn.ModuleList()
        current_dim = input_dim * n_heads

        for i in range(n_gat_layers - 1):
            self.convs_middle.append(
                GATConv(current_dim, current_dim, heads=n_heads, dropout=dropout)
            )
            self.bns_middle.append(nn.BatchNorm1d(current_dim * n_heads))
            current_dim = current_dim * n_heads

        # Last GAT layer (single head for aggregation)
        self.conv_last = GATConv(current_dim, current_dim, heads=1, dropout=dropout)
        self.bn_last = nn.BatchNorm1d(current_dim)

        # FC layers with residual connection
        self.fc_g1 = torch.nn.Linear(current_dim, hid_dim)
        self.bn_fc1 = nn.BatchNorm1d(hid_dim)
        self.fc_g2 = torch.nn.Linear(hid_dim, hid_dim)
        self.bn_fc2 = nn.BatchNorm1d(hid_dim)
        self.fc_g3 = torch.nn.Linear(hid_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, smiles_src):
        x, edge_index, batch = smiles_src.x, smiles_src.edge_index, smiles_src.batch

        # First layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Middle layers
        for conv, bn in zip(self.convs_middle, self.bns_middle):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Last layer
        x = self.conv_last(x, edge_index)
        x = self.bn_last(x)
        x = F.relu(x)
        x = gmp(x, batch)  # global max pooling

        # FC layers with residual
        x1 = F.relu(self.bn_fc1(self.fc_g1(x)))
        x1 = self.dropout(x1)
        x2 = F.relu(self.bn_fc2(self.fc_g2(x1)))
        x2 = self.dropout(x2)
        x2 = x2 + x1  # Residual connection
        x3 = F.relu(self.fc_g3(x2))

        return x3


class GCNNet(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, dropout):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.dropout = nn.Dropout(dropout)
        self.conv1 = GCNConv(input_dim, hid_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.bn2 = nn.BatchNorm1d(hid_dim)
        self.conv3 = GCNConv(hid_dim, hid_dim)
        self.bn3 = nn.BatchNorm1d(hid_dim)
        self.fc_g1 = torch.nn.Linear(hid_dim, hid_dim)
        self.bn4 = nn.BatchNorm1d(hid_dim)
        self.fc_g2 = torch.nn.Linear(hid_dim, out_dim)
        self.out_dim = out_dim


    def forward(self, smiles_src):

        # get graph input
        x, edge_index, batch = smiles_src.x, smiles_src.edge_index, smiles_src.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = gmp(x, batch)       # global max pooling

        x1 = self.fc_g1(x)
        x1 = self.bn4(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        x2 = self.fc_g2(x1)
        return(x2)


class GCNNet_Enhanced(torch.nn.Module):
    def __init__(self, input_dim, gcn_hid_dims, fc_hid_dim, out_dim, dropout, n_gcn_layers=3):
        super(GCNNet_Enhanced, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.n_gcn_layers = n_gcn_layers

        # Build variable depth GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First GCN layer
        self.convs.append(GCNConv(input_dim, gcn_hid_dims[0]))
        self.bns.append(nn.BatchNorm1d(gcn_hid_dims[0]))

        # Middle GCN layers
        for i in range(1, n_gcn_layers):
            in_dim = gcn_hid_dims[i-1]
            out_dim_layer = gcn_hid_dims[i] if i < len(gcn_hid_dims) else gcn_hid_dims[-1]
            self.convs.append(GCNConv(in_dim, out_dim_layer))
            self.bns.append(nn.BatchNorm1d(out_dim_layer))

        # FC layers with residual connection
        final_gcn_dim = gcn_hid_dims[-1] if n_gcn_layers <= len(gcn_hid_dims) else gcn_hid_dims[-1]
        self.fc_g1 = torch.nn.Linear(final_gcn_dim, fc_hid_dim)
        self.bn_fc1 = nn.BatchNorm1d(fc_hid_dim)
        self.fc_g2 = torch.nn.Linear(fc_hid_dim, fc_hid_dim)
        self.bn_fc2 = nn.BatchNorm1d(fc_hid_dim)
        self.fc_g3 = torch.nn.Linear(fc_hid_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, smiles_src):
        # get graph input
        x, edge_index, batch = smiles_src.x, smiles_src.edge_index, smiles_src.batch

        # GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Global max pooling
        x = gmp(x, batch)

        # FC layers with residual
        x1 = F.relu(self.bn_fc1(self.fc_g1(x)))
        x1 = self.dropout(x1)
        x2 = F.relu(self.bn_fc2(self.fc_g2(x1)))
        x2 = self.dropout(x2)
        x2 = x2 + x1  # Residual connection
        x3 = self.fc_g3(x2)

        return x3


class LSTM_Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, out_dim, n_layers,  dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        
        self.out_dim = out_dim
        
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        #self.relu = leaky_relu
        self.relu = selu
        
        self.fc = nn.Linear(n_layers*hid_dim, out_dim)

    def forward(self, src, batch_size):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        hidden = hidden.permute(1,0,2)
        
        hidden = torch.reshape(hidden,[batch_size,self.n_layers*self.hid_dim])
        
        output = self.dropout(self.fc(hidden))
        
        return output


class CNN_Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, out_dim, n_filters, filter_sizes, dropout):
        super().__init__()

        
        self.out_dim = out_dim
        
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx = 0)
        
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = emb_dim, 
                                              out_channels = n_filters, 
                                              kernel_size = fs,
                                              padding_mode='zeros',
                                              padding=0)
                                    for fs in filter_sizes
                                    ])
        
        self.dropout = nn.Dropout(dropout)
       
        self.batch_norm_cnn = nn.BatchNorm1d(n_filters)
        
        #self.relu = leaky_relu
        self.relu = selu
        
        self.maxpool = max_pool1d
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, out_dim)

        self.batch_norm_out = nn.BatchNorm1d(out_dim)

    def forward(self, src, batch_size):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        #embedded = [src len, batch size, emb dim]
        
        embedded = embedded.permute(1, 2, 0)
        #embedded = [batch size, emb dim, src len]
        
        conved = [self.batch_norm_cnn(self.relu(conv(embedded))) for conv in self.convs]
        #conved_n = [batch size, n_filters, src len - filter_sizes[n] + 1]
        
        pooled = [self.maxpool(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        output = self.dropout(self.relu(self.fc(cat)))
        

        return output


class Seq2Func(nn.Module):
    def __init__(self, cell_encoder, smiles_encoder, hid_dim, out_dim, dropout, device):
        super().__init__()
        
        self.cell_encoder = cell_encoder
        
        self.smiles_encoder = smiles_encoder
        
        self.device = device
        
        self.fc1 = nn.Linear(cell_encoder.out_dim+smiles_encoder.out_dim, hid_dim)
        
        self.fc2 = nn.Linear(hid_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.relu = leaky_relu
        
    def forward(self, cell_src, smiles_src):
        
        #Get protein encoder output
        cell_output = self.cell_encoder(cell_src, cell_src.shape[1]) 
        #cell_output = [batch size, cell out_dim]
        
        #Get smiles encoder output
        smiles_output = self.smiles_encoder(smiles_src, smiles_src.shape[1])
        #smiles_output = [batch size, smiles out_dim]
        
        ls_output = torch.cat((cell_output,smiles_output),1)
        #ls_output = [batch size, cell out_dim + smiles out_dim]
        
        o1 = self.dropout(self.relu(self.fc1(ls_output)))
        #o1 = [batch size, hid_dim]
        
        final_output = self.relu(self.fc2(o1))
        #final_output = [batch_size, 1]
        
        return final_output


class Seq2Func_Net(nn.Module):
    def __init__(self, cell_encoder, smiles_encoder, hid_dim, out_dim, dropout, device):
        super().__init__()

        self.cell_encoder = cell_encoder

        self.smiles_encoder = smiles_encoder

        self.device = device

        self.fc1 = nn.Linear(cell_encoder.out_dim+smiles_encoder.out_dim, hid_dim)

        self.fc2 = nn.Linear(hid_dim, out_dim)

        self.dropout = nn.Dropout(dropout)

        self.relu = leaky_relu

    def forward(self, cell_src, smiles_src):

        #Get protein encoder output
        cell_output = self.cell_encoder(cell_src, cell_src.shape[1])
        #cell_output = [batch size, cell out_dim]

        #Get smiles encoder output
        smiles_output = self.smiles_encoder(smiles_src)
        #smiles_output = [batch size, smiles out_dim]

        ls_output = torch.cat((cell_output,smiles_output),1)
        #ls_output = [batch size, cell out_dim + smiles out_dim]

        o1 = self.dropout(self.relu(self.fc1(ls_output)))
        #o1 = [batch size, hid_dim]

        final_output = self.relu(self.fc2(o1))
        #final_output = [batch_size, 1]

        return final_output


class Seq2Func_Net_Enhanced(nn.Module):
    def __init__(self, cell_encoder, smiles_encoder, hid_dims, out_dim, dropout, device):
        super().__init__()

        self.cell_encoder = cell_encoder
        self.smiles_encoder = smiles_encoder
        self.device = device

        combined_dim = cell_encoder.out_dim + smiles_encoder.out_dim

        # Build multi-layer fusion network
        self.fcs = nn.ModuleList()
        prev_dim = combined_dim

        for hid_dim in hid_dims:
            self.fcs.append(nn.Sequential(
                nn.Linear(prev_dim, hid_dim),
                nn.BatchNorm1d(hid_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ))
            prev_dim = hid_dim

        # Output layer
        self.fc_out = nn.Linear(prev_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cell_src, smiles_src):
        #Get cell encoder output
        cell_output = self.cell_encoder(cell_src, cell_src.shape[1])
        #cell_output = [batch size, cell out_dim]

        #Get smiles encoder output
        smiles_output = self.smiles_encoder(smiles_src)
        #smiles_output = [batch size, smiles out_dim]

        # Concatenate
        x = torch.cat((cell_output, smiles_output), 1)
        #x = [batch size, cell out_dim + smiles out_dim]

        # Multi-layer fusion
        for fc in self.fcs:
            x = fc(x)

        # Output
        final_output = self.fc_out(x)
        #final_output = [batch_size, 1]

        return final_output


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.05, 0.05)


# +
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# -

def evaluation(model, iterator, criterion, DEVICE):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            cell_src = batch[1].to(DEVICE)
            smiles_src = batch[0].permute(1,0).to(DEVICE)
            trg = batch[2].to(DEVICE)

            output = model(cell_src, smiles_src).squeeze(1) 
            #output = [batch size]
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            
            del cell_src
            del smiles_src
            torch.cuda.empty_cache()
        
    return epoch_loss / len(iterator)


def evaluation_net(model, iterator, criterion, N_dim, DEVICE):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, data in enumerate(iterator):

            cell_src = data.cell_src
            cell_src = cell_src.reshape(data.c_size.shape[0],N_dim)
            cell_src = cell_src.to(DEVICE)
            smiles_src = data.to(DEVICE)
            trg = data.y.to(DEVICE)

            output = model(cell_src, smiles_src).squeeze(1) 
            #output = [batch size]
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            
            del cell_src
            del smiles_src
            torch.cuda.empty_cache()
        
    return epoch_loss / len(iterator)


def training(model, iterator, optimizer, criterion, clip, DEVICE):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        cell_src = batch[1].to(DEVICE)
        smiles_src = batch[0].permute(1,0).to(DEVICE)
        trg = batch[2].to(DEVICE)
        
        optimizer.zero_grad()
        
        output = model(cell_src, smiles_src).squeeze(1)
        #output = [batch size]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        del cell_src
        del smiles_src
        torch.cuda.empty_cache()
        
    return epoch_loss / len(iterator)


def training_net(model, iterator, optimizer, criterion, N_dim, clip, DEVICE):
    
    model.train()
    
    epoch_loss = 0
    
    for i, data in enumerate(iterator):
        
        cell_src = data.cell_src
        cell_src = cell_src.reshape(data.c_size.shape[0],N_dim)
        cell_src = cell_src.to(DEVICE)
        smiles_src = data.to(DEVICE)
        trg = data.y.to(DEVICE)
        
        optimizer.zero_grad()
        
        output = model(cell_src, smiles_src).squeeze(1)
        #output = [batch size]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        del cell_src
        del smiles_src
        torch.cuda.empty_cache()
        
    return epoch_loss / len(iterator)


def evaluation_net_performance(model, iterator, criterion, N_dim, DEVICE):
    
    model.eval()
    
    epoch_loss = 0
    output_list, label_list = [],[]
    
    with torch.no_grad():
    
        for i, data in enumerate(iterator):

            cell_src = data.cell_src
            cell_src = cell_src.reshape(data.c_size.shape[0],N_dim)
            cell_src = cell_src.to(DEVICE)
            smiles_src = data.to(DEVICE)
            trg = data.y.to(DEVICE)

            output = model(cell_src, smiles_src).squeeze(1) 
            #output = [batch size]
            
            rev_output = output.tolist()
            rev_trg = trg.tolist()
            
            for j in range(len(rev_output)):
                output_list.append(rev_output[j])
                label_list.append(rev_trg[j])
            
            del cell_src
            del smiles_src
            torch.cuda.empty_cache()
        
    return (output_list,label_list)


def evaluation_performance(model, iterator, criterion, DEVICE):
    
    model.eval()
    
    epoch_loss = 0
    output_list, label_list = [],[]
    
    with torch.no_grad():
    
        for i,batch in enumerate(iterator):

            cell_src = batch[1].to(DEVICE)
            smiles_src = batch[0].permute(1,0).to(DEVICE)
            trg = batch[2].to(DEVICE)

            output = model(cell_src, smiles_src).squeeze(1) 
            #output = [batch size]
            
            rev_output = output.tolist()
            rev_trg = trg.tolist()
            
            for j in range(len(rev_output)):
                output_list.append(rev_output[j])
                label_list.append(rev_trg[j])
            
            del cell_src
            del smiles_src
            torch.cuda.empty_cache()
        
    return (output_list,label_list)


