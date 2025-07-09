import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torch_geometric.nn import SAGEConv, GCNConv, to_hetero, BatchNorm, MessagePassing, HANConv, HGTConv, GATConv
from torch.nn import Parameter as Param

class GCN(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x

class MyModel(nn.Module):
    def __init__(self, hidden_size, omics_specific_flag, network_flag, cell_flag, task, network_specific):
        super(MyModel, self).__init__()
        self.network_flag = network_flag
        self.omics_specific_flag = omics_specific_flag
        self.cell_flag = cell_flag
        self.task = task
        self.network_specific = network_specific

        if self.cell_flag:
            self.CCLE_exp_encoder = nn.Sequential(
                nn.LazyLinear(hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, int(hidden_size/2)),
                nn.ReLU(True)
            )

            self.CCLE_ess_encoder = nn.Sequential(
                nn.LazyLinear(hidden_size),
                nn.ReLU(True),
                #nn.Dropout(p=0.5),
                nn.Linear(hidden_size, int(hidden_size/2)),
                nn.ReLU(True)
            )

        if self.omics_specific_flag:
            self.specific_omics_encoder = nn.Sequential(
                nn.LazyLinear(hidden_size),
                nn.ReLU(True),
            )

        if self.network_flag:
            self.network_encoder = GCN(2*hidden_size, 2*hidden_size, hidden_size) ####input_size:2/ 32-4*hidden_size, 128-hidden_size, 64-2*hidd

        self.predictor = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, inputs, edge_index, node_feats, cell_index=None):
        cell_input_exp, cell_input_ess, specific_omics_1, specific_omics_2, gene1_idx, gene2_idx = inputs

        if self.cell_flag:
            enc_cell_exp = self.CCLE_exp_encoder(cell_input_exp)
            enc_cell_ess = self.CCLE_ess_encoder(cell_input_ess)
            enc_cell = torch.cat((enc_cell_exp, enc_cell_ess), 1)
        
        if self.omics_specific_flag:
            enc_specific_omics_1 = self.specific_omics_encoder(specific_omics_1)
            enc_specific_omics_2 = self.specific_omics_encoder(specific_omics_2)
            enc_specific_omics = torch.cat((enc_specific_omics_1, enc_specific_omics_2), 1)

        if self.network_flag:
            network_x = self.network_encoder(node_feats, edge_index)
            network_gene1_x = network_x[ gene1_idx ]
            network_gene2_x = network_x[ gene2_idx ]
            enc_network_feats = torch.cat((network_gene1_x, network_gene2_x), 1)
        
        if self.omics_specific_flag:
            enc_combined = enc_specific_omics

            if self.network_flag:
                enc_combined = torch.cat((enc_combined, enc_network_feats), 1)

                if self.cell_flag:
                    enc_combined = torch.cat((enc_combined, enc_cell), 1)
            else:
                if self.cell_flag:
                    enc_combined = torch.cat((enc_combined, enc_cell), 1)

        output = self.predictor(enc_combined)

        return output.view(-1)
