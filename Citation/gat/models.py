import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv, GATv2Conv


class GAT(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.dropout = dropout


        self.conv1 =  GATConv(nfeat, nhid, heads=8, dropout=dropout)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(nhid*8, nclass, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index), alpha=0.2)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def predict(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index), alpha=0.2)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class T_GAT(torch.nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.dropout = dropout
        self.instance_w = args.instance_w


        self.conv1 =  GATConv(nfeat, nhid, heads=8, dropout=dropout)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(nhid*8, nclass, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index), alpha=0.2)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def predict(self, x, edge_index):
        x_1 = F.dropout(x, p=self.dropout, training=self.training)
        x_1 = F.elu(self.conv1(x_1, edge_index), alpha=0.2)

        x_2 = F.dropout(x_1, p=self.dropout, training=self.training)
        x_2 = self.conv2(x_2, edge_index)

        predicted_class = torch.argmax(x_2, dim=1)
        centric = self.conv2.lin_src.weight[predicted_class, :]
        # print(self.conv2.lin_src.weight.shape)
        # print(centric.shape)
        # print(self.instance_w.shape)
        # print(x_1.shape)
        aug_input = torch.multiply(1 - self.instance_w, x_1) + torch.multiply(self.instance_w, centric)

        x = F.dropout(aug_input, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return x