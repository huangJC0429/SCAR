import torch.nn as nn
import torch.nn.functional as F
import torch
from gcn.layers import GraphConvolution, MLPLayer, Trust_GraphConvolution
from torch_geometric.utils import one_hot, spmm


class GCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, tau=0.0):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.normalize = args.normalize

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        # if self.normalize == 1:
        #     x = F.normalize(x, p=2, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # if self.normalize == 1:
        #     x = 0.1*x - 0.9*F.normalize(x, p=2, dim=1)
        return x


class Trust_GCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, tau=0.0):
        super(Trust_GCN, self).__init__()

        self.gc1 = Trust_GraphConvolution(nfeat, nhid)
        self.gc2 = Trust_GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.normalize = args.normalize
        self.instance_w = args.instance_w

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        # if self.normalize == 1:
        #     x = F.normalize(x, p=2, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # if self.normalize == 1:
        #     x = 0.1*x - 0.9*F.normalize(x, p=2, dim=1)
        return x

    def predict(self,x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        # if self.normalize == 1:
        #     x = F.normalize(x, p=2, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2.predict(x, adj, self.instance_w)

        return x



class DeepGCN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, tau=0.0):
        super(DeepGCN, self).__init__()


        self.gcn_convs = nn.ModuleList([GraphConvolution(nfeat, nhid)])
        if args.layer > 2:
            for _ in range(args.layer-2):
                self.gcn_convs.append(GraphConvolution(nhid, nhid))
        # output layer
        self.output_layer = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.layer = args.layer
        self.representations = None

    def forward(self, x, adj):
        representations = []
        for i, gcn_layer in enumerate(self.gcn_convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = gcn_layer(x, adj)
            x = F.relu(x)
            representations.append(x.cpu().detach())
        # output layer
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.output_layer(x, adj)

        representations.append(x.cpu().detach())


        self.representations = representations

        return x

class DeepMLP(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, tau=0.0):
        super(DeepMLP, self).__init__()


        self.gcn_convs = nn.ModuleList([nn.Linear(nfeat, nhid)])
        for _ in range(args.layer-2):
            self.gcn_convs.append(nn.Linear(nhid, nhid))
        # output layer
        self.output_layer = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.layer = args.layer
        self.representations = None

    def forward(self, x, adj):
        representations = []
        for i, mlp_layer in enumerate(self.gcn_convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = mlp_layer(x)
            x = F.relu(x)
            representations.append(x.cpu().detach())
        # output layer
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.output_layer(x)

        representations.append(x.cpu().detach())


        self.representations = representations

        return x


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 + pt) ** self.gamma * ce_loss  # 根据inverse Focal Loss公式计算
        return focal_loss

class AULS(nn.Module):
    def __init__(self, epsilon=-0.5, gamma=0.5, labels=None):
        super(AULS, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        # since training label is not completed
        self.labels = labels
        self.K = max(labels) + 1

    def forward(self, inputs, targets, epoch, mask):
        inputs = F.softmax(inputs,dim=1)
        y = F.one_hot(self.labels)[mask]
        # print(y.shape)
        # exit()
        ep = self.epsilon + self.gamma*(1/(1+epoch))
        loss = (y*(1-ep) + ep/self.K)*torch.log(inputs + 0.000000001)
        loss = torch.sum(loss,dim=1)
        loss = - torch.mean(loss, dim=0)
        return loss

    # def label_smooth(self, inputs, targets):
    #     inputs = F.softmax(inputs, dim=1)
    #     y = F.one_hot(self.labels)[mask]
    #     ep = self.epsilon + self.gamma * (1 / (1 + epoch))
    #     loss = (y * (1 - ep) + ep / self.K) * torch.log(inputs + 0.000000001)


class MLP_DNN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, dropout, tau=0.1):
        super(MLP_DNN, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid)
        self.layer2 = nn.Linear(nhid, nclass)
        self.dropout = dropout
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.layer1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x)
        return x

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid)
        self.layer2 = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.layer1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x)
        return  torch.nn.functional.log_softmax(x, dim=1)
class LP(torch.nn.Module):
    def __init__(self, y, train_mask, K, alpha=0.9, temp=1):
        super(LP, self).__init__()
        self.y = y
        self.train_mask = train_mask
        self.alpha = alpha
        self.temp = temp
        self.num_layers = K
    def forward(self,  homo_adj):
        # return self.LLP(self.y, homo_adj, mask=self.train_mask)

        # out = self.LP(self.y, homo_adj, mask=self.train_mask)
        if self.y.dtype == torch.long and self.y.size(0) == self.y.numel():
            self.y = one_hot(self.y.view(-1))
        # out = self.y

        out = torch.zeros_like(self.y).float()
        out[self.train_mask] = self.y[self.train_mask]
        # print(out.shape)
        # print(homo_adj)

        for _ in range(self.num_layers):
            # propagate_type: (y: Tensor, edge_weight: OptTensor)
            out = self.alpha*(homo_adj@out) + (1-self.alpha)*out
            # fix label y on each epoch
            out[self.train_mask] = self.y[self.train_mask]
            out = torch.clip(out, 0, 1)

        return out