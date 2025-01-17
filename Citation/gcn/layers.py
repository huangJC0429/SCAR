import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class MLPLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output = adj@support
        # output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Trust_GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Trust_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output = adj@support
        # output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def predict(self, input, adj, instance_w):
        # instance_w是到标签节点的距离
        # input = torch.multiply(instance_w, input)
        input = adj @ input
        output= torch.mm(input, self.weight)

        if self.bias is not None:
            output = output + self.bias
        else:
            output = output
        predicted_class = torch.argmax(output, dim=1)
        centric = self.weight[:, predicted_class].t()
        # print(centric.shape)
        aug_input = torch.multiply(1-instance_w, input) + torch.multiply(instance_w, centric)

        # re-compute output
        aug_output = torch.mm(aug_input, self.weight)
        if self.bias is not None:
            aug_output = aug_output + self.bias
        else:
            aug_output = aug_output

        return aug_output
    def predict1(self, input, adj, instance_w):
        # instance_w是到标签节点的距离
        # input = torch.multiply(instance_w, input)
        support = torch.mm(input, self.weight)
        output = adj@support
        if self.bias is not None:
            output = output + self.bias
        else:
            output = output
        predicted_class = torch.argmax(output, dim=1)
        centric = self.weight[:, predicted_class].t()
        print(centric.shape)
        aug_input = torch.multiply(instance_w, input) + torch.multiply(1 - instance_w, centric)

        # re-compute output
        aug_support = torch.mm(aug_input, self.weight)
        aug_output = adj @ aug_support
        if self.bias is not None:
            aug_output = aug_output + self.bias
        else:
            aug_output = aug_output

        return aug_output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
