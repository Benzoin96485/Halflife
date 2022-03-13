#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
# std libs
# ...

# 3rd party libs
import torch
from torch import nn
from dgl.readout import sum_nodes
from dgl.nn.pytorch.conv import GatedGraphConv, RelGraphConv, NNConv
import torch.nn.functional as F

# my modules


def fc_layer(dropout, in_feats, out_feats):
    """
    :param dropout: float in [0,1)
    :param in_feats: int: number of input features
    :param out_feats: int: number of output features
    :returns: a fully connected layer
    """
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feats, out_feats),
        nn.ReLU(),
        nn.BatchNorm1d(out_feats)
    )


def output_layer(in_feats, out_feats):
    """
    :param in_feats: int: number of input features
    :param out_feats: int: number of output features
    :returns: an linear output layer
    """
    return nn.Sequential(
        nn.Linear(in_feats, out_feats)
    )


class FCN(nn.Module):
    def __init__(self, n_tasks, fcn_in_feats, fcn_hidden_feats, fcn_out_feats=1, fcn_dropout=0.2, **kwargs):
        """
        A fully connected network at GNN downstream.
        :param n_tasks: int: number of tasks
        :param fcn_in_feats: int: number of input features
        :param fcn_hidden_feats: list of int: numbers of features of hidden layers in sequence
        """
        super(FCN, self).__init__()
        self.n_tasks = n_tasks
        self.n_fc_feats = [fcn_in_feats] + fcn_hidden_feats
        self.fc_layers_list = nn.ModuleList()
        for i in range(len(self.n_fc_feats) - 1):
            self.fc_layers_list.append(
                nn.ModuleList(
                    [fc_layer(
                        fcn_dropout, self.n_fc_feats[i], self.n_fc_feats[i + 1]
                    ) for _ in range(self.n_tasks)]
                )
            )
        self.output_layer = nn.ModuleList(
            [output_layer(fcn_hidden_feats[-1], fcn_out_feats) for _ in range(self.n_tasks)])

    def forward(self, mol_feats_list):
        """
        forward function
        :param mol_feats_list: int: number of upstream mol features (length of vector)
        :return: the predicted values
        """
        for i in range(self.n_tasks):
            # mol_feats = torch.cat([feats_list[-1], feats_list[i]], dim=1)
            # 将 feats_list 中第 i 个任务的 feats 传入全连接层进行预测
            h = mol_feats_list[i]
            for layer in self.fc_layers_list:
                h = layer[i](h)
            predict = self.output_layer[i](h)
            if i == 0:
                prediction_all = predict
            else:
                prediction_all = torch.cat([prediction_all, predict], dim=1)
        return prediction_all


class WeightAndSum(nn.Module):
    def __init__(self, gnn_out_feats, n_tasks=1, specific_weighting=True, **kwargs):
        """
        A summation readout layer with task-specific self-attention weight
        :param gnn_out_feats: the atom feature size of last gnn layer
        :param n_tasks: number of tasks
        :param specific_weighting: bool: True if use task-specific attention weight, False if use shared weight
        """
        super(WeightAndSum, self).__init__()
        self.specific_weighting = specific_weighting
        self.in_feats = gnn_out_feats
        self.n_tasks = n_tasks
        if specific_weighting:
            self.weight = nn.ModuleList([self.atom_weight(self.in_feats) for _ in range(self.n_tasks)])
        else:
            self.weight = nn.ModuleList(self.atom_weight(self.in_feats))

    def forward(self, bg, feats):
        """
        :param bg: Batched DGL graph, processing multiple molecules in parallel
        :param feats: input features
        :return: readout features
        """
        if self.specific_weighting:
            feat_list, atom_list = [], []
            for i in range(self.n_tasks):
                with bg.local_scope():  # 在 local_scope 上下文中对图做修改，离开时图的特征不会被改变
                    bg.ndata['h'] = feats  # 存入特征
                    bg.ndata['w'] = self.weight[i](feats)  # 计算 atom_weighting_specific 权重并存入权重
                    specific_feats_sum = sum_nodes(bg, 'h', 'w')  # 以 w 为权对所有节点的 h 求和
                    atom_list.append(bg.ndata['w'])
                feat_list.append(specific_feats_sum)
            return feat_list, atom_list
        else:
            with bg.local_scope():
                bg.ndata['h'] = feats
                weight = self.shared_weighting(feats)
                bg.ndata['w'] = weight
                shared_feats_sum = sum_nodes(bg, 'h', 'w')
            return shared_feats_sum, weight

    def atom_weight(self, in_feats):
        """
        :param in_feats: the input atom feature size
        :return: attention weight of atom
        """
        return nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )


class GNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_rels, edge_nn_feats=[], model_type="RGCN",
                 gnn_residual=True, gnn_bn=True, edge_nn_dropout=0.2, **kwargs):
        super(GNNLayer, self).__init__()
        self.model_type = model_type
        self.activation = F.relu
        if model_type == "RGCN":
            self.graph_conv_layer = RelGraphConv(
                in_feats, out_feats, num_rels=num_rels, regularizer='basis',
                num_bases=None, bias=True, activation=F.relu,
                self_loop=kwargs["rgcn_loop"], dropout=kwargs["rgcn_dropout"]
            )
        elif model_type == "GGNN":
            self.graph_conv_layer = GatedGraphConv(
                in_feats, out_feats, n_etypes=num_rels,
                n_steps=kwargs["n_conv_step"]
            )
        elif model_type == "MPNN0":
            edge_nn_feats = [num_rels] + edge_nn_feats + [in_feats * out_feats]
            self.graph_conv_layer = NNConv(
                in_feats, out_feats,
                edge_func=nn.Sequential(
                    *[fc_layer(
                        dropout=edge_nn_dropout,
                        in_feats=edge_nn_feats[i],
                        out_feats=edge_nn_feats[i+1]
                    ) for i in range(len(edge_nn_feats) - 1)]
                ),
                residual=kwargs["mpnn0_residual"]
            )
        self.residual = gnn_residual
        if gnn_residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = gnn_bn
        if gnn_bn:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, node_feats, etype, **kwargs):
        # if self.model_type == "MPNN0":
            # if etype.dtype != torch.float32:
                # etype = etype.type(torch.float32)
        new_feats = self.graph_conv_layer(bg, node_feats, etype)
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        del res_feats
        torch.cuda.empty_cache()
        return new_feats


class MPNN(nn.Module):
    def __init__(self, **kwargs):
        """
        A basic framework of MPNN-like GNN
        :param kwargs: other args specific to structure
        """
        super(MPNN, self).__init__()
        self.gnn_layers = nn.ModuleList([])
        self.readout = None
        self.build_gnn_layers(**kwargs)
        self.build_readout(**kwargs)
        self.fcn = FCN(**kwargs)

    def forward(self, bg, node_feats, e_feats, **kwargs):
        """
        :param bg: Batched DGL graph, processing multiple molecules in parallel
        :param node_feats: FloatTensor of shape (N, M1)
            N: the total number of atoms in the batched graph
            M1: the input atom feature size
        :param e_feats: original features of edges
        :param norm:
        :return:
            prediction_all: the predicted label
            mol_embedding: the vector features from MPNN
            weight: the self attention weight using for embedding
        """
        for gnn in self.gnn_layers:
            if gnn.model_type == "RGCN":
                node_feats = gnn(bg, node_feats, e_feats, norm=None)
            elif gnn.model_type in ["GGNN", "MPNN0"]:
                node_feats = gnn(bg, node_feats, e_feats)

        mol_embedding, weight = self.readout(bg, node_feats)
        prediction_all = self.fcn(mol_embedding)
        return prediction_all, mol_embedding, weight

    def build_gnn_layers(self, gnn_in_nfeats, gnn_hidden_nfeats, **kwargs):
        in_nfeats = gnn_in_nfeats
        for out_nfeats in gnn_hidden_nfeats:
            self.gnn_layers.append(GNNLayer(in_nfeats, out_nfeats, **kwargs))
            in_nfeats = out_nfeats

    def build_readout(self, readout_type="weight_and_sum", **kwargs):
        if readout_type == "weight_and_sum":
            self.readout = WeightAndSum(**kwargs)
