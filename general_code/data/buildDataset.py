#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define functions for create graphs and datasets
"""

# Imports
# std libs
# ...

# 3rd party libs
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import *
import dgl
import torch
import pandas as pd

# my modules



def make_ratio_frac(split_ratio: list):
    ratio_sum = sum(split_ratio)
    return tuple(map(lambda x: x / ratio_sum, split_ratio))


def df2gdata(
    df, graph_type, 
    node_featurizer, edge_featurizer,
    smi_col, cache_path, task_names):
    graph_funcs = {
        "bigraph": smiles_to_bigraph
    }
    return MoleculeCSVDataset(
        df, graph_funcs[graph_type], 
        node_featurizer=node_featurizer, 
        edge_featurizer=edge_featurizer, 
        smiles_column=smi_col, 
        cache_file_path=cache_path,
        task_names=task_names
    )


def split_data(dataset, method, ratio, seed):
    if method == "random":
        splitter = RandomSplitter()
    frac_train, frac_val, frac_test = make_ratio_frac(ratio)
    return splitter.train_val_test_split(
        dataset, 
        frac_train=frac_train, frac_val=frac_val, frac_test=frac_test,
        random_state=seed
    )

def collate_molgraphs(data):
    smiles, graphs, labels, mask = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    mask = torch.stack(mask, dim=0)
    return smiles, bg, labels, mask