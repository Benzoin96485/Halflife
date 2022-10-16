#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define functions for create graphs and datasets
"""

# Imports
# std libs
import os

# 3rd party libs
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import RandomSplitter
import dgl
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import random

# my modules
from general_code.data.GNNFeaturizer import smiles_to_bigraph3, smiles_to_aqua_bigraph


GRAPH_FUNCS = {
    "bigraph": smiles_to_bigraph3,
    "aqua_bigraph": smiles_to_aqua_bigraph
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_ratio_frac(split_ratio: list):
    ratio_sum = sum(split_ratio)
    return tuple(map(lambda x: x / ratio_sum, split_ratio))


def df2gdata(
    df, graph_type, 
    node_featurizer, edge_featurizer,
    smi_col, cache_path, task_names):
    return MoleculeCSVDataset(
        df, GRAPH_FUNCS[graph_type], 
        node_featurizer=node_featurizer, 
        edge_featurizer=edge_featurizer, 
        smiles_column=smi_col, 
        cache_file_path=cache_path,
        task_names=task_names
    )


def df2gdata1(
        df, graph_type, 
        node_featurizer, edge_featurizer,
        smi_col, cache_path, task_names, graph_name="graphs"
    ):
    dataset_ = MoleculeCSVDataset(
        df, GRAPH_FUNCS[graph_type], 
        node_featurizer=node_featurizer, 
        edge_featurizer=edge_featurizer, 
        smiles_column=smi_col, 
        cache_file_path=cache_path + f".{graph_name}.bin",
        task_names=task_names
    )
    return [
        {
            "smiles": smiles,
            graph_name: graphs,
            "labels": labels,
            "mask": mask
        } for smiles, graphs, labels, mask in dataset_
    ]


def gdata_fusion(*gdatas):
    first_gdata = gdatas[0]
    if len(gdatas) > 1:
        for gdata in gdatas[1:]:
            print(len(gdata), len(first_gdata))
            assert len(gdata) == len(first_gdata)
            for i in range(len(gdata)):
                first_gdata[i].update(gdata[i])
    return first_gdata


def df2gdata3(
        df, graph_type, 
        node_featurizer, edge_featurizer,
        smi_col, cache_path, task_names
    ):
    dataset = df2gdata1(df=df, graph_type=graph_type, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, smi_col=smi_col, cache_path=cache_path, task_names=task_names, graph_name="graphs")
    scaffold_dataset = df2gdata1(df=df, graph_type=graph_type, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, smi_col=smi_col + "_scaffold", cache_path=cache_path, task_names=task_names, graph_name="graphs_scaffold")
    sidechain_dataset = df2gdata1(df=df, graph_type=graph_type, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, smi_col=smi_col + "_sidechain", cache_path=cache_path, task_names=task_names, graph_name="graphs_sidechain")
    return gdata_fusion(dataset, scaffold_dataset, sidechain_dataset)


def reg2cls(dfs, task_names, thresholds):
    if isinstance(dfs, pd.DataFrame):
        df_list = [dfs]
    else:
        df_list = dfs
    for df in df_list:
        for i, task_name in enumerate(task_names):
            threshold = thresholds[i]
            if isinstance(threshold, list):
                labels = df[task_name].to_list()
                for j, label in enumerate(labels):
                    n_cuts = len(threshold)
                    if label <= threshold[0]:
                        labels[j] = 0
                    elif label > threshold[-1]:
                        labels[j] = n_cuts
                    else:
                        for k in range(1, n_cuts):
                            if label > threshold[k-1] and label <= threshold[k]:
                                labels[j] = k
                df[task_name] = labels
            else:
                df[task_name] = df[task_name] <= thresholds[i]
    df[task_name].astype("int")


def split_data(dataset, method, ratio, seed):
    frac_train, frac_val, frac_test = make_ratio_frac(ratio)
    if method == "random":
        splitter = RandomSplitter()
        return splitter.train_val_test_split(
            dataset, 
            frac_train=frac_train, frac_val=frac_val, frac_test=frac_test,
            random_state=seed
        )
    elif method == "smiles":
        datalist = list(dataset)
        d = dict()
        for data in datalist:
            if data[0] not in d:
                d[data[0]] = [data]
            else:
                d[data[0]].append(data)
        l = len(dataset)
        count = sorted(list(d.values()), key=lambda x: len(x), reverse=True)
        l_cur = 0
        train_set, val_set, test_set = [], [], []
        for i in range(len(count)):
            if l_cur < l * frac_train:
                train_set.extend(count[i])
            elif l_cur <= l * (frac_train + frac_val):
                val_set.extend(count[i])
            else:
                test_set.extend(count[i])
            l_cur += len(count[i])
        return train_set, val_set, test_set
        

def collate_molgraphs(data):
    if len(data[0]) > 4:
        smiles, graphs, labels, mask, extra = map(list, zip(*data))
        extra = torch.Tensor(extra)
    else:
        smiles, graphs, labels, mask = map(list, zip(*data))
        extra = None
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    mask = torch.stack(mask, dim=0)
    return smiles, bg, labels, mask, extra


def collate_molgraphs3(data):
    smiles = [datapoint["smiles"] for datapoint in data]
    graphs = [datapoint["graphs"] for datapoint in data]
    graphs_scaffold = [datapoint["graphs_scaffold"] for datapoint in data]
    graphs_sidechain = [datapoint["graphs_sidechain"] for datapoint in data]
    labels = [datapoint["labels"] for datapoint in data]
    mask = [datapoint["mask"] for datapoint in data]
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    bg_scaffold = dgl.batch(graphs_scaffold)
    bg_scaffold.set_n_initializer(dgl.init.zero_initializer)
    bg_scaffold.set_e_initializer(dgl.init.zero_initializer)
    bg_sidechain = dgl.batch(graphs_sidechain)
    bg_sidechain.set_n_initializer(dgl.init.zero_initializer)
    bg_sidechain.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    mask = torch.stack(mask, dim=0)
    return {
        "smiles": smiles,
        "bg": bg,
        "bg_scaffold": bg_scaffold,
        "bg_sidechain": bg_sidechain,
        "labels": labels,
        "mask": mask
    }


def csv2df(config, splitted=False, inner_path=None):
    if splitted:
        train_df = pd.read_csv(os.path.join(config.split_data_path, inner_path, config.train_data))
        val_df = pd.read_csv(os.path.join(config.split_data_path, inner_path, config.val_data))
        test_df = pd.read_csv(os.path.join(config.split_data_path, inner_path, config.test_data))
        return train_df, val_df, test_df
    else:
        return pd.read_csv(config.data_path)


def make_dataset(data_df, config, splitted=False, seed_this_time=0):
    kwargs = {
        "graph_type": config.graph_type,
        "node_featurizer": config.atom_featurizer,
        "edge_featurizer": config.bond_featurizer,
        "smi_col": config.smi_col,
        "cache_path": config.cache_path,
        "task_names": config.task_names
    }
    if hasattr(config, "classify_threshold") and config.classify_threshold:
        reg2cls(data_df, config.task_names, config.classify_threshold)
    if splitted:
        train_df, val_df, test_df = data_df
        return df2gdata(df=train_df, **kwargs), df2gdata(df=val_df, **kwargs),df2gdata(df=test_df, **kwargs)
    else:
        dataset = df2gdata(df=data_df, **kwargs)
        train_set, val_set, test_set = split_data(
            dataset=dataset,
            method=config.split_method,
            ratio=config.split_ratio,
            seed=seed_this_time
        )
        return train_set, val_set, test_set


def make_dataset3(data_df, config, splitted=False, seed_this_time=0):
    kwargs = {
        "graph_type": config.graph_type,
        "node_featurizer": config.atom_featurizer,
        "edge_featurizer": config.bond_featurizer,
        "cache_path": config.cache_path,
        "smi_col": config.smi_col,
        "task_names": config.task_names
    }
    if hasattr(config, "classify_threshold") and config.classify_threshold:
        reg2cls(data_df, config.task_names, config.classify_threshold)
    if splitted:
        train_df, val_df, test_df = data_df
        return df2gdata3(df=train_df, **kwargs), df2gdata3(df=val_df, **kwargs),df2gdata3(df=test_df, **kwargs)
    else:
        dataset = df2gdata3(df=data_df, **kwargs)
        train_set, val_set, test_set = split_data(
            dataset=dataset,
            method=config.split_method,
            ratio=config.split_ratio,
            seed=seed_this_time
        )
        return train_set, val_set, test_set


def make_loaders(dataset, batch_size, collate_fn, seed=0):
    train_set, val_set, test_set = dataset
    train_loader = DataLoader(
        dataset=train_set, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_set, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker
    )
    return train_loader, val_loader, test_loader
