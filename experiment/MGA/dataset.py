from sklearn.utils import shuffle
from general_code.data.buildDataset import collate_molgraphs, df2gdata, split_data, collate_molgraphs
from torch.utils.data import DataLoader
import pandas as pd
import os

def make_dataset(config, splitted=False, inner_path=None):
    if splitted:
        train_df = pd.read_csv(os.path.join(config.split_data_path, inner_path, config.train_data))
        val_df = pd.read_csv(os.path.join(config.split_data_path, inner_path, config.val_data))
        test_df = pd.read_csv(os.path.join(config.split_data_path, inner_path, config.test_data))
        return df2gdata(
            df=train_df,
            graph_type=config.graph_type,
            node_featurizer=config.atom_featurizer,
            edge_featurizer=config.bond_featurizer,
            smi_col=config.smi_col, 
            cache_path=config.cache_path,
            task_names=config.task_names
        ), df2gdata(
            df=val_df,
            graph_type=config.graph_type,
            node_featurizer=config.atom_featurizer,
            edge_featurizer=config.bond_featurizer,
            smi_col=config.smi_col, 
            cache_path=config.cache_path,
            task_names=config.task_names
        ), df2gdata(
            df=test_df,
            graph_type=config.graph_type,
            node_featurizer=config.atom_featurizer,
            edge_featurizer=config.bond_featurizer,
            smi_col=config.smi_col, 
            cache_path=config.cache_path,
            task_names=config.task_names
        )
    else:
        df = pd.read_csv(config.data_path)
        return df2gdata(
            df=df,
            graph_type=config.graph_type,
            node_featurizer=config.atom_featurizer,
            edge_featurizer=config.bond_featurizer,
            smi_col=config.smi_col, 
            cache_path=config.cache_path,
            task_names=config.task_names
        )

def make_loaders(dataset, config, seed, splitted=False):
    if splitted:
        train_set, val_set, test_set = dataset
    else:
        train_set, val_set, test_set = split_data(
            dataset=dataset,
            method=config.split_method,
            ratio=config.split_ratio,
            seed=seed
        )
    batch_size = config.batch_size
    train_loader = DataLoader(
        dataset=train_set, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_molgraphs
    )
    val_loader = DataLoader(
        dataset=val_set, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_molgraphs
    )
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_molgraphs
    )
    return train_loader, val_loader, test_loader