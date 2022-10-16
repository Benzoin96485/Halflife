from sklearn.utils import shuffle
from general_code.data.buildDataset import collate_molgraphs, df2gdata, split_data, collate_molgraphs
from torch.utils.data import DataLoader
import pandas as pd
import os
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

def make_dataset(config, splitted=False, inner_path=None):
    generators = rdNormalizedDescriptors.RDKit2DNormalized()
    config.extra_fcn_nfeats = len(generators.GetColumns()) - 1
    def rdkit_feature(smi):
        return generators.process(smi)[1:]

    if splitted:
        train_df = pd.read_csv(os.path.join(config.split_data_path, inner_path, config.train_data))
        val_df = pd.read_csv(os.path.join(config.split_data_path, inner_path, config.val_data))
        test_df = pd.read_csv(os.path.join(config.split_data_path, inner_path, config.test_data))
        df_partition = (train_df, val_df, test_df)
        dataset_partition = []
        for df in df_partition:
            other_list = df[config.smi_col].apply(rdkit_feature)
            gdata = df2gdata(
                df=df,
                graph_type=config.graph_type,
                node_featurizer=config.atom_featurizer,
                edge_featurizer=config.bond_featurizer,
                smi_col=config.smi_col, 
                cache_path=config.cache_path,
                task_names=config.task_names
            )
            other_list = [other_list[i] for i in range(other_list.shape[0]) if i in gdata.valid_ids]
            dataset = []
            for i, value in enumerate(gdata):
                dataset.append((*value, other_list[i]))
            dataset_partition.append(dataset)
        return tuple(dataset_partition)

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