from general_code.data.tdcDataset import make_multitask_data
from general_code.data.buildDataset import collate_molgraphs, df2gdata, split_data, collate_molgraphs
from torch.utils.data import DataLoader
import pandas as pd

def make_dataset(config):
    if config.dump_dataset == True:
        df = make_multitask_data(
            task_name_list=config.task_names, 
            check_name=config.check_names, 
            atom_allow_type=config.atom_allow_type
        )
        df.to_csv(config.dump_dataset)
    else:
        df = pd.read_csv(config.dump_dataset)
    return df2gdata(
        df=df,
        graph_type=config.graph_type,
        node_featurizer=config.atom_featurizer,
        edge_featurizer=config.bond_featurizer,
        smi_col=config.smi_col, 
        cache_path=config.cache_path,
        task_names=config.task_names
    )

def make_loaders(dataset, config, seed):
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