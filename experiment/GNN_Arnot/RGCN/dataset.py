from sklearn.utils import shuffle
from general_code.data.buildDataset import csv2gdata, split_data
from torch.utils.data import DataLoader

def make_dataset(config):
    return csv2gdata(
        data_path=config.data_path, 
        graph_type=config.graph_type, 
        node_featurizer=config.atom_featurizer, 
        edge_featurizer=config.bond_featurizer, 
        smi_col=config.smi_col, 
        cache_path=config.cache_path
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
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_set, 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_set, 
        batch_size=batch_size, 
        shuffle=True
    )
    return train_loader, val_loader, test_loader