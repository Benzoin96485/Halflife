from general_code.data.buildDataset import csv2gdata, split_data


def load_data(config, seed):
    dataset = csv2gdata(
        csv=config.data["data_path"], 
        graph_type=config.graph["graph_type"], 
        node_feat_name=config.graph["atom_feat_name"], 
        edge_feat_name=config.graph["bond_feat_name"], 
        smi_col=config.data["smi_col"], 
        cache_path=config.data["cache_path"]
    )
    train_set, val_set, test_set = split_data(
        dataset=dataset,
        method=config.data["split_method"],
        ratio=
    )