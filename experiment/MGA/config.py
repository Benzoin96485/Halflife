from general_code.config.makeConfig import GNNConfig
from general_code.utils.env import make_path
import torch

def make_config(config_path="config.json", file_path=__file__):
    config = GNNConfig(config_path, "json")
    config.device = "cpu"
    if config.gpu:
        if torch.cuda.is_available():
            config.device = "cuda:3"
    config.make_atom_featurizer(config.atom_feat_name)
    config.make_bond_featurizer(config.bond_feat_name, config.bond_feat_type)
    config.make_loss_criterion(config.loss_name)
    config.log_path = make_path("experiment", "log", file_path, "")
    config.checkpoint_path = make_path("experiment", "log", file_path, "checkpoint.pth")
    config.cache_path = make_path("experiment", "log", file_path, "graph.bin")
    config.gnn_in_nfeats = config.atom_featurizer.feat_size()
    config.num_rels = config.bond_featurizer.feat_size()
    print(f"gnn_in_nfeats: {config.gnn_in_nfeats}")
    print(f"num_rels: {config.num_rels}")
    return config