from general_code.config.makeConfig import GNNConfig
from general_code.utils.env import make_path
import torch

def make_config(config_path="config.json"):
    config = GNNConfig(config_path, "json")
    config.device = "cpu"
    if config.gpu:
        if torch.cuda.is_available():
            config.device = "cuda:0"
    config.make_atom_featurizer(config.atom_feat_name)
    config.make_bond_featurizer(config.bond_feat_name, config.bond_feat_type)
    config.make_loss_criterion(config.loss_name)
    config.checkpoint_path = make_path("experiment", "log", __file__, "checkpoint.pth")
    config.cache_path = make_path("experiment", "log", __file__, "graph.bin")
    config.net_config["gnn_in_nfeats"] = config.atom_featurizer.feat_size()
    config.net_config["num_rels"]= config.bond_featurizer.feat_size()
    return config