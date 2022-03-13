from general_code.model.BaseGNN import MPNN
from config import ConfigRGCN


def make_model(config: ConfigRGCN):
    config.add_gnn_in_feats()
    model = MPNN(
        **config.net,
    )