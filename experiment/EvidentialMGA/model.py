from general_code.model.BaseGNN import MPNN
from general_code.utils.makeConfig import RGCNConfig

def make_model(config: RGCNConfig):
    n_tasks = len(config.task_names)
    model = MPNN(
        gnn_hidden_nfeats = config.gnn_hidden_nfeats,
        readout_type = config.readout_type,
        rgcn_loop = config.rgcn_loop,
        rgcn_dropout = config.rgcn_dropout,
        gnn_in_nfeats = config.gnn_in_nfeats,
        num_rels = config.num_rels,
        gnn_out_nfeats = config.gnn_hidden_nfeats[-1],
        fcn_in_nfeats = config.fcn_in_nfeats,
        fcn_hidden_nfeats = config.fcn_hidden_nfeats,
        fcn_out_nfeats = config.fcn_out_nfeats,
        n_tasks = n_tasks
    )
    model.to(config.device)
    return model