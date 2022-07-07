from general_code.model.BaseGNN import MPNN


def make_model(config):
    gnn_out_nfeats = config.gnn_hidden_nfeats[-1]
    print(f"gnn_out_nfeats: {gnn_out_nfeats}")
    if hasattr(config, "extra_fcn_nfeats"):
        fcn_in_nfeats = gnn_out_nfeats + config.extra_fcn_nfeats
    else:
        fcn_in_nfeats = gnn_out_nfeats
    print(f"fcn_in_nfeats: {fcn_in_nfeats}")
    n_tasks = len(config.task_names)
    model = MPNN(
        gnn_hidden_nfeats = config.gnn_hidden_nfeats,
        readout_type = config.readout_type,
        rgcn_loop = config.rgcn_loop,
        rgcn_dropout = config.rgcn_dropout,
        gnn_in_nfeats = config.gnn_in_nfeats,
        num_rels = config.num_rels,
        gnn_out_nfeats = gnn_out_nfeats,
        fcn_in_nfeats = fcn_in_nfeats,
        fcn_hidden_nfeats = config.fcn_hidden_nfeats,
        n_tasks = n_tasks
    )
    model.to(config.device)
    return model