from general_code.model.BaseGNN import MPNN


def make_model(config):
    model = MPNN(
        **config.net,
    )