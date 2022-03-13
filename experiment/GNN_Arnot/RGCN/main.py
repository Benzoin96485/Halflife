from general_code.utils.argParse import parseArgs
from general_code.utils.log import Logger
from general_code.config.makeConfig import Config
from dataset import make_dataset, make_loaders
from general_code.config.makeConfig import GNNConfig
from general_code.model.BaseGNN import MPNN


def main():
    config = GNNConfig(parseArgs()["config"], "json")
    logger = Logger(config.label_names)
    for time_id in range(config.eval_times):
        seed_this_time = time_id + config.seed_init
        logger.start(seed_this_time)
        dataset = make_dataset(config)
        if time_id:
            if config.resplit_each_time:
                train_loader, val_loader, test_loader = make_loaders(dataset, config, seed_this_time)
        else:
            train_loader, val_loader, test_loader = make_loaders(dataset, config, seed_this_time)
        model = MPNN(**config.net).to(config.device)
    
        

if __name__ == "__main__":
    main()

