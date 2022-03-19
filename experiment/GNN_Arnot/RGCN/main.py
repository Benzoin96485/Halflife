from general_code.utils.argParse import parseArgs
from general_code.utils.log import Logger
from dataset import make_dataset, make_loaders
from general_code.model.BaseGNN import MPNN
from torch.optim import Adam
from config import make_config
from dgllife.utils import EarlyStopping
from general_code.train.runGNN import train_epoch, eval_epoch
import numpy as np
import torch

def main():
    config = make_config(parseArgs().config)
    logger = Logger(config)
    for time_id in range(config.eval_times):
        seed_this_time = time_id + config.seed_init
        logger.start(seed_this_time, time_id)
        dataset = make_dataset(config)
        if time_id:
            if config.resplit_each_time:
                train_loader, val_loader, test_loader = make_loaders(dataset, config, seed_this_time)
        else:
            train_loader, val_loader, test_loader = make_loaders(dataset, config, seed_this_time)
        model = MPNN(**config.net_config).to(config.device)
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        stopper = EarlyStopping(
            mode=config.early_stop_mode,
            patience=config.patience,
            filename=config.checkpoint_path,
            metric=config.metric
        )

        for epoch in range(config.num_epochs):
            train_epoch(device=config.device, 
                        epoch=epoch, 
                        model=model,
                        num_epochs=config.num_epochs, 
                        data_loader=train_loader,
                        loss_criterion=config.loss_criterion,
                        optimizer=optimizer,
                        metric=config.metric,
                        logger=logger)
            val_result = eval_epoch(config.device, model, val_loader)
            logger.cur_dict()["val_mid_result"].append(val_result)
            val_score = np.mean(val_result)
            logger.cur_dict()["val_score"].append(val_score)
            early_stop = stopper.step(val_score, model)
            logger.report_value_score(stopper.best_score)
            if early_stop:
                break

        stopper.load_checkpoint(model)
        test_result = eval_epoch(config.device, model, test_loader)
        train_result = eval_epoch(config.device, model, train_loader)
        val_result = eval_epoch(config.device, model, val_loader)

        logger.end(test_result, train_result, val_result)
        torch.cuda.empty_cache()
        

if __name__ == "__main__":
    main()

