import sys
sys.path.append("/work02/home/wlluo/halflife")
sys.path.append("/work02/home/wlluo/halflife/experiment/MGA")
sys.path.append("/work02/home/wlluo/halflife/experiment/MGA/pk1HL")

from general_code.utils.argParse import parseArgs
from general_code.utils.log import Logger
from general_code.train.runGNN import train_epoch, eval_epoch
from general_code.model.BaseGNN import MPNN

from dataset_ import make_dataset, make_loaders
from config import make_config
from model import make_model

import numpy as np
import torch
from torch.optim import Adam
from dgllife.utils import EarlyStopping



def main():
    config = make_config("/work02/home/wlluo/halflife/experiment/MGA/pk1HL/config.json", file_path=__file__)
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
        model = make_model(config)
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        stopper = EarlyStopping(
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
            val_result = eval_epoch(config.device, model, val_loader, config.metric)
            logger.cur_dict()["val_mid_result"].append(val_result)
            val_score = np.mean(val_result)
            logger.cur_dict()["val_score"].append(val_score)
            early_stop = stopper.step(val_score, model)
            logger.report_value_score(epoch, stopper.best_score)
            if early_stop:
                break

        stopper.load_checkpoint(model)
        test_result = eval_epoch(config.device, model, test_loader, config.metric)
        train_result = eval_epoch(config.device, model, train_loader, config.metric)
        val_result = eval_epoch(config.device, model, val_loader, config.metric)

        logger.end(train_result, val_result, test_result)
        torch.cuda.empty_cache()
    logger.log()

if __name__ == "__main__":
    main()