from general_code.utils.env import set_random_seed
from general_code.utils.makeConfig import parseArgs, RGCNConfig
from general_code.utils.log import Logger
from general_code.train.runGNN import train_epoch, eval_epoch
from general_code.data.buildDataset import csv2df, make_loaders, make_dataset, collate_molgraphs

from model import make_model

import numpy as np
import os
import torch
from torch.optim import Adam
from dgllife.utils import EarlyStopping

def main(config):
    logger = Logger(config)
    for time_id in range(config.eval_times):
        seed_this_time = time_id + config.seed_init
        logger.start(seed_this_time, time_id)
        set_random_seed(seed_this_time)
        dfs = csv2df(config, splitted=True, inner_path=f"fold_{time_id}")
        dataset = make_dataset(dfs, config, splitted=True, seed_this_time=seed_this_time)
        train_loader, val_loader, test_loader = make_loaders(dataset, config.batch_size, collate_molgraphs)
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
            val_result = eval_epoch(config.device, model, val_loader, config.metric)[0]
            logger.cur_dict()["val_mid_result"].append(val_result)
            val_score = np.mean(val_result)
            logger.cur_dict()["val_score"].append(val_score)
            early_stop = stopper.step(val_score, model)
            logger.report_value_score(epoch, stopper.best_score)
            if early_stop:
                break

        stopper.load_checkpoint(model)
        if hasattr(config, "extra_metrics"):
            extra_metrics = config.extra_metrics
        test_result = eval_epoch(config.device, model, test_loader, config.metric, extra_metrics)
        train_result = eval_epoch(config.device, model, train_loader, config.metric, extra_metrics)
        val_result = eval_epoch(config.device, model, val_loader, config.metric, extra_metrics)

        logger.end(train_result, val_result, test_result)
        torch.cuda.empty_cache()
    logger.log()

if __name__ == "__main__":
    config = RGCNConfig(parseArgs().config, file_path=__file__)
    set_random_seed(config.seed_init)
    main(config)

