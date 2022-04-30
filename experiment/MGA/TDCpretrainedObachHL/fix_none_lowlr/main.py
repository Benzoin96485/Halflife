from general_code.utils.argParse import parseArgs
from general_code.utils.log import Logger
from general_code.utils.env import make_path
from general_code.train.runGNN import train_epoch, eval_epoch
from general_code.model.BaseGNN import MPNN

from dataset import make_dataset, make_loaders
from config import make_config
from model import make_model

import numpy as np
import torch
from torch.optim import Adam
from dgllife.utils import EarlyStopping



def main():
    config = make_config(parseArgs().config)
    config.log_path = make_path("experiment", "log", __file__, "")
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
        model.load_state_dict(torch.load(config.pretrain_path), strict=False)
        gnn_params, other_params = [], []
        for name, param in model.named_parameters():
            if "gnn" in name.lower():
                gnn_params.append(param)
            else:
                other_params.append(param)
        optimizer = Adam(
            params=[
                {"params": gnn_params, "lr": config.pretrain_lr},
                {"params": other_params}
            ],
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
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
