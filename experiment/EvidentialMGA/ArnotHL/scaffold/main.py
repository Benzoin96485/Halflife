from general_code.utils.env import set_random_seed
from general_code.utils.makeConfig import parseArgs, RGCNConfig
from general_code.utils.log import Logger
from general_code.train.runGNN import train_epoch, eval_epoch
from general_code.train.runNN import EvidentialCalibration
from general_code.data.buildDataset import csv2df, make_loaders, make_dataset, collate_molgraphs

from model import make_model

import numpy as np
import torch
import pandas as pd
from torch.optim import Adam
from dgllife.utils import EarlyStopping
from copy import deepcopy

def main():
    config = RGCNConfig(parseArgs().config, file_path=__file__)
    set_random_seed(config.seed_init)
    logger = Logger(config)
    for time_id in range(config.eval_times):
        seed_this_time = time_id + config.seed_init
        set_random_seed(seed_this_time)
        dfs = csv2df(config, splitted=True, inner_path=f"fold_{time_id}")
        dataset = make_dataset(dfs, config, splitted=True, seed_this_time=seed_this_time)
        train_loader, val_loader, test_loader = make_loaders(dataset, config.batch_size, collate_molgraphs)
        
        lamda_list = [0.6, 0.65, 0.7, 0.75, 0.8]
        best_calibrate = 1
        best_lamda = 0
        test_result_dict = None
        for lamda in lamda_list:
            tmp_logger = Logger(config)
            tmp_logger.start(seed_this_time, time_id)
            model = make_model(config)
            optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            stopper = EarlyStopping(
                patience=config.patience,
                filename=config.checkpoint_path,
                metric=config.metric
            )
            config.lamda = lamda
            config.make_loss_criterion(config.loss_name)
            EC = EvidentialCalibration()
            EC2 = EvidentialCalibration()

            for epoch in range(config.num_epochs):
                train_epoch(device=config.device, 
                            epoch=epoch, 
                            model=model,
                            num_epochs=config.num_epochs, 
                            data_loader=train_loader,
                            loss_criterion=config.loss_criterion,
                            optimizer=optimizer,
                            metric=config.metric,
                            logger=tmp_logger)
                tmp_val_result = eval_epoch(config.device, model, val_loader, config.metric)[0]
                tmp_logger.cur_dict()["val_mid_result"].append(tmp_val_result)
                val_score = np.mean(tmp_val_result)
                tmp_logger.cur_dict()["val_score"].append(val_score)
                if epoch > config.warmup_epochs:
                    early_stop = stopper.step(val_score, model)
                    tmp_logger.report_value_score(epoch, stopper.best_score)
                else:
                    early_stop=False
                    tmp_logger.report_value_score(epoch, warmup=True)
                if early_stop:
                    break

            stopper.load_checkpoint(model)
            if hasattr(config, "extra_metrics"):
                extra_metrics = config.extra_metrics
            
            tmp_val_result = eval_epoch(config.device, model, val_loader, config.metric, extra_metrics, give_result=True, evidential=True, result_dict=EC.result_dict)
            tmp_test_result = eval_epoch(config.device, model, test_loader, config.metric, extra_metrics, give_result=True, evidential=True, result_dict=EC2.result_dict)
            tmp_train_result = eval_epoch(config.device, model, train_loader, config.metric, extra_metrics)
            calibrate = EC.calibrate()
            print(f"Calibration at lambda={config.lamda} is {calibrate}")
            if abs(calibrate) >= best_calibrate:
                continue
            else:
                best_calibrate = abs(calibrate)
                best_lamda = lamda
                train_result = tmp_train_result
                val_result = tmp_val_result
                test_result = tmp_test_result
                test_result_dict = EC2.result_dict.copy()
                logger = deepcopy(tmp_logger)
            
        print(f"best lamda={best_lamda}")
        pd.DataFrame(test_result_dict).to_csv("test_result2.csv")
        logger.end(train_result, val_result, test_result)
        torch.cuda.empty_cache()
    logger.log()

if __name__ == "__main__":
    main()

