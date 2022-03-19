#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
# std libs
import time

# 3rd party libs
import pandas as pd

# my modules
from general_code.utils.env import set_random_seed 


class Logger:
    def __init__(self, config):
        self.config = config
        # self.result_pd = pd.DataFrame(columns=(config.label_name_list + ['group']) * 3)
        self.start_time = 0
        self.end_time = 0
        self.result_list = []
        self.time_id = 0

    def cur_dict(self):
        return self.result_list[-1]

    def start(self, seed, time_id):
        self.time_id = time_id
        self.result_list.append(dict())
        self.cur_dict().update({"loss": [], "train_score": [], "val_score": [], "val_mid_result": [], "seed": seed, "time_id": time_id})
        set_random_seed(seed)
        self.start_time = time.time()
        
    def end(self, train_result, val_result, test_result):
        self.end_time = time.time()
        self.time_list.append(self.end_time - self.start_time)
        self.report_result(train_result, val_result, test_result)
        self.cur_dict()["time used"] = self.report_time(mode="recent")

    def report_time(self, mode="recent"):
        if mode == "recent":
            elapsed = self.time_list[-1]
        elif mode == "total":
            elapsed = sum(self.time_list[-1])
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        hms = f"{h:d}:{m:d}:{s:d}"
        print("Time used:", hms)
        return hms

    def report_train_score(self, epoch):
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, self.config.num_epochs, self.config.metric, self.cur_dict["train_score"][-1]))

    def report_value_score(self, epoch, best_score):
        print('epoch {:d}/{:d}, validation {:.4f}, best validation {:.4f}'.format(
                epoch + 1, self.config.num_epochs,
                self.cur_dict["val_score"][-1]) + ' validation result:', self.cur_dict["val_result"][-1])

    def report_result(self, train_result, val_result, test_result):
        print(
            '********************************{}, {}, {}_times_result*******************************'.format(
                self.config.project_name, self.config.experiment_name,
                self.time_id + 1
            )
        )
        print("train_result:", train_result)
        print("val_result:", val_result)
        print("test_result:", test_result)