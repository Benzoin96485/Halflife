#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
# std libs
from functools import reduce
import time
import os

# 3rd party libs
import pandas as pd
import matplotlib.pyplot as plt

# my modules
from general_code.utils.env import set_random_seed 


class Logger:
    def __init__(self, config):
        self.config = config
        if hasattr(config, "extra_metrics"):
            self.extra_metric = True
            metrics = [config.metric] + config.extra_metrics
        else:
            self.extra_metric = False
            metrics = [config.metric]
        task_cols = []
        for group in ["train", "val", "test"]:
            for metric in metrics:
                for task_name in config.task_names:
                    task_cols.append(f"{group}_{task_name}_{metric}")
        self.result_pd = pd.DataFrame(columns=task_cols)
        self.total_time = 0
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
        self.report_result(train_result, val_result, test_result)
        self.cur_dict()["time used"] = self.report_time(mode="recent")

    def report_time(self, mode="recent"):
        if mode == "recent":
            elapsed = int(self.end_time - self.start_time)
            self.total_time += elapsed
        else:
            elapsed = int(self.total_time)
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        hms = f"{h:d}:{m:d}:{s:d}"
        print("Time used:", hms)
        return hms

    def report_train_score(self, epoch):
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, self.config.num_epochs, self.config.metric, self.cur_dict()["train_score"][-1]))

    def report_value_score(self, epoch, best_score):
        print('epoch {:d}/{:d}, validation {:.4f}, best validation {:.4f}'.format(
                epoch + 1, self.config.num_epochs,
                self.cur_dict()["val_score"][-1], best_score))

    def report_result(self, train_result, val_result, test_result):
        plt.figure()
        row = reduce(lambda x, y: x+y, train_result) + reduce(lambda x, y: x+y, val_result) + reduce(lambda x, y: x+y, test_result)
        print(row)
        self.result_pd.loc[self.time_id] = row
        print(
            '********************************{}, {}, {}_times_result*******************************'.format(
                self.config.project_name, self.config.experiment_name,
                self.time_id + 1
            )
        )
        # print("train_result:", train_result)
        # print("val_result:", val_result)
        print("test_result:", test_result)
        self.plot_score()
        self.plot_loss()
    
    def plot_score(self):
        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel(f"{self.config.metric}")
        plt.plot(range(len(self.cur_dict()["train_score"])), self.cur_dict()["train_score"], label="train")
        plt.plot(range(len(self.cur_dict()["val_score"])), self.cur_dict()["val_score"], label="val")
        plt.legend()
        if not os.path.exists(self.config.log_path + "/train_val_score/"):
            os.makedirs(self.config.log_path + "/train_val_score/")
        plt.savefig(self.config.log_path + f"/train_val_score/train_val_score_{self.time_id}.png")

    def plot_loss(self):
        plt.figure()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(range(len(self.cur_dict()["loss"])), self.cur_dict()["loss"])
        if not os.path.exists(self.config.log_path + "/loss/"):
            os.makedirs(self.config.log_path + "/loss/")
        plt.savefig(self.config.log_path + f"/loss/loss_{self.time_id}.png")

    def log(self):
        n = len(self.config.task_names)
        if self.extra_metric:
            n *= (len(self.config.extra_metrics) + 1)
        with open(os.path.join(self.config.log_path, "result.txt"), mode="w+") as f:
            f.write("train:\n")
            for task in self.result_pd.columns[:n]:
                print(self.result_pd[task])
                f.write(f"\t{task} mean: {self.result_pd[task].mean()}\n")
                if self.config.eval_times > 1:
                    f.write(f"\t{task} std: {self.result_pd[task].std()}\n")
            f.write("val:\n")
            for task in self.result_pd.columns[n:2*n]:
                f.write(f"\t{task} mean: {self.result_pd[task].mean()}\n")
                if self.config.eval_times > 1:
                    f.write(f"\t{task} std: {self.result_pd[task].std()}\n")
            f.write("test:\n")
            for task in self.result_pd.columns[2*n:3*n]:
                f.write(f"\t{task} mean: {self.result_pd[task].mean()}\n")
                if self.config.eval_times > 1:
                    f.write(f"\t{task} std: {self.result_pd[task].std()}\n")

        self.result_pd.to_csv(os.path.join(self.config.log_path, "result.csv"))