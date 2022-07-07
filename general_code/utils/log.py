#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
# std libs
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
        self.result_pd = pd.DataFrame(columns=(config.task_names + ['group']) * 3)
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
        self.result_pd.loc[self.time_id] = train_result + ["train"] + val_result + ["val"] + test_result + ["test"]
        print(
            '********************************{}, {}, {}_times_result*******************************'.format(
                self.config.project_name, self.config.experiment_name,
                self.time_id + 1
            )
        )
        print("train_result:", train_result)
        print("val_result:", val_result)
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
        with open(self.config.log_path + "/result.txt") as f:
            f.write("train:")
            for task in self.result_pd.columns[:n]:
                f.write(f"\t{task} mean: {self.result_pd[task].mean()}")
                f.write(f"\t{task} std: {self.result_pd[task].std()}")
            f.write("val:")
            for task in self.result_pd.columns[n+1:2*n+1]:
                f.write(f"\t{task} mean: {self.result_pd[task].mean()}")
                f.write(f"\t{task} std: {self.result_pd[task].std()}")
            f.write("test:")
            for task in self.result_pd.columns[2*n+2:3*n+2]:
                f.write(f"\t{task} mean: {self.result_pd[task].mean()}")
                f.write(f"\t{task} std: {self.result_pd[task].std()}")

        self.result_pd.to_csv(self.config.log_path + "/result.csv")