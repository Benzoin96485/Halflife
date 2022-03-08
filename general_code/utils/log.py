#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
# std libs
import time

# 3rd party libs
import pandas as pd

# my modules
from general_code.utils.settings import set_random_seed 


class Logger:
    def __init__(self, label_names):
        self.result_pd = pd.DataFrame(columns=(label_names + ['group']) * 3)
        self.start_time = []
        self.stop_time = []

    def start(self, seed):
        self.start_time.append(time.time())
        set_random_seed(seed)
    
        