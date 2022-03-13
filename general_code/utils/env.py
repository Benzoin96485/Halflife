#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
# std libs
import random

# 3rd party libs
import torch
import numpy as np

# my modules


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
