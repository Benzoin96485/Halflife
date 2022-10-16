#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
# std libs
import random
import os

# 3rd party libs
import torch
import numpy as np
import dgl

# my modules


def set_random_seed(seed):
    seed = int(seed)
    os.environ["PYTHONSEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dgl.random.seed(seed)


def make_path(old_module: str, new_module: str, file_path: str, add_path: str) -> str:
    dir_list = file_path.split("/")
    old_index = dir_list.index(old_module)
    dir_list[old_index] = new_module
    new_path = "/".join(dir_list[:-1])
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        print(f"make new directory: {new_path}")
    return new_path + "/" + add_path

set_random_seed(0)