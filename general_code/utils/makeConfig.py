#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Define the class Config and its derivatives for experiment configuration.
'''

# Imports
# std libs
import json
import argparse
import random
import os

# 3rd party libs
from dgllife.utils import *
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount
from torch.cuda import is_available
import torch
import numpy as np
import dgl

# my modules
from general_code.data.GNNFeaturizer import RGCNBondFeaturizer
from general_code.model.loss import NIGLoss, CGCELoss


def unfold(d: dict) -> dict:
    """
    Unfold nested dictionary.

    Params:
    -----
    `d`: A nested dictionary

    Returns:
    ----
    An unfolded dictionary
    """
    def _unfold(d):
        kvlist = []
        for key, value in d.items():
            kvlist.append((key, value))
            if isinstance(value, dict):
                kvlist.extend(_unfold(value))
        return kvlist
    return dict(_unfold(d))


def parseArgs():
    """
    Parse the configurations of the experiment from the command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default="config.json",
        help="The path of the configuration file"
    )
    flags, unparsed = parser.parse_known_args()
    return flags


class Config():
    """
    A general class for operating the configuration of molecular ML experiments.
    """
    def __init__(self, config_path: str=None, file_path: str=None):
        """
        Initialize the config object.

        Params:
        -----
        `config_path`: the path of configuration file with json format
        """
        self.project_name = "My project"
        self.experiment_name = "My experiment"
        self.data_path = ""
        self.smi_col = "smiles"
        self.seed_init = 114514
        self.task_names = ["task"]
        self.split_method = "random"
        self.split_ratio = (8, 1, 1)
        self.use_splitted = True
        self.split_data_path = ""
        self.train_data = ""
        self.test_data = ""
        self.val_data = ""

        if config_path:
            with open(config_path) as f:
                self.config = json.load(f)
            self.config = unfold(self.config)
            for key, value in self.config.items():
                self.__setattr__(key, value)

        self.log_path = self.make_path("experiment", "log", file_path, "")
        self.checkpoint_path = self.make_path("experiment", "log", file_path, "checkpoint.pth")
        self.cache_path = self.make_path("experiment", "log", file_path, "graph.bin")

    def make_loss_criterion(self, criterion):
        if criterion == "mse":
            self.loss_criterion = MSELoss(reduction="none")
        elif criterion == "evidential":
            self.loss_criterion = NIGLoss(reduction="none", lamda=self.lamda)
        elif criterion == "bcel":
            self.loss_criterion = BCEWithLogitsLoss(reduction="none")
        elif criterion == "ce":
            self.loss_criterion = CrossEntropyLoss(reduction="none")
        elif criterion == "cgce":
            self.loss_criterion = CGCELoss(reduction="none", n_tasks=len(self.task_names), thresholds=self.classify_threshold)

    def set_random_seed(self, seed=None):
        if seed is None:
            seed = self.seed_init
        seed = int(seed)
        random.seed(seed)
        os.environ["PYTHONSEED"] = str(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dgl.random.seed(seed)

    def make_path(self, old_module: str, new_module: str, file_path: str, add_path: str) -> str:
        if file_path:
            dir_list = file_path.split("/")
            old_index = dir_list.index(old_module)
            dir_list[old_index] = new_module
            new_path = "/".join(dir_list[:-1])
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                print(f"make new directory: {new_path}")
            return new_path + "/" + add_path
        else:
            raise FileNotFoundError


class NNConfig(Config):
    """
    A general class for operating the configuration of NN experiments.
    """
    def __init__(self, config_path: str=None, file_path: str=None):
        self.pretrain_path = ""
        self.eval_times = 1
        self.batch_size = 128
        self.gpu = True
        self.cuda_index = -1
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.metric = ""
        self.extra_metrics = []
        self.loss_name = ""
        self.earlystop_mode = ""
        self.num_epochs = 100
        self.patience = 10
        self.earlystop_mode = "higher"
        super().__init__(config_path, file_path)
        
        if self.gpu and is_available():
            nvmlInit()
            gpu_count = nvmlDeviceGetCount()
            if self.cuda_index == -1:
                min_memuse = 2 << 40
                self.cuda_index = 0
                for i in range(gpu_count):
                    nvmlInit()
                    handle=nvmlDeviceGetHandleByIndex(i)
                    meminfo=nvmlDeviceGetMemoryInfo(handle)
                    memuse = meminfo.used
                    if memuse < min_memuse:
                        min_memuse = memuse
                        self.cuda_index = i
                print(f"device: cuda: {self.cuda_index}")
            if self.cuda_index < gpu_count:
                self.device = f"cuda:{self.cuda_index}"
            else:
                self.device = f"cuda:0"
        else:
            self.device = "cpu"


class GNNConfig(NNConfig):
    """
    A general class for operating the configuration of GNN experiments.
    """
    def __init__(self, config_path: str=None, file_path: str=None):
        self.graph_type = "bigraph"
        self.atom_feat_name = "dgllife_canonical"
        self.bond_feat_name = "dgllife_canonical"
        self.bond_feat_type = []
        self.extra_fcn_nfeats = 0
        self.gnn_hidden_nfeats = [64, 128]
        self.readout_type = "weight_and_sum"
        self.rgcn_loop = True
        self.rgcn_dropout = 0.2
        self.fcn_hidden_nfeats = [128, 128]
        super().__init__(config_path, file_path)
        self.make_atom_featurizer(self.atom_feat_name)
        self.make_bond_featurizer(self.bond_feat_name, self.bond_feat_type)
        self.make_loss_criterion(self.loss_name)
        self.gnn_in_nfeats = self.atom_feat_size
        self.fcn_in_nfeats = self.gnn_hidden_nfeats[-1] + self.extra_fcn_nfeats
    
    def make_atom_featurizer(self, name, atom_feat_type=[]):
        if name == "dgllife_canonical":
            self.atom_featurizer = CanonicalAtomFeaturizer(atom_data_field="atom")
        elif name == "AttentiveFP":
            self.atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field="atom")
        elif name == "Custom":
            self.atom_featurizer = BaseAtomFeaturizer({"atom": ConcatFeaturizer(atom_feat_type)})
        if hasattr(self, "atom_featurizer"):
            self.atom_feat_size = self.atom_featurizer.feat_size()

    def make_bond_featurizer(self, name, bond_feat_type=[]):
        if name == "dgllife_canonical":
            self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field="bond")
        elif name == "AttentiveFP":
            self.bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field="bond")
        elif name == "Relation":
            self.bond_featurizer = RGCNBondFeaturizer(bond_feat_type)
        elif name == "Custom":
            self.bond_featurizer = BaseBondFeaturizer({"bond": ConcatFeaturizer(bond_feat_type)})
        if hasattr(self, "bond_featurizer"):
            self.bond_feat_size = self.bond_featurizer.feat_size()


class RGCNConfig(GNNConfig):
    """
    A general class for operating the configuration of GNN experiments.
    """
    def __init__(self, config_path: str=None, file_path: str=None):
        super().__init__(config_path, file_path)
        self.num_rels = self.bond_featurizer.feat_size()