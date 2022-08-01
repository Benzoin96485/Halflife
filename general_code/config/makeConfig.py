#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Define the class Config.
'''

# Imports
# std libs
import json

# 3rd party libs
from dgllife.utils import *
from torch.nn import MSELoss

# my modules
from general_code.data.GNNFeaturizer import RGCNBondFeaturizer


class Config():
    """
    A general class for operating the configuration of experiments.
    """
    def __init__(self, config_path: str, format="json"):
        """
        Initialize the config object.


        Arguments:
            config_path: the path of configuration file
            format: the format of the file
        """
        with open(config_path) as f:
            if format == "json":
                self.config = json.load(f)
        for name, group in self.config.items():
            self.__setattr__(name, group)
            if type(group) == type(dict()):
                for key, value in group.items():
                    self.__setattr__(key, value)

    def make_loss_criterion(self, criterion):
        if criterion == "mse":
            self.loss_criterion = MSELoss(reduction="none")



class GNNConfig(Config):
    """
    A general class for operating the configuration of GNN experiments.
    """
    def __init__(self, config_path, format):
        super().__init__(config_path, format)
    
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
