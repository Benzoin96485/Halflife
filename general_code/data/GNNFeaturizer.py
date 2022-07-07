#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A specific featurizer for GNN like model
"""

# Imports
# std libs
from math import ceil

# 3rd party libs
from rdkit.Chem.rdchem import Mol, BondType
import numpy as np
from torch import Tensor

# my modules
# ...


ALLOWABLE_SET = {
    "atom_pair": [
        set("CC"), set("CN"), set("CO"), set("CF"), set("CS"), 
        set("CCl"), set("CBr"), set("CI"), set("NN"), set("NO"), 
        set("NS"), set("OP"), set("OS"), "others"
    ],
    "bond_type": [
        BondType.SINGLE, BondType.AROMATIC,
        BondType.DOUBLE, BondType.TRIPLE, "others"
    ],
    "stereo": [
        "STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "others"
    ],
    "is_conjugated": [True, False],
    "is_in_ring": [True, False]
}


def calc_full_types(types):
    return 2 ** ceil(np.log2(len(types)))


class RGCNBondFeaturizer():
    def __init__(self, bond_feat_types):
        self.feat_bases = dict()
        self.feat_size_num = 1
        self.bond_feat_types = bond_feat_types
        for feat_type in bond_feat_types:
            self.feat_bases[feat_type] = self.feat_size_num
            self.feat_size_num *= len(ALLOWABLE_SET[feat_type])
            

    def __call__(self, mol: Mol):
        to_tensor = []
        for i in range(mol.GetNumBonds()):
            feat = 0
            bond = mol.GetBondWithIdx(i)
            bond_feat_all = {
                "atom_pair": {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()},
                "bond_type": bond.GetBondType(),
                "is_conjugated": bond.GetIsConjugated(),
                "is_in_ring": bond.IsInRing(),
                "stereo": str(bond.GetStereo())
            }
            for feat_type in self.bond_feat_types:
                if bond_feat_all[feat_type] not in ALLOWABLE_SET[feat_type]:
                    assert ALLOWABLE_SET[feat_type][-1] == "others"
                    feat += ALLOWABLE_SET[feat_type].index("others") * self.feat_bases[feat_type]
                else:
                    feat += ALLOWABLE_SET[feat_type].index(bond_feat_all[feat_type]) * self.feat_bases[feat_type]
            to_tensor.append(feat)
            to_tensor.append(feat)
        return {"bond": Tensor(to_tensor).int()}

    def feat_size(self):
        return self.feat_size_num
                    
