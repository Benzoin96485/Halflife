#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A specific featurizer for GNN like model
"""

# Imports
# std libs
from math import ceil

# 3rd party libs
from rdkit.Chem.rdchem import Mol, BondType, Atom
from rdkit.Chem import MolFromSmiles, RWMol
import numpy as np
from torch import Tensor, int32
from dgllife.utils import smiles_to_bigraph, mol_to_bigraph
from dgl import graph

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
            # if bond.HasProp("HBond"):
            #     if bond.GetProp("HBond") == "donor":
            #         feat = 
            #     elif bond.GetProp("HBond") == "acceptor":
            #         feat = 
            #     else:
            #         raise ValueError
            # else:
            if True:
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
               
                    
def smiles_to_bigraph3(smi, **kwargs):
    if smi is not None:
        return smiles_to_bigraph(smi, **kwargs)
    else:
        return graph(([], []), idtype=int32)


def smiles_to_aqua_bigraph(smi, **kwargs):
    mol = RWMol(MolFromSmiles(smi))
    newOs = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num in (7, 8, 9):
            # charge = atom.GetFormalCharge()
            numH = atom.GetTotalNumHs()
            valence = atom.GetTotalValence()
            index = atom.GetIdx()
            numEP = (8 - valence * 2) // 2
            newOs.append((index, numH, numEP))
        else:
            continue
    for index, numH, numEP in newOs:
        for i in range(numH):
            newidx = mol.AddAtom(Atom(8))
            mol.AddBond(index, newidx)
            bond = mol.GetBondBetweenAtoms(index, newidx)
            bond.SetProp(key="HBond", val="donor")
        for j in range(numEP):
            newidx = mol.AddAtom(Atom(8))
            mol.AddBond(index, newidx)
            bond = mol.GetBondBetweenAtoms(index, newidx)
            bond.SetProp(key="HBond", val="acceptor")
    mol_aqua = mol.GetMol()
    mol_aqua.UpdatePropertyCache()
    return mol_to_bigraph(mol_aqua, canonical_atom_order=False, **kwargs)