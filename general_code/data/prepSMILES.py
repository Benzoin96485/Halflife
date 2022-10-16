#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define functions for checking SMILES in datasets
"""

# Imports
# std libs
from functools import reduce

# 3rd party libs
from chembl_structure_pipeline import standardize_mol, get_parent_mol
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles, GetScaffoldForMol
from rdkit.Chem import MolFromSmiles, DeleteSubstructs, MolToSmiles
from rdkit import Chem
import pandas as pd

ALLOW_ATOM_TYPE = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "I"]

class SmiChecker:
    def check(self, smi, name, **kwargs):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                if type(name) == type([]):
                    return reduce(lambda x, y: x and y, [getattr(self, f"check_{n}")(mol, **kwargs) for n in name])
                else:
                    return getattr(self, f"check_{name}")(mol, **kwargs)
            else:
                return False
        except:
            return False

    def check_atom_type(self, mol, atom_allow_type=None, **kwargs):
        if not atom_allow_type:
            atom_allow_type = ALLOW_ATOM_TYPE
        atoms = mol.GetAtoms()
        for atom in atoms:
            if atom.GetSymbol() not in atom_allow_type:
                return False
        return True


def smi_cleaner(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = get_parent_mol(mol)[0]
        mol = standardize_mol(mol)
        return Chem.MolToSmiles(mol)
    except:
        return smi


def list_atom_types(df, smi_col):
    atom_dict = {}
    for smi in df[smi_col]:
        try:
            mol = Chem.MolFromSmiles(smi)
            atoms = mol.GetAtoms()
            for atom in atoms:
                symbol = atom.GetSymbol()
                if symbol in atom_dict:
                    atom_dict[symbol] += 1
                else:
                    atom_dict[symbol] = 1
        except:
            print(smi)
    return atom_dict


def standardize_df(df: pd.DataFrame, smi_col):
    df[smi_col] = df[smi_col].apply(smi_cleaner)
    return df


def wash_df(df: pd.DataFrame, smi_col, check_name, **kwargs):
    old_cols = df.columns
    S = SmiChecker()
    df["smi_check"] = df[smi_col].apply(lambda x: S.check(x, check_name, **kwargs))
    return df[df["smi_check"]][old_cols].copy()


def scaffold_extract(smi):
    return MurckoScaffoldSmiles(smiles=smi, includeChirality=False)


def sidechain_extract(smi):
    mol = MolFromSmiles(smi)
    scaffold_patt = GetScaffoldForMol(mol)
    sidechain_mol = DeleteSubstructs(mol, scaffold_patt)
    return MolToSmiles(sidechain_mol)

def scaffold_decomp_df(df: pd.DataFrame, smi_col="smiles"):
    df[smi_col + "_scaffold"] = df[smi_col].apply(scaffold_extract)
    df[smi_col + "_sidechain"] = df[smi_col].apply(sidechain_extract)