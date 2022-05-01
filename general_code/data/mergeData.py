#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define functions for merging different datasets
"""

# Imports
# std libs
# ...

# 3rd party libs
import pandas as pd

def merge_hetero(df_list, label_name_list, new_smi_col, old_smi_col="smiles", label_col="label"):
    assert len(df_list) == len(label_name_list)
    for i, df in enumerate(df_list):
        df.rename(columns={label_col: label_name_list[i]}, inplace=True)
    df0 = df_list[0][[old_smi_col, label_name_list[0]]]
    if len(df_list) > 1:
        for i, df in enumerate(df_list[1:]):
            df0 = pd.merge(left=df0, right=df[[old_smi_col, label_name_list[i+1]]], how="outer", on=old_smi_col)
    df0.rename(columns={old_smi_col: new_smi_col}, inplace=True)
    return df0