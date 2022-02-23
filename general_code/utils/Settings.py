import os
import random
from math import ceil
import torch
import numpy as np
import sklearn


def calc_dim(settings, key1, key2):
    ans = 0
    if settings[key2] == "onehot":
        for val_dict in settings[key1].values():
            if "allowable_set" in val_dict:
                ans += len(val_dict["allowable_set"])
            else:
                ans += 1
    elif settings[key2] == "binseg":
        for val_dict in settings[key1].values():
            if "allowable_set" in val_dict:
                wide = len(val_dict["allowable_set"])
                ans += ceil(np.log2(wide))
            else:
                ans += 1
        ans = 1 << ans
    return ans


def check_settings_rgcn(settings):
    settings["n_tasks"] = len(settings["label_name_list"])
    if not (os.path.exists(settings["splited_path"]) and os.path.exists(settings["bin_path"])):
        settings["renew_dataset"] = True

    settings["gnn_in_feats"] = calc_dim(settings, "atom_feat_dict", "atom_concat_mode")
    settings["num_rels"] = calc_dim(settings, "bond_feat_dict", "bond_concat_mode")
    settings["pretrained_path"] = settings["model_path"] = settings["model_folder"] + settings["task_name"] + ".pth"
    for key in settings:
        if "path" in key:
            dir_path = os.path.dirname(settings[key])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        elif "folder" in key:
            if not os.path.exists(settings[key]):
                os.makedirs(settings[key])

    settings["gnn_out_feats"] = settings["gnn_hidden_feats"][-1]
    settings["fcn_in_feats"] = settings["gnn_out_feats"] + settings["fcn_in_feats_extra"]


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


