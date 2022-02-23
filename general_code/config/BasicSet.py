import torch
from rdkit import Chem
import copy

A_dict_H = {
        "atomic_symbol": {
            "allowable_set": [
                "H", "C", "N", "O", "F",
                "P", "S", "Cl", "Br", "I", "other"
            ],
            "others": True,
            "mode": "onehot"
        },
        "degree": {"mode": "direct"},
        "formal_charge": {"mode": "direct"},
        "num_radical_electron": {"mode": "direct"},
        "hybridization": {
            "allowable_set": [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2, 'other'
            ],
            "others": True,
            "mode": "onehot"
        },
        "numH": {"mode": "direct"},
        "is_aromatic": {"mode": "direct"},
        "is_in_ring": {"mode": "direct"},
        "is_chiral": {"mode": "direct"},
        "stereo": {
            "allowable_set": [
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED
            ],
            "others": True,
            "mode": "onehot"
        },
    }
B_dict_H = {
        "atom_pair": {
            "allowable_set": [
                set("HC"), set("HN"), set("HO"), set("HS"),
                set("CC"), set("CN"), set("CO"), set("CF"), set("CS"), set("CCl"), set("CBr"),
                set("CI"), set("NN"), set("NO"), set("NS"), set("OP"), set("OS"), "others"
            ],
            "others": True,
            "mode": "binseg"
        },
        "bond_type": {
            "allowable_set": [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.AROMATIC,
                              Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                              "others"],
            "others": True,
            "mode": "binseg"
        },
        "is_conjugated": {"mode": "binseg"},
        "is_in_ring": {"mode": "binseg"},
        "stereo": {
            "allowable_set": ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
            "others": True,
            "mode": "binseg"
        }
    }
A_dict_nH = {
        "atomic_symbol": {
            "allowable_set": [
                "C", "N", "O", "F",
                "P", "S", "Cl", "Br", "I", "other"
            ],
            "others": True,
            "mode": "onehot"
        },
        "degree": {"mode": "direct"},
        "formal_charge": {"mode": "direct"},
        "num_radical_electron": {"mode": "direct"},
        "hybridization": {
            "allowable_set": [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2, 'other'
            ],
            "others": True,
            "mode": "onehot"
        },
        "numH": {"mode": "direct"},
        "is_aromatic": {"mode": "direct"},
        "is_in_ring": {"mode": "direct"},
        "is_chiral": {"mode": "direct"},
        "stereo": {
            "allowable_set": [
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED
            ],
            "others": True,
            "mode": "onehot"
        },
    }
B_dict_nH = {
        "atom_pair": {
            "allowable_set": [
                set("CC"), set("CN"), set("CO"), set("CF"), set("CS"), set("CCl"), set("CBr"),
                set("CI"), set("NN"), set("NO"), set("NS"), set("OP"), set("OS"), "others"
            ],
            "others": True,
            "mode": "binseg"
        },
        "bond_type": {
            "allowable_set": [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.AROMATIC,
                              Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                              "others"],
            "others": True,
            "mode": "binseg"
        },
        "is_conjugated": {"mode": "binseg"},
        "is_in_ring": {"mode": "binseg"},
        "stereo": {
            "allowable_set": ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
            "others": True,
            "mode": "binseg"
        }
    }

settings_RGCN = {
    "model_type": "RGCN",
    "readout_type": "weight_and_sum",
    "log_folder": "../logs/",
    "atom_feat_dict": A_dict_nH,
    "atom_concat_mode": "onehot",
    "bond_feat_dict": B_dict_nH,
    "bond_concat_mode": "binseg",
    "explicitH": False,
    "label_name_list": ["lg_HL"],
    "smiles_name": "smiles_nonchiral",
    "mask_value": 114514,
    "origin_path": "/data01/wlluo/halflife/data/OA_std.csv",
    "splited_path": "/data01/wlluo/halflife/data/OA_std_split.csv",
    "bin_path": "/data01/wlluo/halflife/data/OA_std_1.bin",
    "result_folder": "../results/",
    "renew_dataset": True,
    "renew_graph": True,
    "partition_ratio": (8, 1, 1),
    "eval_times": 10,
    "task_name": "my_halflife",
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    "batch_size": 128,
    "loss_r": torch.nn.MSELoss(reduction='none'),
    "gnn_hidden_feats": [64, 128],
    "fcn_hidden_feats": [128, 128],
    "edge_nn_feats": [64, 128],
    "fcn_in_feats_extra": 0,
    "rgcn_loop": False,
    "gnn_residual": True,
    "gnn_bn": True,
    "rgcn_dropout": 0.2,
    "fcn_dropout": 0.2,
    "edge_nn_dropout": 0.2,
    "n_conv_step": 2,
    "mpnn0_residual": True,
    "specific_weighting": True,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 100,
    "earlystop_mode": "higher",
    "model_folder": "/data01/wlluo/halflife/model/",
    "pretrained_path": "",
    "num_epochs": 1000,
    "regression_metric_name": "r2",
}

settings_GGNN = copy.deepcopy(settings_RGCN)
settings_GGNN.update(
    {
        "model_type": "GGNN",
        "batch_size": 64,
        "gnn_hidden_feats": [128],
        "weight_decay": 1e-4,
    }
)


settings_MPNN0 = copy.deepcopy(settings_RGCN)
settings_MPNN0.update(
    {
        "model_type": "MPNN0",
        "lr": 0.003,
        "bond_feat_dict": {
            "atom_pair": {
                "allowable_set": [
                    set("HC"), set("HN"), set("HO"), set("HS"),
                    set("CC"), set("CN"), set("CO"), set("CF"), set("CS"), set("CCl"), set("CBr"),
                    set("CI"), set("NN"), set("NO"), set("NS"), set("OP"), set("OS"), "others"
                ],
                "others": True,
                "mode": "onehot"
            },
            "bond_level": {
                "mode": "direct"
            },
            "is_conjugated": {"mode": "direct"},
            "is_in_ring": {"mode": "direct"},
            "stereo": {
                "allowable_set": ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
                "others": True,
                "mode": "onehot"
            }
        },
        "bond_concat_mode": "onehot",
        "renew_dataset": True
    }
)

