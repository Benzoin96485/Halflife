from functools import reduce
import dgl
from dgl import DGLGraph
from rdkit.Chem import MolFromSmiles
from rdkit import Chem
import numpy as np
import torch
import pandas as pd
from dgl.data.graph_serialize import save_graphs, load_graphs
from sklearn.utils import shuffle
from math import ceil

dtype_dict = {"onehot": torch.float32, "binseg": torch.int32}


def one_of_k_encoding(x, allowable_set=None, others=True, mode="onehot"):
    """
    :param x: attribute for encoding
    :param allowable_set: list: range of x for encoding
    :param others: if allowable set contains "others" at the last
    :param mode:
        "onehot": return a one-hot list of bool, whose length is equal to
            allowable set
        "binseg": return a bit string, whose length is equal to effective length
            of binary form of len(allowable_set)
        "direct": return x
    :return: depend on mode
    """
    if allowable_set and x not in allowable_set:
        if others:
            x = allowable_set[-1]
        else:
            raise Exception("input {0} not in allowable set:{1}".format(
                x, allowable_set))

    if mode == "onehot":
        return [int(x == s) for s in allowable_set]
    elif mode == "binseg":
        if type(x) == bool:
            return str(int(x))
        else:
            wide = len(allowable_set)
            index = allowable_set.index(x)
            binseg = bin(index)[2:].zfill(ceil(np.log2(wide)))
            return binseg
    elif mode == "direct":
        if type(x) == bool:
            return [int(x)]
        else:
            return [x]
    else:
        raise KeyError(f"{mode}")


def concat_encoding(encoding_list, mode="onehot"):
    """
    :param encoding_list: list of several encodings with particular mode
    :param mode: mode matching encoding list
        "onehot": add from left to right
        "binseg": add from right to left
    :return: concatenated encoding
    """
    if mode == "onehot":
        return list(map(float, reduce(lambda x, y: x + y, encoding_list)))
    elif mode == "binseg":
        return int(reduce(lambda x, y: y + x, encoding_list), base=2)
    else:
        raise KeyError(mode)


def atom_features(atom, atom_feat_dict, atom_concat_mode="onehot", **kwargs):
    """
    :param atom: the atom with feature to encode
    :param atom_feat_dict:
        {
            key in atom_feat_all:
            tuple (allowable set, others, mode)
        }
    :param atom_concat_mode: concatenate mode in concat_encoding
    :return: atom feature
    """
    encoding_list = []
    atom_feat_all = {
        "atomic_symbol": atom.GetSymbol(),
        "degree": atom.GetDegree(),
        "formal_charge": atom.GetFormalCharge(),
        "num_radical_electron": atom.GetNumRadicalElectrons(),
        "hybridization": atom.GetHybridization(),
        "mass": atom.GetMass(),
        "numH": atom.GetTotalNumHs(),
        "is_aromatic": atom.GetIsAromatic(),
        "is_in_ring": atom.IsInRing(),
        "is_chiral": atom.HasProp('_ChiralityPossible'),
        "stereo": atom.GetChiralTag()
    }
    for feat, feat_dict in atom_feat_dict.items():
        encoding_list.append(one_of_k_encoding(atom_feat_all[feat], **feat_dict))
    return concat_encoding(encoding_list, mode=atom_concat_mode)


def bond_features(bond, bond_feat_dict, bond_concat_mode="bitseg", **kwargs):
    """
    :param bond: the bond with feature to encode
    :param bond_feat_dict: feat_dict:
        {
            key in bond_feat_all:
            tuple (allowable set, others, mode)
        }
    :param bond_concat_mode: concatenate mode in concat_encoding
    :return: bond feature
    """
    encoding_list = []
    bond_feat_all = {
        "atom_pair": {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()},
        "bond_type": bond.GetBondType(),
        "bond_level": bond.GetBondTypeAsDouble(),
        "is_aromatic": bond.GetIsAromatic(),
        "is_conjugated": bond.GetIsConjugated(),
        "is_in_ring": bond.IsInRing(),
        "stereo": str(bond.GetStereo())
    }
    for feat, feat_dict in bond_feat_dict.items():
        encoding_list.append(one_of_k_encoding(bond_feat_all[feat], **feat_dict))
    return concat_encoding(encoding_list, mode=bond_concat_mode)


def smiles2bigraph(smiles, explicitH=False, **kwargs):
    """
    transform smiles to bigraph
    :param smiles: smiles of molecule to transform
    :param explicitH: use graph contains explicitH
    :return: bigraph
    """
    g = DGLGraph()
    mol = MolFromSmiles(smiles)
    if explicitH:
        mol = Chem.AddHs(mol)
    g.add_nodes(mol.GetNumAtoms())
    atoms_feature_all = []
    for atom in mol.GetAtoms():
        atoms_feature_all.append(atom_features(atom, **kwargs))
    g.ndata["atom"] = torch.tensor(atoms_feature_all)
    src_list, dst_list = [], []
    bond_feature_all = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        bond_feature = bond_features(bond, **kwargs)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_feature_all.append(bond_feature)
        bond_feature_all.append(bond_feature)
    g.add_edges(src_list, dst_list)
    g.edata["bond"] = torch.tensor(bond_feature_all)
    return g


def build_mask(label_value_list, mask_value=114514, **kwargs):
    """
    :param label_value_list: values of one label
    :param mask_value: mask value of the label
    :return: list of 0 or 1: if masked
    """
    return [int(i != mask_value) for i in label_value_list]


def build_dataset(df_smiles, label_name_list, smiles_name="smiles", **kwargs):
    """
    :param df_smiles: Dataframe with smiles, labels and splited groups
    :param label_name_list: list of used label names
    :param smiles_name: name of column of smiles
    :return:
    """
    dataset = []
    failed_molecule = []
    labels = df_smiles[label_name_list]
    split_index = df_smiles['group']
    smilesList = df_smiles[smiles_name]
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        try:
            g = smiles2bigraph(smiles, **kwargs)
            mask = build_mask(labels.loc[i], **kwargs)
            molecule = [smiles, g, labels.loc[i], mask, split_index.loc[i]]
            dataset.append(molecule)
            print('{}/{} molecule is transformed!'.format(i + 1, molecule_number))
        except ValueError:
            print('{} is transformed failed!'.format(smiles))
            molecule_number = molecule_number - 1
            failed_molecule.append(smiles)
    print('{}({}) is transformed failed!'.format(failed_molecule, len(failed_molecule)))
    return dataset


def random_split(origin_path, splited_path, partition_ratio=(8, 1, 1), **kwargs):
    """
    random split dataset according to partition ratio
    :param origin_path: unsplited dataset path
    :param splited_path: splited dataset path for saving
    :param partition_ratio: tuple (train, val, test)
    """
    df = shuffle(pd.read_csv(origin_path))
    size = len(df)
    first = partition_ratio[0]
    mid_sum = first + partition_ratio[1]
    ratio_sum = mid_sum + partition_ratio[2]
    train_seq = [x for x in range(size) if x % ratio_sum in range(first)]
    val_seq = [x for x in range(size) if x % ratio_sum in range(first, mid_sum)]
    test_seq = [x for x in range(size) if x % ratio_sum in range(mid_sum, ratio_sum)]
    train_index = df.iloc[train_seq].index
    val_index = df.iloc[val_seq].index
    test_index = df.iloc[test_seq].index
    df.loc[train_index, 'group'] = 'train'
    df.loc[val_index, 'group'] = 'val'
    df.loc[test_index, 'group'] = 'test'
    df = df.reset_index()
    df.to_csv(splited_path)


def build_and_save(splited_path, bin_path, **kwargs):
    """
    build graph-like dataset and save
    :param splited_path: origin grouped dataset csv path
    :param bin_path: molgraphs bin save path
    """
    data_origin = pd.read_csv(splited_path, index_col=0)
    data_origin = data_origin.fillna(kwargs["mask_value"])
    dataset = build_dataset(df_smiles=data_origin, **kwargs)

    smiles, graphs, labels, mask, split_index = map(list, zip(*dataset))
    graph_labels = {'labels': torch.tensor(labels), 'mask': torch.tensor(mask)}
    save_graphs(bin_path, graphs, graph_labels)
    print('Molecules graph is saved!')


def load_graph(splited_path, bin_path, **kwargs):
    smiles = pd.read_csv(splited_path, index_col=None).smiles.values
    group = pd.read_csv(splited_path, index_col=None).group.to_list()
    graphs, detailed_information = load_graphs(bin_path)
    labels = detailed_information['labels']
    mask = detailed_information['mask']
    notuse_mask = torch.mean(mask.float(), 1).numpy().tolist()
    use_index = [index for index, notuse in enumerate(notuse_mask) if notuse != 0]
    sets = []
    for part in ["train", "val", "test"]:
        index = [index for index in use_index if group[index] == part]
        smiles_part = smiles[index]
        graphs_part = np.array(graphs)[index]
        labels_part = labels.numpy()[index]
        mask_part = mask.numpy()[index]
        sets.append(list(map(list, zip(smiles_part, graphs_part, labels_part, mask_part))))
    return tuple(sets)


def collate_molgraphs(data):
    smiles, graphs, labels, mask = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.tensor(labels)
    mask = torch.tensor(mask)
    return smiles, bg, labels, mask
