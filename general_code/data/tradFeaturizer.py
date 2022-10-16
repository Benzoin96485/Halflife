from deepchem.feat import RDKitDescriptors, MordredDescriptors, MACCSKeysFingerprint, PubChemFingerprint, Mol2VecFingerprint, CircularFingerprint

descriptors = {
    "rdkit": RDKitDescriptors,
    "mordred": MordredDescriptors
}

fingerprints = {
    "maccs": (MACCSKeysFingerprint, 167),
    "pubchem": (PubChemFingerprint, 881),
    "mol2vec": (Mol2VecFingerprint, 300),
    "morgan": (CircularFingerprint, 2048)
}

def featurize_df(df, smi_col, des_type, fp_types):
    if des_type:
        des_featurizer = descriptors[des_type.lower()]()
        des_names = des_featurizer.descriptors
        X = des_featurizer.featurize(df[smi_col])
        df[des_names] = X
    if type(fp_types) != type([]):
        fp_types = [fp_types]
    for fp_type in fp_types:
        fp_featurizer = fingerprints[fp_type.lower()][0]()
        fp_dim = fingerprints[fp_type.lower()][1]
        fp_names = [f"{fp_type}_{i}" for i in range(fp_dim)]
        X = fp_featurizer.featurize(df[smi_col])
        df[fp_names] = X

