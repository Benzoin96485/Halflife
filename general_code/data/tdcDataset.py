from tdc.single_pred import ADME
from general_code.data.prepSMILES import wash_df, standardize_df
from general_code.data.mergeData import merge_hetero

tdc_data_path = "data/tdc"
NAMES = {
    "regression": [
        "Caco2_Wang",
        "Lipophilicity_AstraZeneca",
        "Solubility_AqSolDB",
        "HydrationFreeEnergy_FreeSolv",
        "PPBR_AZ",
        "VDss_Lombardo",
        "Half_Life_Obach",
        "Clearance_Hepatocyte_AZ",
        "Clearance_Microsome_AZ"
    ],
    "classification": [
        'HIA_Hou',
        'Pgp_Broccatelli',
        'Bioavailability_Ma',
        'BBB_Martins',
        'CYP2C19_Veith',
        'CYP2D6_Veith',
        'CYP3A4_Veith',
        'CYP1A2_Veith',
        'CYP2C9_Veith',
        'CYP2C9_Substrate_CarbonMangels',
        'CYP2D6_Substrate_CarbonMangels',
        'CYP3A4_Substrate_CarbonMangels',
    ]
}

def make_multitask_data(task_name_list, check_name, **kwargs):
    df_list = [
        standardize_df(
            wash_df(
                ADME(name=name, path=tdc_data_path).get_data(),
                smi_col="Drug",
                check_name=check_name,
                **kwargs
            ),
            smi_col="Drug"
        )
        for name in task_name_list
    ]
    
    return merge_hetero(
        df_list=df_list,
        label_name_list=task_name_list,
        label_col="Y",
        old_smi_col="Drug",
        new_smi_col="smiles"
    )

if __name__ == "__main__":
    for name in NAMES["regression"]:
        data = ADME(name=name, path=tdc_data_path)
    for name in NAMES["classification"]:
        data = ADME(name=name, path=tdc_data_path)

