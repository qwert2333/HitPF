import gzip
import pickle
import mplhep as hep
from src.utils.pid_conversion import pid_conversion_dict

#hep.style.use("CMS")
import matplotlib

import numpy as np
import pandas as pd



def open_mlpf_dataframe(path_mlpf, neutrals_only=False, charged_only=False):
    data = pd.read_pickle(path_mlpf)
    sd = data
    sd["pid_4_class_true"] = sd["pid"].map(pid_conversion_dict)
    if "pred_pid_matched" in sd.columns:
        sd.loc[sd["pred_pid_matched"] < -1, "pred_pid_matched"] = np.nan
    return sd

def concat_with_batch_fix(dfs, batch_key="number_batch"):

    corrected_dfs = []
    batch_offset = 0

    for df in dfs:
        df = df.copy()
        if batch_key in df.columns:
            df[batch_key] = df[batch_key] + batch_offset
            batch_offset = df[batch_key].max() + 1
        else:
            raise KeyError(f"'{batch_key}' not found in one of the DataFrames.")
        corrected_dfs.append(df)
    return pd.concat(corrected_dfs, ignore_index=True)

