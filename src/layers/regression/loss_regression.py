
import torch
import numpy as np



def obtain_PID_charged(dic,pid_true_matched, pids_charged, args, pid_conversion_dict):
    charged_PID_pred = dic["charged_PID_pred"]
    charged_PID_true = np.array(pid_true_matched)[dic["charged_idx"].cpu().tolist()]
    # one-hot encoded
    charged_PID_true_onehot = torch.zeros(
        len(charged_PID_true), len(pids_charged)
    ).to(charged_PID_pred.device)
    mask_charged = torch.ones(len(charged_PID_true))
    pids_charged_arr = np.array(pids_charged)
    for i, pid in enumerate(charged_PID_true):
        if pid not in pid_conversion_dict:
            print("Unknown PID", pid)
        true_idx = pid_conversion_dict.get(pid, 3)
        col = np.where(pids_charged_arr == true_idx)[0]
        if len(col) == 0:
            mask_charged[i] = 0
        else:
            charged_PID_true_onehot[i, col[0]] = 1
    return charged_PID_pred, charged_PID_true_onehot, mask_charged






def obtain_PID_neutral(dic,pid_true_matched,pids_neutral, args, pid_conversion_dict):
    neutral_PID_pred = dic["neutral_PID_pred"]
    neutral_idx = dic["neutrals_idx"]
    neutral_PID_true = np.array(pid_true_matched)[neutral_idx.cpu()]
    if type(neutral_PID_true) == np.float64:
        neutral_PID_true = [neutral_PID_true]
    # One-hot encoded
    neutral_PID_true_onehot = torch.zeros(
        len(neutral_PID_true), len(pids_neutral)
    ).to(neutral_PID_pred.device)
    mask_neutral = torch.ones(len(neutral_PID_true))

    # convert from true PID to int list PID (4-class encoding)
    pids_neutral_arr = np.array(pids_neutral)
    for i, pid in enumerate(neutral_PID_true):
        if pid not in pid_conversion_dict:
            print("Unknown PID", pid)
        true_idx = pid_conversion_dict.get(pid, 3)
        col = np.where(pids_neutral_arr == true_idx)[0]
        if len(col) == 0:
            mask_neutral[i] = 0
        else:
            neutral_PID_true_onehot[i, col[0]] = 1
    neutral_PID_true_onehot = neutral_PID_true_onehot.to(neutral_idx.device)
    return neutral_PID_pred, neutral_PID_true_onehot, mask_neutral
    


