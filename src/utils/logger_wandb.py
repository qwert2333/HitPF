import wandb
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score
import json
import dgl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch_scatter import scatter_max
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def log_losses_wandb(
    logwandb, num_batches, local_rank, losses, loss, val=False
):
    if val:
        val_ = " val"
    else:
        val_ = ""
    if logwandb and ((num_batches - 1) % 10) == 0 and local_rank == 0:
        wandb.log(
            {
                "loss" + val_ + " regression": loss,
                "loss" + val_ + " lv": losses[0],
                "loss" + val_ + " beta": losses[1],
                "loss" + val_ + " beta sig": losses[2],
                "loss" + val_ + " beta noise": losses[3],
                "loss" + val_ + " attractive": losses[12],
                "loss" + val_ + " repulsive": losses[13],
            }
        ) 

