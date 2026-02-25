"""
This file includes code adapted from:

    densitypeakclustering
    https://github.com/lanbing510/DensityPeakCluster

The original implementation has been modified and integrated into this project.
Please refer to the original repository for authorship, documentation,
and license information.
"""
import dgl
import torch
import pandas as pd
import numpy as np
import wandb

from src.layers.clustering import (
    local_density_energy,
    DPC_custom_CLD,
    remove_bad_tracks_from_cluster,
    remove_labels_of_double_showers,
)
from src.layers.shower_matching import (
    CachedIndexList,
    get_labels_pandora,
    obtain_intersection_matrix,
    obtain_union_matrix,
    obtain_intersection_values,
    match_showers,
)
from src.layers.shower_dataframe import (
    get_correction_per_shower,
    distance_to_true_cluster_of_track,
    distance_to_cluster_track,
    generate_showers_data_frame,
)

# Re-export everything so existing callers (utils_training, Gatr_pf_e_noise, …)
# that do `from src.layers.inference_oc import X` continue to work unchanged.
__all__ = [
    "local_density_energy",
    "DPC_custom_CLD",
    "remove_bad_tracks_from_cluster",
    "remove_labels_of_double_showers",
    "CachedIndexList",
    "get_labels_pandora",
    "obtain_intersection_matrix",
    "obtain_union_matrix",
    "obtain_intersection_values",
    "match_showers",
    "get_correction_per_shower",
    "distance_to_true_cluster_of_track",
    "distance_to_cluster_track",
    "generate_showers_data_frame",
    "log_efficiency",
    "store_at_batch_end",
    "create_and_store_graph_output",
]


def log_efficiency(df, pandora=False, clustering=False):
    mask = ~np.isnan(df["reco_showers_E"])
    eff = np.sum(~np.isnan(df["pred_showers_E"][mask].values)) / len(
        df["pred_showers_E"][mask].values
    )
    if pandora:
        wandb.log({"efficiency validation pandora": eff})
    elif clustering:
        wandb.log({"efficiency validation clustering": eff})
    else:
        wandb.log({"efficiency validation": eff})


def _make_save_path(path_save, local_rank, step, epoch, suffix=""):
    return path_save + str(local_rank) + "_" + str(step) + "_" + str(epoch) + suffix + ".pt"


def store_at_batch_end(
    path_save,
    df_batch1,
    df_batch_pandora,
    local_rank=0,
    step=0,
    epoch=None,
    predict=False,
    store=False,
    pandora_available=False,
):
    path_save_ = _make_save_path(path_save, local_rank, step, epoch)
    if store and predict:
        df_batch1.to_pickle(path_save_)
    if predict and pandora_available:
        path_save_pandora = _make_save_path(path_save, local_rank, step, epoch, "_pandora")
        if store and predict:
            df_batch_pandora.to_pickle(path_save_pandora)
    log_efficiency(df_batch1)
    if predict and pandora_available:
        log_efficiency(df_batch_pandora, pandora=True)


def create_and_store_graph_output(
    batch_g,
    model_output,
    y,
    local_rank,
    step,
    epoch,
    path_save,
    store=False,
    predict=False,
    e_corr=None,
    ec_x=None,
    store_epoch=False,
    total_number_events=0,
    pred_pos=None,
    pred_ref_pt=None,
    use_gt_clusters=False,
    pred_pid=None,
    number_of_fakes=None,
    extra_features=None,
    fakes_labels=None,
    pandora_available=False,
    truth_tracks=False,
):
    number_of_showers_total = 0
    number_of_showers_total1 = 0
    number_of_fake_showers_total1 = 0
    batch_g.ndata["coords"] = model_output[:, 0:3]
    batch_g.ndata["beta"] = model_output[:, 3]
    if e_corr is None:
        batch_g.ndata["correction"] = model_output[:, 4]
    graphs = dgl.unbatch(batch_g)
    batch_id = y.batch_number.view(-1)
    df_list1 = []
    df_list_pandora = []
    for i in range(0, len(graphs)):
        mask = batch_id == i
        dic = {}
        dic["graph"] = graphs[i]
        y1 = y.copy()
        y1.mask(mask)
        dic["part_true"] = y1
        X = dic["graph"].ndata["coords"]
        labels_clusters_removed_tracks = torch.zeros(
            dic["graph"].num_nodes(), device=model_output.device
        )
        if use_gt_clusters:
            labels_hdb = dic["graph"].ndata["particle_number"].type(torch.int64)
        else:
            labels_hdb = DPC_custom_CLD(X, dic["graph"], model_output.device)
            if not truth_tracks:
                labels_hdb, labels_clusters_removed_tracks = remove_bad_tracks_from_cluster(
                    dic["graph"], labels_hdb
                )
        if predict and pandora_available:
            labels_pandora = get_labels_pandora(dic, model_output.device)
        particle_ids = torch.unique(dic["graph"].ndata["particle_number"])

        shower_p_unique_hdb, row_ind_hdb, col_ind_hdb, i_m_w_hdb, iou_m = match_showers(
            labels_hdb,
            dic,
            particle_ids,
            model_output,
            local_rank,
            i,
            path_save,
            hdbscan=True,
        )
        if predict and pandora_available:
            (
                shower_p_unique_pandora,
                row_ind_pandora,
                col_ind_pandora,
                i_m_w_pandora,
                iou_m_pandora,
            ) = match_showers(
                labels_pandora,
                dic,
                particle_ids,
                model_output,
                local_rank,
                i,
                path_save,
                pandora=True,
            )

        if len(shower_p_unique_hdb) > 1:
            df_event1, number_of_showers_total1, number_of_fake_showers_total1 = generate_showers_data_frame(
                labels_hdb,
                dic,
                shower_p_unique_hdb,
                particle_ids,
                row_ind_hdb,
                col_ind_hdb,
                i_m_w_hdb,
                e_corr=e_corr,
                number_of_showers_total=number_of_showers_total1,
                step=step,
                number_in_batch=total_number_events,
                ec_x=ec_x,
                pred_pos=pred_pos,
                pred_ref_pt=pred_ref_pt,
                pred_pid=pred_pid,
                number_of_fakes=number_of_fakes,
                number_of_fake_showers_total=number_of_fake_showers_total1,
                extra_features=extra_features,
                labels_clusters_removed_tracks=labels_clusters_removed_tracks,
            )
            if len(df_event1) > 1:
                df_list1.append(df_event1)
            if predict and pandora_available:
                df_event_pandora = generate_showers_data_frame(
                    labels_pandora,
                    dic,
                    shower_p_unique_pandora,
                    particle_ids,
                    row_ind_pandora,
                    col_ind_pandora,
                    i_m_w_pandora,
                    pandora=True,
                    step=step,
                    number_in_batch=total_number_events,
                )
                if df_event_pandora is not None and type(df_event_pandora) is not tuple:
                    df_list_pandora.append(df_event_pandora)
                else:
                    print("Not appending to df_list_pandora")
            total_number_events = total_number_events + 1

    df_batch1 = pd.concat(df_list1)
    if predict and pandora_available:
        df_batch_pandora = pd.concat(df_list_pandora)
    else:
        df_batch = []
        df_batch_pandora = []
    if store:
        store_at_batch_end(
            path_save,
            df_batch1,
            df_batch_pandora,
            local_rank,
            step,
            epoch,
            predict=predict,
            store=store_epoch,
            pandora_available=pandora_available,
        )
    if predict:
        return df_batch_pandora, df_batch1, total_number_events
    else:
        return df_batch1
