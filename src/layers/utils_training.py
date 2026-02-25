
from lightning.pytorch.callbacks import BaseFinetuning
import torch
import dgl
from src.layers.inference_oc import DPC_custom_CLD
from src.layers.inference_oc import match_showers
from src.layers.inference_oc import remove_bad_tracks_from_cluster
class FreezeClustering(BaseFinetuning):
    def __init__(
        self,
    ):
        super().__init__()
        
    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.ScaledGooeyBatchNorm2_1)
        self.freeze(pl_module.gatr)
        self.freeze(pl_module.clustering)
        self.freeze(pl_module.beta)

        print("CLUSTERING HAS BEEN FROOOZEN")

    def finetune_function(self, pl_module, current_epoch, optimizer):
        print("Not finetunning")
       


def obtain_batch_numbers(x, g):
    dev = x.device
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes).to(dev))
        # num_nodes = gj.number_of_nodes()

    batch = torch.cat(batch_numbers, dim=0)
    return batch



def obtain_clustering_for_matched_showers(
    batch_g, model_output, y_all, local_rank, use_gt_clusters=False, add_fakes=True
):

    graphs_showers_matched = []
    graphs_showers_fakes = []
    true_energy_showers = []
    reco_energy_showers = []
    reco_energy_showers_fakes = []
    energy_true_daughters = []
    y_pids_matched = []
    y_coords_matched = []
    if not  use_gt_clusters:
        batch_g.ndata["coords"] = model_output[:, 0:3]
        batch_g.ndata["beta"] = model_output[:, 3]
    graphs = dgl.unbatch(batch_g)
    batch_id = y_all.batch_number
    for i in range(0, len(graphs)):
        mask = batch_id == i
        dic = {}
        dic["graph"] = graphs[i]
        y = y_all.copy()
       
        y.mask(mask.flatten())
        dic["part_true"] = y
        if not  use_gt_clusters:
            betas = torch.sigmoid(dic["graph"].ndata["beta"])
            X = dic["graph"].ndata["coords"]
  
        if use_gt_clusters:
            labels = dic["graph"].ndata["particle_number"].type(torch.int64)
        else:
            labels =DPC_custom_CLD(X, dic["graph"], model_output.device)
            
            labels, _ = remove_bad_tracks_from_cluster(dic["graph"], labels)
        particle_ids = torch.unique(dic["graph"].ndata["particle_number"])
        shower_p_unique = torch.unique(labels)
        shower_p_unique, row_ind, col_ind, i_m_w, _ = match_showers(
            labels, dic, particle_ids, model_output, local_rank, i, None
        )
        row_ind = torch.Tensor(row_ind).to(model_output.device).long()
        col_ind = torch.Tensor(col_ind).to(model_output.device).long()
        if torch.sum(particle_ids == 0) > 0:
            row_ind_ = row_ind - 1
        else:
            # if there is no zero then index 0 corresponds to particle 1.
            row_ind_ = row_ind
        index_matches = col_ind + 1
        index_matches = index_matches.to(model_output.device).long()
        
        for j, unique_showers_label in enumerate(index_matches):
            if torch.sum(unique_showers_label == index_matches) == 1:
                index_in_matched = torch.argmax(
                    (unique_showers_label == index_matches) * 1
                )
                mask = labels == unique_showers_label
                sls_graph = graphs[i].ndata["pos_hits_xyz"][mask][:, 0:3]
                g = dgl.graph(([], []))
                g.add_nodes(sls_graph.shape[0])
                g =  g.to(sls_graph.device)
                g.ndata["h"] = graphs[i].ndata["h"][mask]
                if "pos_pxpypz" in graphs[i].ndata:
                    g.ndata["pos_pxpypz"] = graphs[i].ndata["pos_pxpypz"][mask]
                if "pos_pxpypz_at_vertex" in graphs[i].ndata:
                    g.ndata["pos_pxpypz_at_vertex"] = graphs[i].ndata[
                        "pos_pxpypz_at_vertex"
                    ][mask]
                g.ndata["chi_squared_tracks"] = graphs[i].ndata["chi_squared_tracks"][mask]
                energy_t = dic["part_true"].E.to(model_output.device)
                energy_t_corr_daughters = dic["part_true"].m.to(
                    model_output.device
                )
                true_energy_shower = energy_t[row_ind_[j]]
                y_pids_matched.append(y.pid[row_ind_[j]].item())
                y_coords_matched.append(y.coord[row_ind_[j]].detach().cpu().numpy())
                energy_true_daughters.append(energy_t_corr_daughters[row_ind_[j]])
                reco_energy_shower = torch.sum(graphs[i].ndata["e_hits"][mask])
                graphs_showers_matched.append(g)
                true_energy_showers.append(true_energy_shower.view(-1))
                reco_energy_showers.append(reco_energy_shower.view(-1))
        pred_showers = shower_p_unique
        pred_showers[index_matches] = -1
        pred_showers[
            0
        ] = (
            -1
        )
        mask_fakes = pred_showers != -1
        fakes_idx = torch.where(mask_fakes)[0]
        if add_fakes:
            for j in fakes_idx:
                mask = labels == j
                sls_graph = graphs[i].ndata["pos_hits_xyz"][mask][:, 0:3]
                g = dgl.graph(([], []))
                g.add_nodes(sls_graph.shape[0])
                g =  g.to(sls_graph.device)
                
                g.ndata["h"] = graphs[i].ndata["h"][mask]
                   
                if "pos_pxpypz" in graphs[i].ndata:
                    g.ndata["pos_pxpypz"] = graphs[i].ndata["pos_pxpypz"][mask]
                if "pos_pxpypz_at_vertex" in graphs[i].ndata:
                    g.ndata["pos_pxpypz_at_vertex"] = graphs[i].ndata[
                        "pos_pxpypz_at_vertex"
                    ][mask]
                g.ndata["chi_squared_tracks"] = graphs[i].ndata["chi_squared_tracks"][mask]
                graphs_showers_fakes.append(g)
                reco_energy_shower = torch.sum(graphs[i].ndata["e_hits"][mask])
                reco_energy_showers_fakes.append(reco_energy_shower.view(-1))
    graphs_showers_matched = dgl.batch(graphs_showers_matched + graphs_showers_fakes)
    true_energy_showers = torch.cat(true_energy_showers, dim=0)
    reco_energy_showers = torch.cat(reco_energy_showers + reco_energy_showers_fakes, dim=0)
    e_true_corr_daughters = torch.cat(energy_true_daughters, dim=0)
    number_of_fakes = len(reco_energy_showers_fakes)
    return (
        graphs_showers_matched,
        true_energy_showers,
        reco_energy_showers,
        y_pids_matched,
        e_true_corr_daughters,
        y_coords_matched,
        number_of_fakes,
        fakes_idx
    )
