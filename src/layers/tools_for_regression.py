import torch
import numpy as np
from torch_scatter import scatter_mean, scatter_sum

def pick_lowest_chi_squared(pxpypz, chi_s, batch_idx, xyz_nodes):
    unique_batch = torch.unique(batch_idx)
    p_direction = []
    track_xyz = []
    for i in range(0, len(unique_batch)):
        mask = batch_idx == unique_batch[i]
        if torch.sum(mask) > 1:
            chis = chi_s[mask]
            ind_min = torch.argmin(chis)
            p_direction.append(pxpypz[mask][ind_min].view(-1, 3))
            track_xyz.append(xyz_nodes[mask][ind_min].view(-1, 3))

        else:
            p_direction.append(pxpypz[mask].view(-1, 3))
            track_xyz.append(xyz_nodes[mask].view(-1, 3))
    return torch.concat(p_direction, dim=0), torch.stack(track_xyz)[:, 0]



class AverageHitsP(torch.nn.Module):
    # Same layout of the module as the GNN one, but just computes the average of the hits. Try to compare this + ML clustering with Pandora
    def __init__(self, ecal_only=False):
        super(AverageHitsP, self).__init__()
        self.ecal_only = ecal_only
    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        assert graphs_new is not None
        batch_num_nodes = graphs_new.batch_num_nodes()  # Num. of hits in each graph
        batch_idx = []
        batch_bounds = []
        if self.ecal_only:
            mask_ecal_only = [] # whether to consider only ECAL or ECAL+HCAL
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
            batch_bounds.append(n)
        batch_idx = np.array(batch_idx)
        for i in range(len(np.unique(batch_idx))):
            if self.ecal_only:
                n_ecal_hits = (graphs_new.ndata["h"][batch_idx == i, 5] > 0).sum()
                n_hcal_hits = (graphs_new.ndata["h"][batch_idx == i, 6] > 0).sum()
                for _ in range(batch_num_nodes[i]):
                    mask_ecal_only.append((n_ecal_hits / (n_hcal_hits + n_ecal_hits)).item())
        batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
        if self.ecal_only:
            mask_ecal_only = torch.tensor(mask_ecal_only)  # round().int().bool().to(graphs_new.device)
            mask_ecal_only = (mask_ecal_only > 0.05).int().bool().to(graphs_new.device)
            #mask_ecal_only=torch.zeros(len(mask_ecal_only)).bool().to(graphs_new.device)
        xyz_hits = graphs_new.ndata["h"][:, :3]
        E_hits = graphs_new.ndata["h"][:, 8]
        if self.ecal_only:
            hcal_hits = graphs_new.ndata["h"][:, 6] > 0
            E_hits[mask_ecal_only & (hcal_hits)] = 0
        weighted_avg_hits = scatter_sum(xyz_hits * E_hits.unsqueeze(1), batch_idx, dim=0)
        E_total = scatter_sum(E_hits, batch_idx, dim=0)
        p_direction = weighted_avg_hits / E_total.unsqueeze(1)
        p_tracks = torch.norm(p_direction, dim=1)
        p_direction = p_direction / torch.norm(p_direction, dim=1).unsqueeze(1)
        # if self.pos_regression:
        return p_tracks, p_direction,  weighted_avg_hits / E_total.unsqueeze(1) * 3300 # Reference point
        # return p_tracks



class PickPAtDCA(torch.nn.Module):
    # Same layout of the module as the GNN one, but just picks the track
    def __init__(self):
        super(PickPAtDCA, self).__init__()

    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        assert graphs_new is not None
        batch_num_nodes = graphs_new.batch_num_nodes()  
        batch_idx = []
        batch_bounds = []
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
            batch_bounds.append(n)
        batch_idx = torch.tensor(batch_idx).to(graphs_new.device)

        ht = graphs_new.ndata["h"][:, 3:7].argmax(dim=1)
        filt = ht == 1  # track
        filt_hits = ((ht == 2) + (ht == 3)).bool()

        p_direction, p_xyz = pick_lowest_chi_squared(
            graphs_new.ndata["pos_pxpypz_at_vertex"][filt],
            graphs_new.ndata["chi_squared_tracks"][filt],
            batch_idx[filt],
            graphs_new.ndata["h"][filt, :3]
        )
        # Barycenters of clusters of hits
        xyz_hits = graphs_new.ndata["h"][:, :3]
        E_hits = graphs_new.ndata["h"][:, 8]
        weighted_avg_hits = scatter_sum(xyz_hits * E_hits.unsqueeze(1), batch_idx, dim=0)
        E_total = scatter_sum(E_hits, batch_idx, dim=0)
        barycenters = weighted_avg_hits / E_total.unsqueeze(1)
        p_tracks = torch.norm(p_direction, dim=1)
        return p_tracks, p_direction, barycenters - p_xyz



class ECNetWrapperAvg(torch.nn.Module):
    # use the GNN+NN model for energy correction
    # This one concatenates GNN features to the global features
    def __init__(self):
        super(ECNetWrapperAvg, self).__init__()
        self.AvgHits = AverageHitsP(ecal_only=True)

    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        _, p_pred, _ = self.AvgHits.predict(x_global_features, graphs_new)
        p_pred = (p_pred / torch.norm(p_pred, dim=1).unsqueeze(1)).clone()
        return None, p_pred, None, None
