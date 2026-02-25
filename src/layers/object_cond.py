"""
The loss implementation in this file is adapted from the HGCalML repository:

    Repository: https://github.com/jkiesele/HGCalML
    File: modules/lossLayers.py

Original author: Jan Kieseler
License: See the original repository for license details.

The implementation has been modified and integrated into this project.
"""

from typing import Tuple, Union
import numpy as np
import torch
from torch_scatter import scatter_max, scatter_add, scatter_mean
import dgl

def safe_index(arr, index):
    # One-hot index (or zero if it's not in the array)
    if index not in arr:
        return 0
    else:
        return arr.index(index) + 1


def assert_no_nans(x):
    """
    Raises AssertionError if there is a nan in the tensor
    """
    if torch.isnan(x).any():
        print(x)
    assert not torch.isnan(x).any()


def calc_LV_Lbeta(
    original_coords,
    g,
    y,
    distance_threshold,
    energy_correction,
    beta: torch.Tensor,
    cluster_space_coords: torch.Tensor,  # Predicted by model
    cluster_index_per_event: torch.Tensor,  # Truth hit->cluster index
    batch: torch.Tensor,
    predicted_pid=None,  # predicted PID embeddings - will be aggregated by summing up the clusters and applying the post_pid_pool_module MLP afterwards
    # From here on just parameters
    qmin: float = 0.1,
    s_B: float = 1.0,
    noise_cluster_index: int = 0,  # cluster_index entries with this value are noise/noise
    frac_combinations=0,  # fraction of the all possible pairs to be used for the clustering loss
    use_average_cc_pos=0.0,
    loss_type="hgcalimplementation",
) -> Union[Tuple[torch.Tensor, torch.Tensor], dict]:
    """
    Calculates the L_V and L_beta object condensation losses.
    Concepts:
    - A hit belongs to exactly one cluster (cluster_index_per_event is (n_hits,)),
      and to exactly one event (batch is (n_hits,))
    - A cluster index of `noise_cluster_index` means the cluster is a noise cluster.
      There is typically one noise cluster per event. Any hit in a noise cluster
      is a 'noise hit'. A hit in an object is called a 'signal hit' for lack of a
      better term.
    - An 'object' is a cluster that is *not* a noise cluster.
    beta_stabilizing: Choices are ['paper', 'clip', 'soft_q_scaling']:
        paper: beta is sigmoid(model_output), q = beta.arctanh()**2 + qmin
        clip:  beta is clipped to 1-1e-4, q = beta.arctanh()**2 + qmin
        soft_q_scaling: beta is sigmoid(model_output), q = (clip(beta)/1.002).arctanh()**2 + qmin
    huberize_norm_for_V_attractive: Huberizes the norms when used in the attractive potential
    beta_term_option: Choices are ['paper', 'short-range-potential']:
        Choosing 'short-range-potential' introduces a short range potential around high
        beta points, acting like V_attractive.
    Note this function has modifications w.r.t. the implementation in 2002.03605:
    - The norms for V_repulsive are now Gaussian (instead of linear hinge)
    """
    # remove dummy rows added for dataloader #TODO think of better way to do this
    device = beta.device
    if torch.isnan(beta).any():
        print("There are nans in beta! L198", len(beta[torch.isnan(beta)]))

    beta = torch.nan_to_num(beta, nan=0.0)
    assert_no_nans(beta)
    # ________________________________

    # Calculate a bunch of needed counts and indices locally

    # cluster_index: unique index over events
    # E.g. cluster_index_per_event=[ 0, 0, 1, 2, 0, 0, 1], batch=[0, 0, 0, 0, 1, 1, 1]
    #      -> cluster_index=[ 0, 0, 1, 2, 3, 3, 4 ]
    cluster_index, n_clusters_per_event = batch_cluster_indices(
        cluster_index_per_event, batch
    )
    n_clusters = n_clusters_per_event.sum()
    n_hits, cluster_space_dim = cluster_space_coords.size()
    batch_size = batch.max() + 1
    n_hits_per_event = scatter_count(batch)

    # Index of cluster -> event (n_clusters,)
    batch_cluster = scatter_counts_to_indices(n_clusters_per_event)

    # Per-hit boolean, indicating whether hit is sig or noise
    is_noise = cluster_index_per_event == noise_cluster_index
    is_sig = ~is_noise
    n_hits_sig = is_sig.sum()
    n_sig_hits_per_event = scatter_count(batch[is_sig])

    # Per-cluster boolean, indicating whether cluster is an object or noise
    is_object = scatter_max(is_sig.long(), cluster_index)[0].bool()
    is_noise_cluster = ~is_object


    if noise_cluster_index != 0:
        raise NotImplementedError
    object_index_per_event = cluster_index_per_event[is_sig] - 1
    object_index, n_objects_per_event = batch_cluster_indices(
        object_index_per_event, batch[is_sig]
    )
    n_hits_per_object = scatter_count(object_index)
    # print("n_hits_per_object", n_hits_per_object)
    batch_object = batch_cluster[is_object]
    n_objects = is_object.sum()
 
    assert object_index.size() == (n_hits_sig,)
    assert is_object.size() == (n_clusters,)
    assert torch.all(n_hits_per_object > 0)
    assert object_index.max() + 1 == n_objects

    # ________________________________
    # L_V term

    # Calculate q
    q = (beta.clip(0.0, 1 - 1e-4).arctanh() / 1.01) ** 2 + qmin
    assert_no_nans(q)
    assert q.device == device
    assert q.size() == (n_hits,)

    # Calculate q_alpha, the max q per object, and the indices of said maxima
    # assert hit_energies.shape == q.shape
    # q_alpha, index_alpha = scatter_max(hit_energies[is_sig], object_index)
    q_alpha, index_alpha = scatter_max(q[is_sig], object_index)
    assert q_alpha.size() == (n_objects,)

    # Get the cluster space coordinates and betas for these maxima hits too
    x_alpha = cluster_space_coords[is_sig][index_alpha]
    x_alpha_original = original_coords[is_sig][index_alpha]
    if use_average_cc_pos > 0:
        x_alpha_sum = scatter_add(
            q[is_sig].view(-1, 1).repeat(1, 3) * cluster_space_coords[is_sig],
            object_index,
            dim=0,
        )  # * beta[is_sig].view(-1, 1).repeat(1, 3)
        qbeta_alpha_sum = scatter_add(q[is_sig], object_index) + 1e-9  # * beta[is_sig]
        div_fac = 1 / qbeta_alpha_sum
        div_fac = torch.nan_to_num(div_fac, nan=0)
        x_alpha_mean = torch.mul(x_alpha_sum, div_fac.view(-1, 1).repeat(1, 3))
        x_alpha = use_average_cc_pos * x_alpha_mean + (1 - use_average_cc_pos) * x_alpha
    
    beta_alpha = beta[is_sig][index_alpha]
    assert x_alpha.size() == (n_objects, cluster_space_dim)
    assert beta_alpha.size() == (n_objects,)


    # Connectivity matrix from hit (row) -> cluster (column)
    # Index to matrix, e.g.:
    # [1, 3, 1, 0] --> [
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1],
    #     [0, 1, 0, 0],
    #     [1, 0, 0, 0]
    #     ]
    M = torch.nn.functional.one_hot(cluster_index).long()

    # Anti-connectivity matrix; be sure not to connect hits to clusters in different events!
    M_inv = get_inter_event_norms_mask(batch, n_clusters_per_event) - M

    # Throw away noise cluster columns; we never need them
    M = M[:, is_object]
    M_inv = M_inv[:, is_object]
    assert M.size() == (n_hits, n_objects)
    assert M_inv.size() == (n_hits, n_objects)

    # Calculate all norms
    # Warning: Should not be used without a mask!
    # Contains norms between hits and objects from different events
    # (n_hits, 1, cluster_space_dim) - (1, n_objects, cluster_space_dim)
    #   gives (n_hits, n_objects, cluster_space_dim)
    norms = (cluster_space_coords.unsqueeze(1) - x_alpha.unsqueeze(0)).norm(dim=-1)
    assert norms.size() == (n_hits, n_objects)
    L_clusters = torch.tensor(0.0).to(device)
    if frac_combinations != 0:
        L_clusters = L_clusters_calc(
            batch, cluster_space_coords, cluster_index, frac_combinations, q
        )

    # -------
    # Attractive potential term
    # First get all the relevant norms: We only want norms of signal hits
    # w.r.t. the object they belong to, i.e. no noise hits and no noise clusters.
    # First select all norms of all signal hits w.r.t. all objects, mask out later
    
    N_k = torch.sum(M, dim=0)  # number of hits per object
    norms = torch.sum(
        torch.square(cluster_space_coords.unsqueeze(1) - x_alpha.unsqueeze(0)),
        dim=-1,
    ) # take the norm squared
    norms_att = norms[is_sig]
    #att func as in line 159 of object condensation
    
    norms_att = torch.log(
        torch.exp(torch.Tensor([1]).to(norms_att.device)) * norms_att / 2 + 1
    )

    assert norms_att.size() == (n_hits_sig, n_objects)

    # Now apply the mask to keep only norms of signal hits w.r.t. to the object
    # they belong to
    norms_att *= M[is_sig]

    # Sum over hits, then sum per event, then divide by n_hits_per_event, then sum over events

    V_attractive = (q[is_sig]).unsqueeze(-1) * q_alpha.unsqueeze(0) * norms_att
    V_attractive = V_attractive.sum(dim=0)  # K objects
    V_attractive = V_attractive.view(-1) / (N_k.view(-1) + 1e-3)
    L_V_attractive = torch.mean(V_attractive)

  
    norms_rep = torch.relu(1. - torch.sqrt(norms + 1e-6))* M_inv
    

    # (n_sig_hits, 1) * (1, n_objects) * (n_sig_hits, n_objects)
    V_repulsive = q.unsqueeze(1) * q_alpha.unsqueeze(0) * norms_rep

    # No need to apply a V = max(0, V); by construction V>=0
    assert V_repulsive.size() == (n_hits, n_objects)

    # Sum over hits, then sum per event, then divide by n_hits_per_event, then sum up events
    nope = n_objects_per_event - 1
    nope[nope == 0] = 1
    
    L_V_repulsive = V_repulsive.sum(dim=0)
    number_of_repulsive_terms_per_object = torch.sum(M_inv, dim=0)
    L_V_repulsive = L_V_repulsive.view(
        -1
    ) / number_of_repulsive_terms_per_object.view(-1)
    L_V_repulsive = torch.mean(L_V_repulsive)
    L_V_repulsive2 = L_V_repulsive
   
    L_V = (
        L_V_attractive 
        + L_V_repulsive   
        
    )
  
            
 

    n_noise_hits_per_event = scatter_count(batch[is_noise])
    n_noise_hits_per_event[n_noise_hits_per_event == 0] = 1
    L_beta_noise = (
        s_B
        * (
            (scatter_add(beta[is_noise], batch[is_noise])) / n_noise_hits_per_event
        ).sum()
    )

    # L_beta signal term

    beta_per_object_c = scatter_add(beta[is_sig], object_index)
    beta_alpha = beta[is_sig][index_alpha]
    # hit_type_mask = (g.ndata["hit_type"]==1)*(g.ndata["particle_number"]>0)
    # beta_alpha_track = beta[is_sig*hit_type_mask]
    L_beta_sig = torch.mean(
        1 - beta_alpha + 1 - torch.clip(beta_per_object_c, 0, 1)
    )

    L_beta_noise = L_beta_noise / 4


    L_beta = L_beta_noise + L_beta_sig 

    L_alpha_coordinates = torch.mean(torch.norm(x_alpha_original - x_alpha, p=2, dim=1))


    L_exp = L_beta
    if (loss_type == "hgcalimplementation") or (loss_type == "vrepweighted") or (loss_type == "baseline"):
        return (
            L_V,  
            L_beta,
            L_beta_sig,
            L_beta_noise,  
            0, 
            0,  
            0,  
            None,  
            None,  
            0,  
            L_clusters,  
            0,  
            L_V_attractive,
            L_V_repulsive,
            L_alpha_coordinates,
            L_exp,
            norms_rep, 
            norms_att,  
            L_V_repulsive2,
            0
        )


def object_condensation_loss2(
    batch,
    pred,
    pred_2,
    y,
    q_min=0.1,
    use_average_cc_pos=0.0,
    output_dim=4,
    clust_space_norm="none",
):
    """

    :param batch:
    :param pred:
    :param y:
    :param return_resolution: If True, it will only output resolution data to plot for regression (only used for evaluation...)
    :param clust_loss_only: If True, it will only add the clustering terms to the loss
    :return:
    """
    _, S = pred.shape
    
    clust_space_dim = 3
   

    bj = torch.sigmoid(torch.reshape(pred[:, clust_space_dim], [-1, 1]))  # 3: betas
    # print("bj", bj)
    original_coords = batch.ndata["h"][:, 0:clust_space_dim]
    distance_threshold = 0
    energy_correction = pred_2
    xj = pred[:, 0:clust_space_dim]  # xj: cluster space coords
    if clust_space_norm == "twonorm":
        xj = torch.nn.functional.normalize(xj, dim=1)  # 0, 1, 2: cluster space coords
    elif clust_space_norm == "tanh":
        xj = torch.tanh(xj)
    elif clust_space_norm == "none":
        pass
    else:
        raise NotImplementedError
  
    dev = batch.device
    clustering_index_l = batch.ndata["particle_number"]

    len_batch = len(batch.batch_num_nodes())
    batch_numbers = torch.repeat_interleave(
        torch.arange(0, len_batch).to(dev), batch.batch_num_nodes()
    ).to(dev)

    a = calc_LV_Lbeta(
        original_coords,
        batch,
        y,
        distance_threshold,
        energy_correction,
        beta=bj.view(-1),
        cluster_space_coords=xj,  # Predicted by model
        cluster_index_per_event=clustering_index_l.view(
            -1
        ).long(),  # Truth hit->cluster index
        batch=batch_numbers.long(),
        qmin=q_min,
        use_average_cc_pos=use_average_cc_pos,
    )

   
    loss = 1 * a[0] + a[1]  
      
    return loss, a

def formatted_loss_components_string(components: dict) -> str:
    """
    Formats the components returned by calc_LV_Lbeta
    """
    total_loss = components["L_V"] + components["L_beta"]
    fractions = {k: v / total_loss for k, v in components.items()}
    fkey = lambda key: f"{components[key]:+.4f} ({100.*fractions[key]:.1f}%)"
    s = (
        "  L_V                 = {L_V}"
        "\n    L_V_attractive      = {L_V_attractive}"
        "\n    L_V_repulsive       = {L_V_repulsive}"
        "\n  L_beta              = {L_beta}"
        "\n    L_beta_noise        = {L_beta_noise}"
        "\n    L_beta_sig          = {L_beta_sig}".format(
            L=total_loss, **{k: fkey(k) for k in components}
        )
    )
    if "L_beta_norms_term" in components:
        s += (
            "\n      L_beta_norms_term   = {L_beta_norms_term}"
            "\n      L_beta_logbeta_term = {L_beta_logbeta_term}".format(
                **{k: fkey(k) for k in components}
            )
        )
    if "L_noise_filter" in components:
        s += f'\n  L_noise_filter = {fkey("L_noise_filter")}'
    return s


def huber(d, delta):
    """
    See: https://en.wikipedia.org/wiki/Huber_loss#Definition
    Multiplied by 2 w.r.t Wikipedia version (aligning with Jan's definition)
    """
    return torch.where(
        torch.abs(d) <= delta, d**2, 2.0 * delta * (torch.abs(d) - delta)
    )


def batch_cluster_indices(
    cluster_id: torch.Tensor, batch: torch.Tensor
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Turns cluster indices per event to an index in the whole batch
    Example:
    cluster_id = torch.LongTensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    -->
    offset = torch.LongTensor([0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 5, 5, 5])
    output = torch.LongTensor([0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6])
    """
    device = cluster_id.device
    assert cluster_id.device == batch.device
    # Count the number of clusters per entry in the batch
    n_clusters_per_event = scatter_max(cluster_id, batch, dim=-1)[0] + 1
    # Offsets are then a cumulative sum
    offset_values_nozero = n_clusters_per_event[:-1].cumsum(dim=-1)
    # Prefix a zero
    offset_values = torch.cat((torch.zeros(1, device=device), offset_values_nozero))
    # Fill it per hit
    offset = torch.gather(offset_values, 0, batch).long()
    return offset + cluster_id, n_clusters_per_event


def get_clustering(betas: torch.Tensor, X: torch.Tensor, tbeta=0.1, td=1.0):
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes torch.Tensors as input.
    """
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = torch.arange(n_points)
    clustering = -1 * torch.ones(n_points, dtype=torch.long).to(betas.device)
    for index_condpoint in indices_condpoints:
        d = torch.norm(X[unassigned] - X[index_condpoint][0], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint[0]
        unassigned = unassigned[~(d < td)]
    return clustering


def scatter_count(input: torch.Tensor):
    """
    Returns ordered counts over an index array
    Example:
    >>> scatter_count(torch.Tensor([0, 0, 0, 1, 1, 2, 2])) # input
    >>> [3, 2, 2]
    Index assumptions work like in torch_scatter, so:
    >>> scatter_count(torch.Tensor([1, 1, 1, 2, 2, 4, 4]))
    >>> tensor([0, 3, 2, 0, 2])
    """
    return scatter_add(torch.ones_like(input, dtype=torch.long), input.long())


def scatter_counts_to_indices(input: torch.LongTensor) -> torch.LongTensor:
    """
    Converts counts to indices. This is the inverse operation of scatter_count
    Example:
    input:  [3, 2, 2]
    output: [0, 0, 0, 1, 1, 2, 2]
    """
    return torch.repeat_interleave(
        torch.arange(input.size(0), device=input.device), input
    ).long()


def get_inter_event_norms_mask(
    batch: torch.LongTensor, nclusters_per_event: torch.LongTensor
):
    """
    Creates mask of (nhits x nclusters) that is only 1 if hit i is in the same event as cluster j
    Example:
    cluster_id_per_event = torch.LongTensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    Should return:
    torch.LongTensor([
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        ])
    """
    device = batch.device
    # Following the example:
    # Expand batch to the following (nhits x nevents) matrix (little hacky, boolean mask -> long):
    # [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
    batch_expanded_as_ones = (
        batch
        == torch.arange(batch.max() + 1, dtype=torch.long, device=device).unsqueeze(-1)
    ).long()
    # Then repeat_interleave it to expand it to nclusters rows, and transpose to get (nhits x nclusters)
    return batch_expanded_as_ones.repeat_interleave(nclusters_per_event, dim=0).T


def isin(ar1, ar2):
    """To be replaced by torch.isin for newer releases of torch"""
    return (ar1[..., None] == ar2).any(-1)


def L_clusters_calc(batch, cluster_space_coords, cluster_index, frac_combinations, q):
    number_of_pairs = 0
    for batch_id in batch.unique():
        # do all possible pairs...
        bmask = batch == batch_id
        clust_space_filt = cluster_space_coords[bmask]
        pos_pairs_all = []
        neg_pairs_all = []
        if len(cluster_index[bmask].unique()) <= 1:
            continue
        L_clusters = torch.tensor(0.0).to(q.device)
        for cluster in cluster_index[bmask].unique():
            coords_pos = clust_space_filt[cluster_index[bmask] == cluster]
            coords_neg = clust_space_filt[cluster_index[bmask] != cluster]
            if len(coords_neg) == 0:
                continue
            clust_idx = cluster_index[bmask] == cluster
            # all_ones = torch.ones_like((clust_idx, clust_idx))
            # pos_pairs = [[i, j] for i in range(len(coords_pos)) for j in range (len(coords_pos)) if i < j]
            total_num = (len(coords_pos) ** 2) / 2
            num = int(frac_combinations * total_num)
            pos_pairs = []
            for i in range(num):
                pos_pairs.append(
                    [
                        np.random.randint(len(coords_pos)),
                        np.random.randint(len(coords_pos)),
                    ]
                )
            neg_pairs = []
            for i in range(len(pos_pairs)):
                neg_pairs.append(
                    [
                        np.random.randint(len(coords_pos)),
                        np.random.randint(len(coords_neg)),
                    ]
                )
            pos_pairs_all += pos_pairs
            neg_pairs_all += neg_pairs
        pos_pairs = torch.tensor(pos_pairs_all)
        neg_pairs = torch.tensor(neg_pairs_all)
        assert pos_pairs.shape == neg_pairs.shape
        if len(pos_pairs) == 0:
            continue
        cluster_space_coords_filtered = cluster_space_coords[bmask]
        qs_filtered = q[bmask]
        pos_norms = (
            cluster_space_coords_filtered[pos_pairs[:, 0]]
            - cluster_space_coords_filtered[pos_pairs[:, 1]]
        ).norm(dim=-1)

        neg_norms = (
            cluster_space_coords_filtered[neg_pairs[:, 0]]
            - cluster_space_coords_filtered[neg_pairs[:, 1]]
        ).norm(dim=-1)
        q_pos = qs_filtered[pos_pairs[:, 0]]
        q_neg = qs_filtered[neg_pairs[:, 0]]
        q_s = torch.cat([q_pos, q_neg])
        norms_pos = torch.cat([pos_norms, neg_norms])
        ys = torch.cat([torch.ones_like(pos_norms), -torch.ones_like(neg_norms)])
        L_clusters += torch.sum(
            q_s * torch.nn.HingeEmbeddingLoss(reduce=None)(norms_pos, ys)
        )
        number_of_pairs += norms_pos.shape[0]
    if number_of_pairs > 0:
        L_clusters = L_clusters / number_of_pairs

    return L_clusters





