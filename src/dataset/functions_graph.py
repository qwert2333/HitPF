import numpy as np
import torch
import dgl
from src.dataset.functions_data import (
    calculate_distance_to_boundary,
)
import time
from src.dataset.functions_particles import concatenate_Particles_GT, Particles_GT

from src.dataset.dataclasses import Hits

def create_inputs_from_table(
    output, prediction=False, args=None
):
    number_hits = np.int32(len(output["X_track"])+len(output["X_hit"]))
    number_part = np.int32(len(output["X_gen"]))

    hits = Hits.from_data(
    output,
    number_hits,
    args,
    number_part
    )

    y_data_graph = Particles_GT()
    y_data_graph.fill( output, prediction,args)

    result = [
        y_data_graph,  
        hits
    ]
    return result
 



def create_graph(
    output,
    for_training =True, args=None
):
    prediction = not for_training
    graph_empty = False
   
    result = create_inputs_from_table(
        output,
        prediction=prediction, 
        args=args
    )

    if len(result) == 1:
        graph_empty = True
        return [0, 0], graph_empty
    else:
        (y_data_graph,hits) = result

        g = dgl.graph(([], []))
        g.add_nodes(hits.pos_xyz_hits.shape[0])
        g.ndata["h"] = torch.cat(
                (hits.pos_xyz_hits, hits.hit_type_one_hot, hits.e_hits, hits.p_hits), dim=1
            ).float()  
        g.ndata["p_hits"] = hits.p_hits.float() 
        g.ndata["pos_hits_xyz"] = hits.pos_xyz_hits.float()
        g.ndata["pos_pxpypz_at_vertex"] = hits.pos_pxpypz.float()
        g.ndata["pos_pxpypz"] = hits.pos_pxpypz  #TrackState::AtIP
        g.ndata["pos_pxpypz_at_calo"] = hits.pos_pxpypz_calo  #TrackState::AtCalorimeter
        g = calculate_distance_to_boundary(g)
        g.ndata["hit_type"] = hits.hit_type_feature.float()
        g.ndata["e_hits"] = hits.e_hits.float()  
 
        g.ndata["chi_squared_tracks"] = hits.chi_squared_tracks.float()
        g.ndata["particle_number"] = hits.hit_particle_link.float()+1 #(noise idx is 0 and particle MC 0 starts at 1)
 

        if prediction and (args.pandora):
            g.ndata["pandora_pfo"] = hits.pandora_features.pandora_pfo_link.float()
            g.ndata["pandora_pfo_energy"] = hits.pandora_features.pfo_energy.float()
            g.ndata["pandora_momentum"] = hits.pandora_features.pandora_mom_components.float()
            g.ndata["pandora_reference_point"] = hits.pandora_features.pandora_ref_point.float()
            g.ndata["pandora_pid"] = hits.pandora_features.pandora_pid.float()
        graph_empty = False
        if torch.unique(hits.hit_particle_link).shape[0]==1 and torch.unique(hits.hit_particle_link)[0]==-1:
            graph_empty = True 
        if hits.pos_xyz_hits.shape[0] < 10:
            graph_empty = True
    
    return [g, y_data_graph], graph_empty





def graph_batch_func(list_graphs):
    """collator function for graph dataloader

    Args:
        list_graphs (list): list of graphs from the iterable dataset

    Returns:
        batch dgl: dgl batch of graphs
    """
    list_graphs_g = [el[0] for el in list_graphs]
    ys = concatenate_Particles_GT(list_graphs)
    bg = dgl.batch(list_graphs_g)
    return bg, ys
