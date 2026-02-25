import numpy as np
import torch


def calculate_distance_to_boundary(g):
    r = 2150
    r_in_endcap = 2307
    mask_endcap = (torch.abs(g.ndata["pos_hits_xyz"][:, 2]) - r_in_endcap) > 0
    mask_barrer = ~mask_endcap
    weight = torch.ones_like(g.ndata["pos_hits_xyz"][:, 0])
    C = g.ndata["pos_hits_xyz"]
    A = torch.tensor([0, 0, 1], dtype=C.dtype, device=C.device)
    P = (
        r
        * 1
        / (torch.norm(torch.cross(A.view(1, -1), C, dim=-1), dim=1)).unsqueeze(1)
        * C
    )
    P1 = torch.abs(r_in_endcap / g.ndata["pos_hits_xyz"][:, 2].unsqueeze(1)) * C
    weight[mask_barrer] = torch.norm(P - C, dim=1)[mask_barrer]
    weight[mask_endcap] = torch.norm(P1[mask_endcap] - C[mask_endcap], dim=1)
    g.ndata["radial_distance"] = weight
    weight_ = torch.exp(-(weight / 1000))
    g.ndata["radial_distance_exp"] = weight_
    return g

