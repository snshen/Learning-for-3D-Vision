import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):  
        
        # # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        # z_range = torch.arange(start=self.min_depth, end=self.max_depth, step=(self.max_depth-self.min_depth)/self.n_pts_per_ray).to(ray_bundle.directions.device)

        # # TODO (1.4): Sample points from z values
        # # Return
        # return ray_bundle._replace(
        #     sample_points=ray_bundle.origins.unsqueeze(1) + torch.einsum('mi,n->mni', ray_bundle.directions, z_range),
        #     sample_lengths=torch.tile(z_range,(ray_bundle.directions.shape[0],1)).unsqueeze(2) ,
        # )
        z_range = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray).to(ray_bundle.directions.device)
        # TODO (1.4): Sample points from z values
        sample_lengths = z_range.view(1,-1,1).expand(ray_bundle.directions.shape[0],-1,-1)
        sample_points = ray_bundle.origins.unsqueeze(1) + ray_bundle.directions.unsqueeze(1)*sample_lengths.expand(-1,-1,3)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=sample_lengths.reshape(-1,1),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}