import numpy as np
import torch
from ray_utils import RayBundle

def phong(
    normals,
    view_dirs, 
    light_dir,
    params,
    colors
):
    # TODO: Implement a simplified version Phong shading
    # Inputs:
    #   normals: (N x d, 3) tensor of surface normals
    #   view_dirs: (N x d, 3) tensor of view directions
    #   light_dir: (3,) tensor of light direction
    #   params: dict of Phong parameters
    #   colors: (N x d, 3) tensor of colors
    # Outputs:
    #   illumination: (N, 3) tensor of shaded colors
    #
    # Note: You can use torch.clamp to clamp the dot products to [0, 1]
    # Assume the ambient light (i_a) is of unit intensity 
    # While the general Phong model allows rerendering with multiple lights, 
    # here we only implement a single directional light source of unit intensity
    ka, kd, ks, n = params['ka'], params['kd'], params['ks'], params['n']
    R = 2 * torch.sum(normals * light_dir, dim=1, keepdim=True) * normals-light_dir
    diffuse = kd * torch.clamp(torch.sum(normals * light_dir, dim=1, keepdim=True),0,1)
    specular = ks * torch.pow(torch.clamp(torch.sum(R * view_dirs, dim=1, keepdim=True),0,1),n)
    illumination = (ka+diffuse+specular)*colors

    return illumination

relighting_dict = {
    'phong': phong
}