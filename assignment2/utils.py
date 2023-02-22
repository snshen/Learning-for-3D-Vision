import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
    FoVPerspectiveCameras,
    TexturesVertex,
    look_at_view_transform
)
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io import load_obj
from pytorch3d.ops import cubify
import numpy as np
import imageio
import torch


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def render_vox(voxels_src, voxels_tgt = None, mesh_tgt = None, src_path = "submissions/source_vox.gif", tgt_path = "submissions/target_vox.gif", num_views = 24):
#  def render_vox(voxels_src, voxels_tgt = None, src_path = "submissions/source_vox.gif", tgt_path = "submissions/target_vox.gif", num_views = 24):
    
    R, T = look_at_view_transform(dist=3, elev=0, azim=np.linspace(-180, 180, num_views, endpoint=False))
    many_cameras = FoVPerspectiveCameras(R=R, T=T, device=voxels_src.device)
    renderer = get_mesh_renderer(device=voxels_src.device)

    voxels_src = cubify(voxels_src, 0.5)
    src_verts = voxels_src.verts_list()[0]
    src_faces = voxels_src.faces_list()[0]
    textures = TexturesVertex(src_verts.unsqueeze(0))
    src_mesh = Meshes(verts=[src_verts], faces=[src_faces], textures = textures)

    my_images = renderer(src_mesh.extend(num_views), cameras=many_cameras)
    my_images = my_images.cpu().detach().numpy()
    imageio.mimsave(src_path, my_images, fps=12)

    if voxels_tgt is not None:
        voxels_tgt = cubify(voxels_tgt, 0.5)
        tgt_verts = voxels_tgt.verts_list()[0]
        tgt_faces = voxels_tgt.faces_list()[0]
        textures = TexturesVertex(tgt_verts.unsqueeze(0))
        tgt_mesh = Meshes(verts=[tgt_verts], faces=[tgt_faces], textures = textures)
        
        my_images = renderer(tgt_mesh.extend(num_views), cameras=many_cameras)
        my_images = my_images.cpu().detach().numpy()
        imageio.mimsave(tgt_path, my_images, fps=12)
    
    elif mesh_tgt is not None:
        tgt_verts = mesh_tgt.verts_list()[0]
        tgt_faces = mesh_tgt.faces_list()[0]
        textures = TexturesVertex(tgt_verts.unsqueeze(0))
        tgt_mesh = Meshes(verts=[tgt_verts], faces=[tgt_faces], textures = textures).to(voxels_src.device)
        
        my_images = renderer(tgt_mesh.extend(num_views), cameras=many_cameras)
        my_images = my_images.cpu().detach().numpy()
        imageio.mimsave(tgt_path, my_images, fps=12)
    
    return

def render_mesh(src_mesh, tgt_mesh = None, src_path = "submissions/source_mesh.gif", tgt_path = "submissions/target_mesh.gif", num_views = 24):

    R, T = look_at_view_transform(dist=3, elev=0, azim=np.linspace(-180, 180, num_views, endpoint=False))
    many_cameras = FoVPerspectiveCameras(R=R, T=T, device=src_mesh.device)
    renderer = get_mesh_renderer(device=src_mesh.device)
    
    src_verts = src_mesh.verts_list()[0]
    src_faces = src_mesh.faces_list()[0]
    textures = TexturesVertex(src_verts.unsqueeze(0))
    src_mesh = Meshes(verts=[src_verts], faces=[src_faces], textures = textures)
    
    my_images = renderer(src_mesh.extend(num_views), cameras=many_cameras)
    my_images = my_images.cpu().detach().numpy()
    imageio.mimsave(src_path, my_images, fps=12)

    if tgt_mesh is not None:
        tgt_verts = tgt_mesh.verts_list()[0]
        tgt_faces = tgt_mesh.faces_list()[0]
        textures = TexturesVertex(tgt_verts.unsqueeze(0))
        tgt_mesh = Meshes(verts=[tgt_verts], faces=[tgt_faces], textures = textures)
        
        my_images = renderer(tgt_mesh.extend(num_views), cameras=many_cameras)
        my_images = my_images.cpu().detach().numpy()
        imageio.mimsave(tgt_path, my_images, fps=12)

    return

def render_cloud(src_cloud, tgt_cloud = None, src_path = "submissions/source_cloud.gif", tgt_path = "submissions/target_cloud.gif", num_views = 24, dist = 3, radius = 0.03):

    R, T = look_at_view_transform(dist=dist, elev=0, azim=np.linspace(-180, 180, num_views, endpoint=False))
    many_cameras = FoVPerspectiveCameras(R=R, T=T, device=src_cloud.device)
    renderer = get_points_renderer(device=src_cloud.device, radius=radius)

    rgb = torch.ones_like(src_cloud) 
    src_cloud = Pointclouds(points=src_cloud, features=rgb).to(src_cloud.device)
    print(src_cloud.extend(num_views))

    my_images = renderer(src_cloud.extend(num_views), cameras=many_cameras)
    my_images = my_images.cpu().detach().numpy()
    imageio.mimsave(src_path, my_images, fps=12)
    
    if tgt_cloud is not None: 
        tgt_cloud = Pointclouds(points=tgt_cloud, features=rgb).to(src_cloud.device)
        my_images = renderer(tgt_cloud.extend(num_views), cameras=many_cameras)
        my_images = my_images.cpu().detach().numpy()
        imageio.mimsave(tgt_path, my_images, fps=12)
    
    return