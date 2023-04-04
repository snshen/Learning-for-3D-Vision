import os
import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
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

def viz_seg (verts, labels, path, args):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    device = args.device
    k = args.num_points

    image_size=256
    background_color=(1, 1, 1)
    colors = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float).to(device)
    sample_labels = labels.unsqueeze(0).to(device)
    sample_colors = torch.zeros((1,k,3)).to(device)

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels==i] = torch.tensor(colors[i]).to(device)

    sample_colors = sample_colors.repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)

    imageio.mimsave(path, rend, fps=15)


def viz_cloud(
    src_cloud,
    src_path = "submissions/source_cloud.gif", 
    num_views = 30, 
    radius = 0.02):

    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(num_views)]
    device = src_cloud.device
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
    renderer = get_points_renderer(device=src_cloud.device, radius=radius)

    rgb = (src_cloud - src_cloud.min()) / (src_cloud.max() - src_cloud.min())
    src_cloud = pytorch3d.structures.Pointclouds(points=src_cloud, features=rgb).to(src_cloud.device)

    my_images = renderer(src_cloud.extend(num_views), cameras=c)
    my_images = my_images.cpu().detach().numpy()
    imageio.mimsave(src_path, my_images, fps=15)
    
    return
