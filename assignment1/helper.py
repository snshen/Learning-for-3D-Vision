
import pytorch3d
import torch
import matplotlib.pyplot as plt
from starter import utils
import imageio
import numpy as np
import mcubes

def render_torus_mesh(image_size=256, voxel_size=64, r0 = 3, r1 = 1 , gif=True, num_views = 24):

    min_value = -4.1
    max_value = 4.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)

    voxels = (X ** 2 + Y ** 2 + Z ** 2 + r0 ** 2 - r1 ** 2) ** 2 - 4 * r0 ** 2 * (X ** 2 + Y ** 2)

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures)

    if gif:
        lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]])
        renderer = utils.get_mesh_renderer(image_size=image_size)

        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=10,
            elev=0,
            azim=np.linspace(-180, 180, num_views, endpoint=False),
        )
        many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T
        )

        renderer = utils.get_mesh_renderer(image_size=image_size)

        my_images = renderer(mesh.extend(num_views), cameras=many_cameras)
        imageio.mimsave("submissions/torus_mesh.gif", my_images, fps=4)

    return mesh

def render_torus(image_size=256, num_samples=200, r0 = 3, r1 = 1 , gif=True, num_views = 24):
    """
    Renders a torus using parametric sampling. Samples num_samples ** 2 points.
    """

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = (r0+r1*torch.cos(Theta))*torch.cos(Phi)
    y = (r0+r1*torch.cos(Theta))*torch.sin(Phi)
    z = r1*torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(points=[points], features=[color])
    if gif:
        R, T = pytorch3d.renderer.look_at_view_transform(
            dist=10,
            elev=0,
            azim=np.linspace(-180, 180, num_views, endpoint=False),
        )
        many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T
        )

        renderer = utils.get_points_renderer(radius=0.03)
        my_images = renderer(torus_point_cloud.extend(num_views), cameras=many_cameras)
        imageio.mimsave("submissions/torus.gif", my_images, fps=4)
    return torus_point_cloud

def sample_points(mesh_path = "data/cow.obj", num_samples = 500):
    
    # Get the vertex coordinates of the faces
    samples = torch.zeros((num_samples, 3))
    verts, faces, aux = pytorch3d.io.load_obj(mesh_path)
    face_verts = verts[faces.verts_idx]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Get area and weights
    areas = abs(0.5 * torch.bmm((v0-v1).unsqueeze(1), (v0-v2).unsqueeze(2)))
    weight = areas / sum(areas)

    # Randomly generate sampling indices
    sample_ind = []
    for i in range(num_samples):
        thresh = np.random.uniform(0, 1)
        for j, w in enumerate(weight):
            thresh -= w
            if thresh < 0:
                sample_ind.append(j)
                break
            
    uv = torch.rand(2, 1, num_samples)
    u, v = uv[0], uv[1]
    w0 = 1.0 - u.sqrt()
    w1 = u.sqrt() * (1.0 - v)
    w2 = u.sqrt() * v

    # Use indices to get a barycentric point on each face.
    a, b, c = v0[sample_ind], v1[sample_ind], v2[sample_ind]
    samples = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c
    samples = samples.squeeze(0)

    # Generate point cloud
    color = (samples - samples.min()) / (samples.max() - samples.min())
    pc = pytorch3d.structures.Pointclouds(
        points=[samples], features = [color]
    )

    return pc