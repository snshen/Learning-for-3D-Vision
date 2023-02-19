import pytorch3d
import torch
from starter import utils, dolly_zoom, camera_transforms, render_generic
import matplotlib.pyplot as plt
import imageio
import numpy as np
from helper import *

# This should print True if you are using your GPU
print("Using GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("WARNING: Code was written on cpu, not optomized for cuda")
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


#1. Practicing with Cameras
##1.1. 360-degree Renders (5 points)
vertices, faces = utils.load_cow_mesh(path="data/cow.obj")
vertices = vertices.unsqueeze(0) 
faces = faces.unsqueeze(0)

texture_rgb = torch.ones_like(vertices)
texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1])
textures = pytorch3d.renderer.TexturesVertex(texture_rgb)

meshes = pytorch3d.structures.Meshes(
    verts=vertices,
    faces=faces,
    textures=textures,
).to(device) 

num_views = 24
R, T = pytorch3d.renderer.look_at_view_transform(
    dist=3,
    elev=0,
    azim=np.linspace(-180, 180, num_views, endpoint=False),
)
many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
    R=R,
    T=T,
    device=device
)

image_size = 512
renderer = utils.get_mesh_renderer(image_size=image_size, device=device)
lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
my_images = renderer(meshes.extend(num_views), cameras=many_cameras, lights=lights) 

imageio.mimsave("submissions/360.gif", my_images, fps=12)
print("Completed 1.1")

##1.2 Re-creating the Dolly Zoom (10 points)
dolly_zoom.dolly_zoom(num_frames=30, output_file="submissions/dolly_zoom.gif")
print("Completed 1.2")

#2. Practicing with Meshes
##2.1 Constructing a Tetrahedron (5 points)
t_vertices = torch.tensor([[1., -1., 0.], [-1., -1., 0.], [0.2, 1., 0.], [0., 0., 1.]])
t_faces = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 0], [0, 1, 3]])
t_vertices = t_vertices.unsqueeze(0)
t_faces = t_faces.unsqueeze(0)

t_texture_rgb = torch.ones_like(t_vertices)
t_texture_rgb = t_texture_rgb * torch.tensor([0.7, 0.7, 1])
t_textures = pytorch3d.renderer.TexturesVertex(t_texture_rgb)

t_meshes = pytorch3d.structures.Meshes(
    verts=t_vertices,
    faces=t_faces,
    textures=t_textures,
)

my_images = renderer(t_meshes.extend(num_views), cameras=many_cameras, lights=lights)
imageio.mimsave("submissions/360_tetra.gif", my_images, fps=12)
print("Completed 2.1")

##2.2 Constructing a Cube (5 points)
c_vertices = torch.tensor([[1., 0., -0.7071], [0., -1., -0.7071], [-1., 0., -0.7071], [0., 1., -0.7071], 
                        [1., 0., 0.7071], [0., -1., 0.7071], [-1., 0., 0.7071], [0., 1., 0.7071]])
c_faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 4, 5], [0, 5, 1], [0, 4, 7], [0, 7, 3],
                    [6, 7, 3], [6, 2, 3], [6, 1, 5], [6, 2, 1], [6, 4, 7], [6, 5, 4]])

c_vertices = c_vertices.unsqueeze(0)
c_faces = c_faces.unsqueeze(0)

c_texture_rgb = torch.ones_like(c_vertices)
c_texture_rgb = c_texture_rgb * torch.tensor([0.7, 0.7, 1])
c_textures = pytorch3d.renderer.TexturesVertex(c_texture_rgb)

c_meshes = pytorch3d.structures.Meshes(
    verts=c_vertices,
    faces=c_faces,
    textures=c_textures,
)

my_images = renderer(c_meshes.extend(num_views), cameras=many_cameras, lights=lights)
imageio.mimsave("submissions/360_cube.gif", my_images, fps=12)
print("Completed 2.2")

#3. Re-texturing a mesh (10 points)
texture_rgb = vertices.clone()
texture_rgb = (texture_rgb - texture_rgb[:,:,2].min()) / (texture_rgb[:,:,2].max() - texture_rgb[:,:,2].min())
texture_rgb[:,:,0] = texture_rgb[:,:,2]
texture_rgb[:,:,1] = texture_rgb[:,:,2]

color1 = torch.tensor([0, 0.5, 1])
color2 = torch.tensor([1, 0.5, 0])
texture_rgb = texture_rgb * color2 + (1 - texture_rgb) * color1
retexture = pytorch3d.renderer.TexturesVertex(texture_rgb.to(device))

remeshes = pytorch3d.structures.Meshes(
    verts=vertices,
    faces=faces,
    textures=retexture,
) .to(device) 
remeshes.textures = retexture

my_images = renderer(remeshes.extend(num_views), cameras=many_cameras, lights=lights)
imageio.mimsave("submissions/retexture.gif", my_images, fps=12)
print("Completed 3")

#4. Camera Transformations (10 points)

## image 1
R_relative=[[np.cos(np.pi/2), -np.sin(np.pi/2), 0], [np.sin(np.pi/2), np.cos(np.pi/2), 0], [0, 0, 1]]
T_relative=[0, 0, 0]
image1 = camera_transforms.render_cow(R_relative = R_relative, T_relative=T_relative)
plt.imsave("submissions/cow_trans1.jpg", image1)

## image 2
R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
T_relative=[0, 0, 2]
image2 = camera_transforms.render_cow(R_relative = R_relative, T_relative=T_relative)
plt.imsave("submissions/cow_trans2.jpg", image2)

## image 3
R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
T_relative=[0.5, -0.5, 0]
image3 = camera_transforms.render_cow(R_relative = R_relative, T_relative=T_relative)
plt.imsave("submissions/cow_trans3.jpg", image3)

## image 4
R_relative=[[np.cos(-np.pi/2), 0, np.sin(-np.pi/2)], [0, 1, 0], [-np.sin(-np.pi/2), 0, np.cos(-np.pi/2)]]
T_relative=[3, 0, 3]
image4 = camera_transforms.render_cow(R_relative = R_relative, T_relative=T_relative)
plt.imsave("submissions/cow_trans4.jpg", image4)

print("Completed 4")

#5. Rendering Generic 3D Representations
##5.1 Rendering Point Clouds from RGB-D Images (10 points)
data_dict = render_generic.load_rgbd_data()

points1, rgb1 = utils.unproject_depth_image(torch.Tensor(data_dict["rgb1"]), 
                                torch.Tensor(data_dict["mask1"]), 
                                torch.Tensor(data_dict["depth1"]), 
                                data_dict["cameras1"])
pc1 = pytorch3d.structures.Pointclouds(
    points=points1.unsqueeze(0),
    features=rgb1.unsqueeze(0),
).to(device)

points2, rgb2 = utils.unproject_depth_image(torch.Tensor(data_dict["rgb2"]), 
                                torch.Tensor(data_dict["mask2"]), 
                                torch.Tensor(data_dict["depth2"]), 
                                data_dict["cameras2"])
pc2 = pytorch3d.structures.Pointclouds(
    points=points2.unsqueeze(0),
    features=rgb2.unsqueeze(0),
).to(device)

pc3 = pytorch3d.structures.Pointclouds(
    points=torch.cat((points1,points2), 0).unsqueeze(0),
    features=torch.cat((rgb1,rgb2), 0).unsqueeze(0),
).to(device)


R0 = torch.tensor([[float(np.cos(np.pi)), float(-np.sin(np.pi)), 0.], [float(np.sin(np.pi)), float(np.cos(np.pi)), 0.], [0., 0., 1.]])
pc_R, pc_T = pytorch3d.renderer.look_at_view_transform(
    dist=6,
    elev=0,
    azim=np.linspace(-180, 180, num_views, endpoint=False),
)
pc_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
    R=pc_R@R0,
    T=pc_T,
    device=device
)

pc_renderer = utils.get_points_renderer(device=device, radius=0.03)

my_images = pc_renderer(pc1.extend(num_views), cameras=pc_cameras, lights=lights)
imageio.mimsave("submissions/pc1.gif", my_images, fps=4)

my_images = pc_renderer(pc2.extend(num_views), cameras=pc_cameras, lights=lights)
imageio.mimsave("submissions/pc2.gif", my_images, fps=4)

my_images = pc_renderer(pc3.extend(num_views), cameras=pc_cameras, lights=lights)
imageio.mimsave("submissions/pc3.gif", my_images, fps=4)
print("Completed 5.1")

##5.2 Parametric Functions (10 points)
render_torus()
print("Completed 5.2")

##5.3 Implicit Surfaces (15 points)
render_torus_mesh()
print("Completed 5.3")

#6. Do Something Fun (10 points)

vertices = vertices+torch.tensor([0, 0, -0.5])
texture_rgb = torch.ones_like(vertices)
texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1])
textures = pytorch3d.renderer.TexturesVertex(texture_rgb)

cow_mesh = pytorch3d.structures.Meshes(
    verts=vertices,
    faces=faces,
    textures=textures,
)
torus_mesh  =  render_torus_mesh(r0 = 0.6, r1 = .15 ,gif=False)
scene_mesh = pytorch3d.structures.meshes.join_meshes_as_scene([cow_mesh, torus_mesh])

R = torch.eye(3).unsqueeze(0)
T = torch.tensor([[0, 0, 3]])


cameras = pytorch3d.renderer.FoVPerspectiveCameras(
    R=R,
    T=T,
    fov=60,
    device=device,
)
transform = cameras.get_world_to_view_transform()

texture_rgb = torch.ones_like(vertices)
texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1])
textures = pytorch3d.renderer.TexturesVertex(texture_rgb)

cow_mesh = pytorch3d.structures.Meshes(
    verts=vertices,
    faces=faces,
    textures=textures,
)
torus_mesh  =  render_torus_mesh(r0 = 0.6, r1 = .15 ,gif=False)
scene_mesh = pytorch3d.structures.meshes.join_meshes_as_scene([cow_mesh, torus_mesh])

ang = np.pi/3
R0=torch.tensor([[1., 0., 0.], [0., float(np.cos(ang)), float(np.sin(ang))], [0., float(-np.sin(ang)), float(np.cos(ang))]]).unsqueeze(0)

trans = pytorch3d.transforms.Rotate(R0)
new_verts = trans.transform_points(scene_mesh.verts_list()[0])
new_faces = scene_mesh.faces_list()[0]
new_textures = (new_verts - new_verts.min()) / (new_verts.max() - new_verts.min())
new_textures = pytorch3d.renderer.TexturesVertex(new_verts.unsqueeze(0))
new_mesh = pytorch3d.structures.Meshes(
    verts=[new_verts],   
    faces=[new_faces],
    textures = new_textures
)

R, T = pytorch3d.renderer.look_at_view_transform(
    dist=3,
    elev=0,
    azim=np.linspace(-180, 180, num_views, endpoint=False),
)
many_cameras = pytorch3d.renderer.FoVPerspectiveCameras(
    R=R,
    T=T,
    device=device
)
my_images = renderer(new_mesh.extend(num_views), cameras=many_cameras, lights=lights)
imageio.mimsave("submissions/fun.gif", my_images, fps=12)
print("Completed 6")

#(Extra Credit) 7. Sampling Points on Meshes (10 points)

pc_renderer = utils.get_points_renderer(device=device, radius=0.01)

sample_pc = sample_points(mesh_path = "data/cow.obj", num_samples = 100)
my_images = pc_renderer(sample_pc.extend(num_views), cameras=many_cameras)
imageio.mimsave("submissions/cow100.gif", my_images, fps=12)

sample_pc = sample_points(mesh_path = "data/cow.obj", num_samples = 500)
my_images = pc_renderer(sample_pc.extend(num_views), cameras=many_cameras)
imageio.mimsave("submissions/cow500.gif", my_images, fps=12)

sample_pc = sample_points(mesh_path = "data/cow.obj", num_samples = 1000)
my_images = pc_renderer(sample_pc.extend(num_views), cameras=many_cameras)
imageio.mimsave("submissions/cow1000.gif", my_images, fps=12)

sample_pc = sample_points(mesh_path = "data/cow.obj", num_samples = 10000)
my_images = pc_renderer(sample_pc.extend(num_views), cameras=many_cameras)
imageio.mimsave("submissions/cow10000.gif", my_images, fps=12)

sample_pc = sample_points(mesh_path = "data/joint_mesh.obj", num_samples = 100)
my_images = pc_renderer(sample_pc.extend(num_views), cameras=many_cameras)
imageio.mimsave("submissions/joint_mesh100.gif", my_images, fps=12)

sample_pc = sample_points(mesh_path = "data/joint_mesh.obj", num_samples = 500)
my_images = pc_renderer(sample_pc.extend(num_views), cameras=many_cameras)
imageio.mimsave("submissions/joint_mesh500.gif", my_images, fps=12)

sample_pc = sample_points(mesh_path = "data/joint_mesh.obj", num_samples = 1000)
my_images = pc_renderer(sample_pc.extend(num_views), cameras=many_cameras)
imageio.mimsave("submissions/joint_mesh1000.gif", my_images, fps=12)

sample_pc = sample_points(mesh_path = "data/joint_mesh.obj", num_samples = 10000)
my_images = pc_renderer(sample_pc.extend(num_views), cameras=many_cameras)
imageio.mimsave("submissions/joint_mesh10000.gif", my_images, fps=12)

joint_mesh = vertices, faces = utils.load_cow_mesh(path="data/joint_mesh.obj")
vertices = vertices.unsqueeze(0) 
faces = faces.unsqueeze(0)

texture_rgb = torch.ones_like(vertices)
texture_rgb = texture_rgb * torch.tensor([0.7, 0.7, 1])
textures = pytorch3d.renderer.TexturesVertex(texture_rgb)

meshes = pytorch3d.structures.Meshes(
    verts=vertices,
    faces=faces,
    textures=textures,
).to(device) 

my_images = renderer(meshes.extend(num_views), cameras=many_cameras, lights=lights)
imageio.mimsave("submissions/joint_mesh.gif", my_images, fps=12)
print("Completed 7")