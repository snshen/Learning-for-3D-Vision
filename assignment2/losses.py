import torch
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	voxel_src.unsqueeze(1)
	voxel_tgt.type(dtype=torch.LongTensor)
	# loss = torch.nn.functional.cross_entropy(voxel_src, voxel_tgt)
	loss = torch.nn.functional.binary_cross_entropy(voxel_src, voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src, point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	p1_dists, _, _ = knn_points(point_cloud_src, point_cloud_tgt)
	p2_dists, _, _ = knn_points(point_cloud_tgt, point_cloud_src)
	# implement chamfer loss from scratch
	loss_chamfer = torch.sum((p1_dists + p2_dists)) / point_cloud_src.shape[0]
	return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss
	return loss_laplacian