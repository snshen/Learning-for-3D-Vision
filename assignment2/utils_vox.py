import torch
import numpy as np
import torch.nn.functional as F


XMIN = -0.5 # right (neg is left)
XMAX = 0.5 # right
YMIN = -0.5 # down (neg is up)
YMAX = 0.5 # down
ZMIN = -0.5# forward
ZMAX = 0.5 # forward


def voxelize_xyz(xyz_ref, Z, Y, X, already_mem=False):
    B, N, D = list(xyz_ref.shape)
    assert(D==3)
    if already_mem:
        xyz_mem = xyz_ref
    else:
        xyz_mem = Ref2Mem(xyz_ref, Z, Y, X)
    vox = get_occupancy(xyz_mem, Z, Y, X)
    return vox

    
def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    xyz2 = xyz2[:,:,:3]
    return xyz2

def eye_4x4(B, device='cpu'):
    rt = torch.eye(4, device=torch.device(device)).view(1,4,4).repeat([B, 1, 1])
    return rt

def Ref2Mem(xyz, Z, Y, X):
    # xyz is B x N x 3, in ref coordinates
    # transforms velo coordinates into mem coordinates
    B, N, C = list(xyz.shape)
    mem_T_ref = get_mem_T_ref(B, Z, Y, X)
    xyz = apply_4x4(mem_T_ref, xyz)
    return xyz

def get_occupancy(xyz, Z, Y, X):
    # xyz is B x N x 3 and in mem coords
    # we want to fill a voxel tensor with 1's at these inds
    B, N, C = list(xyz.shape)
    assert(C==3)

    # these papers say simple 1/0 occupancy is ok:
    #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf
    #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
    # cont fusion says they do 8-neighbor interp
    # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

    inbounds = get_inbounds(xyz, Z, Y, X, already_mem=True)
    x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
    mask = torch.zeros_like(x)
    mask[inbounds] = 1.0

    # set the invalid guys to zero
    # we then need to zero out 0,0,0
    # (this method seems a bit clumsy)
    x = x*mask
    y = y*mask
    z = z*mask

    x = torch.round(x)
    y = torch.round(y)
    z = torch.round(z)
    x = torch.clamp(x, 0, X-1).int()
    y = torch.clamp(y, 0, Y-1).int()
    z = torch.clamp(z, 0, Z-1).int()

    x = x.view(B*N)
    y = y.view(B*N)
    z = z.view(B*N)

    dim3 = X
    dim2 = X * Y
    dim1 = X * Y * Z

    # base = torch.from_numpy(np.concatenate([np.array([i*dim1]) for i in range(B)]).astype(np.int32))
    # base = torch.range(0, B-1, dtype=torch.int32, device=torch.device('cpu'))*dim1
    base = torch.arange(0, B, dtype=torch.int32, device=torch.device('cpu'))*dim1
    base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B*N)

    vox_inds = base + z * dim2 + y * dim3 + x
    voxels = torch.zeros(B*Z*Y*X, device=torch.device('cpu')).float()
    voxels[vox_inds.long()] = 1.0
    # zero out the singularity
    voxels[base.long()] = 0.0
    voxels = voxels.reshape(B, 1, Z, Y, X)
    # B x 1 x Z x Y x X
    return voxels

def matmul2(mat1, mat2):
    return torch.matmul(mat1, mat2)

def Mem2Ref(xyz_mem, Z, Y, X, device='cpu'):
    # xyz is B x N x 3, in mem coordinates
    # transforms mem coordinates into ref coordinates
    B, N, C = list(xyz_mem.shape)
    # st()
    ref_T_mem = get_ref_T_mem(B, Z, Y, X, device=device)
    xyz_ref = apply_4x4(ref_T_mem, xyz_mem)
    return xyz_ref

def get_ref_T_mem(B, Z, Y, X, device='cpu'):
    mem_T_ref = get_mem_T_ref(B, Z, Y, X, device=device)
    # note safe_inverse is inapplicable here,
    # since the transform is nonrigid
    ref_T_mem = mem_T_ref.inverse()
    return ref_T_mem

def get_mem_T_ref(B, Z, Y, X, device='cpu'):
    # sometimes we want the mat itself
    # note this is not a rigid transform
    
    # for interpretability, let's construct this in two steps...

    # translation
    center_T_ref = eye_4x4(B, device=device)
    center_T_ref[:,0,3] = -XMIN
    center_T_ref[:,1,3] = -YMIN
    center_T_ref[:,2,3] = -ZMIN

    VOX_SIZE_X = (XMAX-XMIN)/float(X)
    VOX_SIZE_Y = (YMAX-YMIN)/float(Y)
    VOX_SIZE_Z = (ZMAX-ZMIN)/float(Z)
    
    # scaling
    mem_T_center = eye_4x4(B, device=device)
    mem_T_center[:,0,0] = 1./VOX_SIZE_X
    mem_T_center[:,1,1] = 1./VOX_SIZE_Y
    mem_T_center[:,2,2] = 1./VOX_SIZE_Z
    mem_T_ref = matmul2(mem_T_center, center_T_ref)
    
    return mem_T_ref

def get_inbounds(xyz, Z, Y, X, already_mem=False):
    # xyz is B x N x 3
    if not already_mem:
        xyz = Ref2Mem(xyz, Z, Y, X)

    x = xyz[:,:,0]
    y = xyz[:,:,1]
    z = xyz[:,:,2]
    
    x_valid = (x>-0.5).byte() & (x<float(X-0.5)).byte()
    y_valid = (y>-0.5).byte() & (y<float(Y-0.5)).byte()
    z_valid = (z>-0.5).byte() & (z<float(Z-0.5)).byte()
    
    inbounds = x_valid & y_valid & z_valid
    return inbounds.bool()


