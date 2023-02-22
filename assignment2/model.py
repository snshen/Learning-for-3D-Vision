from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 1 x 32 x 32 x 32
            self.layer1 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1),
                torch.nn.BatchNorm3d(32),
                torch.nn.ReLU()
            )
            self.layer2 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, bias=False, padding=1),
                torch.nn.BatchNorm3d(16),
                torch.nn.ReLU()
            )
            self.layer3 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, bias=False, padding=1),
                torch.nn.BatchNorm3d(8),
                torch.nn.ReLU()
            )
            self.layer4 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, bias=False, padding=1),
                torch.nn.BatchNorm3d(4),
                torch.nn.ReLU()
            )
            self.layer5 = torch.nn.Sequential(
                torch.nn.ConvTranspose3d(4, 1, kernel_size=1, bias=False),
                torch.nn.Sigmoid()
            )
            # self.decoder =             
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            # self.decoder =             
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            # self.decoder =             

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # for features in image_features:
            gen_volume = encoded_feat.view(-1, 64, 2, 2, 2)
            # print(gen_volume.size())   # torch.Size([batch_size, 2048, 2, 2, 2])
            gen_volume = self.layer1(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
            gen_volume = self.layer2(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 128, 8, 8, 8])
            gen_volume = self.layer3(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 32, 16, 16, 16])
            gen_volume = self.layer4(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])
            voxels_pred = self.layer5(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
                 
            return voxels_pred

        elif args.type == "point":
            # TODO:
            # pointclouds_pred =             
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            # deform_vertices_pred =             
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

