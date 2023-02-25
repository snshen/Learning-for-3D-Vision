import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
from pytorch3d.ops import sample_points_from_meshes
import losses
from eval_model import evaluate_in_train, evaluate_model
from tqdm import tqdm, trange
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    # Model parameters
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--max_epoch', default=150, type=int)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--save_freq', default=10000, type=int)    
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--load_checkpoint', action='store_true')    
    parser.add_argument('--with_eval', action='store_true')           
    return parser

def preprocess(feed_dict,args):
    images = feed_dict['images'].squeeze(1)
    if args.type == "vox":
        voxels = feed_dict['voxels'].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict['mesh']
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)    
        ground_truth_3d = pointclouds_tgt        
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    if args.load_feat:
        feats = torch.stack(feed_dict['feats'])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device)


def calculate_loss(predictions, ground_truth, args):
    if args.type == 'vox':
        loss = losses.voxel_loss(predictions.permute(1,0,2,3,4),ground_truth.permute(1,0,2,3,4))
    elif args.type == 'point':
        loss = losses.chamfer_loss(predictions, ground_truth)
    elif args.type == 'mesh':
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth   

    return loss


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        shuffle = True,
        drop_last=True)
    train_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)  # to use with ViTs
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting training !")
    for step in range(start_iter, args.max_epoch):
        iter_start_time = time.time()

        if step % len(train_loader) == 0: #restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, ground_truth_3d = preprocess(feed_dict,args)
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, args)
        
        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if (step % args.save_freq) == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'checkpoint_{args.type}.pth')

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (step, args.max_epoch, total_time, read_time, iter_time, loss_vis))

    print('Done!')



def train_with_eval(args):
    r2n2_dataset_t = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)
    t_loader = torch.utils.data.DataLoader(
        r2n2_dataset_t,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        shuffle = True,
        drop_last=True)
    train_loader = iter(t_loader)
    del r2n2_dataset_t
    torch.cuda.empty_cache()

    r2n2_dataset_e = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)
    e_loader = torch.utils.data.DataLoader(
        r2n2_dataset_e,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    del r2n2_dataset_e
    torch.cuda.empty_cache()

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.train()

    if args.type == "mesh":
        batch = args.batch_size
        args.batch_size = 1
        eval_model =  SingleViewto3D(args)
        eval_model.to(args.device)
        args.batch_size = batch

    # ============ preparing optimizer ... ============
    if args.type == "vox":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)  # to use with ViTs
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)
    elif args.type == "point":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)  # to use with ViTs
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=40, cooldown = 40)
    elif args.type == "mesh":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)  # to use with ViTs
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=40, cooldown = 40)

    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting training !")
    max_iter = len(train_loader)
    best_f1 = 0
    for epoch in range(args.max_epoch):
        epoch_loses = []
        with trange(max_iter) as tbatches:
            for step in tbatches:

                tbatches.set_description(f"Epoch {epoch}")
                train_loader = iter(train_loader)
                feed_dict = next(train_loader)
                images_gt, ground_truth_3d = preprocess(feed_dict,args)

                prediction_3d = model(images_gt, args)
                loss = calculate_loss(prediction_3d, ground_truth_3d, args)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tbatches.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
                epoch_loses.append(loss.cpu().item())

                # if step == max_iter-1:
                #     torch.save({
                #         'step': step,
                #         'model_state_dict': model.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict()
                #         }, f'checkpoint_{args.type}.pth')
                if (args.type == "point" or args.type == "mesh") and epoch>0:
                    lr_scheduler.step(loss)
                
                torch.cuda.empty_cache()

        train_loader = iter(t_loader)        
        eval_loader = iter(e_loader)
        if args.type == "mesh":
            eval_model.load_state_dict(model.state_dict())
            avg_f1_score = evaluate_in_train(args, eval_model, eval_loader)
        else:
            avg_f1_score = evaluate_in_train(args, model, eval_loader)
        avg_f1_mean = avg_f1_score.mean()
        model.train()
        if avg_f1_mean > best_f1:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, f'checkpoint_{args.type}_small.pth')
            print("Saved new best model!")
            best_f1 = avg_f1_mean
 
        total_time = time.time() - start_time        
        print("Epoch [%4d/%4d] | ttime: %.0f | loss: %.3f | AvgF1: %.3f | lr: %.3f" % (epoch, args.max_epoch, total_time, np.mean(epoch_loses), avg_f1_mean, optimizer.param_groups[0]['lr']))
        if args.type == "vox":
            lr_scheduler.step()

    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.with_eval:
        train_with_eval(args)
    else: 
        train_model(args)
    
