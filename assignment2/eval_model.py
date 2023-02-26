import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
import dataset_location
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes
import mcubes
import utils_vox
import matplotlib.pyplot as plt 

from tqdm import trange

from utils import *

ids = [1, 2, 5]

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=10000, type=str)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--batch_size', default=1, type=str)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        H,W,D = voxels_src.shape[2:]

        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = Meshes([vertices_src], [faces_src])
        
        if len(mesh_src.verts_list()[0]) == 0:
            return None
        
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics


def evaluate_model(args):

    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []
    
    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):

        iter_start_time = time.time()
        read_start_time = time.time()
        feed_dict = next(eval_loader)
        images_gt, mesh_gt = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)
        if args.type == "vox":
            predictions = predictions.permute(0,1,4,3,2)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        if metrics is None:
            print("WARNING: empty mesh found for evaluation ", step)
            continue
        
        if args.type == "point":
            pointclouds_tgt = sample_points_from_meshes(mesh_gt, args.n_points)

        if step == ids[0]:
            if args.type == "vox":
                render_vox(predictions.squeeze(1), mesh_tgt=mesh_gt, src_path = "data/vox/model_vox0.gif", tgt_path = "data/vox/model_vox_t0.gif" )
            elif args.type == "point":
                render_cloud(predictions.squeeze(1), tgt_cloud=pointclouds_tgt, src_path = "data/point/model_cloud0.gif", tgt_path = "data/point/model_cloud_t0.gif", radius = 0.015)
            elif args.type == "mesh":
                render_mesh(predictions, tgt_mesh=mesh_gt, src_path = "data/mesh/model_mesh0.gif", tgt_path = "data/mesh/model_mesh_t0.gif" )
        if step == ids[1]:
            if args.type == "vox":
                render_vox(predictions.squeeze(1), mesh_tgt=mesh_gt, src_path = "data/vox/model_vox1.gif", tgt_path = "data/vox/model_vox_t1.gif" )
            elif args.type == "point":
                render_cloud(predictions.squeeze(1), tgt_cloud=pointclouds_tgt, src_path = "data/point/model_cloud1.gif", tgt_path = "data/point/model_cloud_t1.gif", radius = 0.015)
            elif args.type == "mesh":
                render_mesh(predictions, tgt_mesh=mesh_gt, src_path = "data/mesh/model_mesh1.gif", tgt_path = "data/mesh/model_mesh_t1.gif" )
        if step == ids[2]:
            if args.type == "vox":
                render_vox(predictions.squeeze(1), mesh_tgt=mesh_gt, src_path = "data/vox/model_vox2.gif", tgt_path = "data/vox/model_vox_t2.gif" )
            elif args.type == "point":
                render_cloud(predictions.squeeze(1), tgt_cloud=pointclouds_tgt, src_path = "data/point/model_cloud2.gif", tgt_path = "data/point/model_cloud_t2.gif", radius = 0.015)
            elif args.type == "mesh":
                render_mesh(predictions, tgt_mesh=mesh_gt, src_path = "data/mesh/model_mesh2.gif", tgt_path = "data/mesh/model_mesh_t2.gif" )

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    
    print("The f1 scores of visualized models are: %.3f, %.3f, and %.3f" % (avg_f1_score_05[ids[0]], avg_f1_score_05[ids[1]], avg_f1_score_05[ids[2]]))
    avg_f1_score = torch.stack(avg_f1_score).mean(0)
    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

def visualize_model(args):

    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
    
    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        if step>ids[2]: break
        feed_dict = next(eval_loader)
        if step in ids:
            images_gt, mesh_gt = preprocess(feed_dict, args)
            predictions = model(images_gt, args)

            if args.type == "vox":
                predictions = predictions.permute(0,1,4,3,2)

            metrics = evaluate(predictions, mesh_gt, thresholds, args)

            if metrics is None:
                print("WARNING: empty mesh found for evaluation ", step)
                continue
            
            if args.type == "point":
                pointclouds_tgt = sample_points_from_meshes(mesh_gt, args.n_points)

            if step == ids[0]:
                if args.type == "vox":
                    render_vox(predictions.squeeze(1), mesh_tgt=mesh_gt, src_path = "data/edge/model_vox0.gif", tgt_path = "data/edge/model_vox_t0.gif" )
                elif args.type == "point":
                    render_cloud(predictions.squeeze(1), tgt_cloud=pointclouds_tgt, src_path = "data/point/model_cloud0.gif", tgt_path = "data/point/model_cloud_t0.gif", radius = 0.015)
                elif args.type == "mesh":
                    render_mesh(predictions, tgt_mesh=mesh_gt, src_path = "data/mesh/model_mesh0.gif", tgt_path = "data/mesh/model_mesh_t0.gif" )
            if step == ids[1]:
                if args.type == "vox":
                    render_vox(predictions.squeeze(1), mesh_tgt=mesh_gt, src_path = "data/edge/model_vox1.gif", tgt_path = "data/edge/model_vox_t1.gif" )
                elif args.type == "point":
                    render_cloud(predictions.squeeze(1), tgt_cloud=pointclouds_tgt, src_path = "data/point/model_cloud1.gif", tgt_path = "data/point/model_cloud_t1.gif", radius = 0.015)
                elif args.type == "mesh":
                    render_mesh(predictions, tgt_mesh=mesh_gt, src_path = "data/mesh/model_mesh1.gif", tgt_path = "data/mesh/model_mesh_t1.gif" )
            if step == ids[2]:
                if args.type == "vox":
                    render_vox(predictions.squeeze(1), mesh_tgt=mesh_gt, src_path = "data/edge/model_vox2.gif", tgt_path = "data/edge/model_vox_t2.gif" )
                elif args.type == "point":
                    render_cloud(predictions.squeeze(1), tgt_cloud=pointclouds_tgt, src_path = "data/point/model_cloud2.gif", tgt_path = "data/point/model_cloud_t2.gif", radius = 0.015)
                elif args.type == "mesh":
                    render_mesh(predictions, tgt_mesh=mesh_gt, src_path = "data/mesh/model_mesh2.gif", tgt_path = "data/mesh/model_mesh_t2.gif" )

            f1_05 = metrics['F1@0.050000']
            print("[%4d/%4d]; F1@0.05: %.3f" % (step, max_iter, f1_05))
    print('Done!')

def visualize_inputs(args):

    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        if step>ids[2]: break
        feed_dict = next(eval_loader)
        if step in ids:
            images_gt = feed_dict['images'].squeeze(0)
            plt.imsave('submissions/gt_image_{}.png'.format(step), images_gt.cpu().numpy())
    print('Done!')
    
def evaluate_in_train(args, model, eval_loader):
    
    model.eval()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []
    
    # print("Starting evaluating !")
    max_iter = len(eval_loader)
    with trange(max_iter) as tbatches: 
        for step in tbatches:
            tbatches.set_description(f"Evaluate: ")
            feed_dict = next(eval_loader)
            images_gt, mesh_gt = preprocess(feed_dict, args)

            predictions = model(images_gt, args)
            if args.type == "vox":
                predictions = predictions.permute(0,1,4,3,2)

            metrics = evaluate(predictions, mesh_gt, thresholds, args)
            if metrics is None:
                tbatches.set_postfix(f1=666, avg_f1=666)
                print("WARNING: empty mesh found for evaluation ", step)
                break
            
            if args.vis:
                if step == 1:
                    if args.type == "vox":
                        render_vox(predictions.squeeze(1), src_path = "submissions/model_vox0.gif")
                    elif args.type == "point":
                        render_cloud(predictions.squeeze(1), src_path = "submissions/model_cloud0.gif", radius = 0.015)
                    elif args.type == "mesh":
                        render_mesh(predictions.squeeze(1), src_path = "submissions/model_mesh0.gif")
                if step == 2:
                    if args.type == "vox":
                        render_vox(predictions.squeeze(1), src_path = "submissions/model_vox1.gif")
                    elif args.type == "point":
                        render_cloud(predictions.squeeze(1), src_path = "submissions/model_cloud1.gif", radius = 0.015)
                    elif args.type == "mesh":
                        render_mesh(predictions.squeeze(1), src_path = "submissions/model_mesh1.gif")
                if step == 5:
                    if args.type == "vox":
                        render_vox(predictions.squeeze(1), src_path = "submissions/model_vox2.gif")
                    elif args.type == "point":
                        render_cloud(predictions.squeeze(1), src_path = "submissions/model_cloud2.gif", radius = 0.015)
                    elif args.type == "mesh":
                        render_mesh(predictions.squeeze(1), src_path = "submissions/model_mesh2.gif")

            f1_05 = metrics['F1@0.050000']
            avg_f1_score_05.append(f1_05)

            avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
            avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
            avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

            tbatches.set_postfix(f1=f1_05, avg_f1=torch.tensor(avg_f1_score_05).mean())
            # print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    
    avg_f1_score = torch.stack(avg_f1_score).mean(0)
    save_plot(thresholds, avg_f1_score,  args)
    return avg_f1_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.vis: visualize_model(args)
    else: evaluate_model(args)
    # visualize_inputs(args)
