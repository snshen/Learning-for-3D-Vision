import numpy as np
import argparse

from torch.utils.data import DataLoader

import torch
from models import cls_model, trans_cls_model
from utils import create_dir, viz_cloud

from pytorch3d.transforms import Rotate

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--indices', type=str, default=None, help="specify index of the objects to visualize, seperate values with ,")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')

    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')
    parser.add_argument('--rotate', type=float, default=None, help='Rotates input about x axis by value if given')

    parser.add_argument('--transform', action='store_true', help='Use flag if evaluating transform model')



    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    if args.transform:
        model = trans_cls_model()
        model_path = './checkpoints/trans_cls/{}.pt'.format(args.load_checkpoint)
    else:
        model = cls_model()
        model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    
    # Load Model Checkpoint
    
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    model.to(args.device)
    print ("successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:]).to(args.device)
    test_labels = torch.from_numpy(np.load(args.test_label)).to(args.device)
    torch.cuda.empty_cache()

    if args.rotate is not None:
        R0=torch.tensor([[1., 0., 0.], [0., float(np.cos(args.rotate)), float(np.sin(args.rotate))], [0., float(-np.sin(args.rotate)), float(np.cos(args.rotate))]]).unsqueeze(0)
        R0=torch.tile(R0, (test_data.shape[0], 1, 1)).to(args.device)
        trans = Rotate(R0)
        test_data = trans.transform_points(test_data)

    # ------ TO DO: Make Prediction ------
    data_loader = torch.split(test_data, args.batch_size)
    label_loader = torch.split(test_labels, args.batch_size)
    test_accuracy = 0
    pred_labels = []


    for data, label in zip(data_loader, label_loader):
        pred_label =  model(data)
        pred_label = torch.argmax(pred_label, 1)
        pred_labels.append(pred_label)

    # Compute Accuracy
    pred_labels = torch.cat(pred_labels)
    test_accuracy = pred_labels.eq(test_labels.data).cpu().sum().item() / (test_labels.size()[0])
    if args.indices == None: 
        s_class = []
        s_ind = []
        f_class = []
        f_ind = []
        for i, items in enumerate(zip(test_data, test_labels, pred_labels)):
            cloud, test_label, label = items
            cloud = cloud.unsqueeze(0)
            if test_label==label and test_label not in s_class:
                src_path  = "{}/cls_s_{}_{}.gif".format(args.output_dir, int(test_label), label)
                viz_cloud(cloud, src_path = src_path)
                s_ind.append(i)
                s_class.append(test_label)
            if test_label!=label and test_label not in f_class:
                src_path  = "{}/cls_f_{}_{}.gif".format(args.output_dir, int(test_label), label)
                viz_cloud(cloud, src_path = src_path)
                f_ind.append(i)
                f_class.append(test_label)
        print("S indices: ", s_ind)
        print("F indices: ", f_ind)
    else:
        args.indices = args.indices.split(',') 
        for i in args.indices:
            i = int(i)
            cloud = test_data[i].unsqueeze(0)
            test_label = test_labels[i] 
            pred_label = pred_labels[i]
            src_path  = "{}/cls_{}_{}_{}.gif".format(args.output_dir, args.exp_name, int(test_label), pred_label)
            viz_cloud(cloud, src_path = src_path)

    print ("test accuracy: {}".format(test_accuracy))

