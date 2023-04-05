import numpy as np
import argparse

import torch
from models import seg_model, trans_seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg

from pytorch3d.transforms import Rotate

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--indices', type=str, default=None, help="specify index of the objects to visualize, seperate values with ,")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')

    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    parser.add_argument('--s_thresh', type=float, default=0.9, help='Lower bound for accuracy to be considered success')
    parser.add_argument('--f_thresh', type=float, default=0.7, help='Upper bound for accuracy to be considered failure')


    parser.add_argument('--rotate', type=float, default=None, help='Rotates input about x axis by value if given')
    parser.add_argument('--transform', action='store_true', help='Use flag if evaluating transform model')


    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    if args.transform:
        model = trans_seg_model(args.num_seg_class)
        model_path = './checkpoints/trans_seg/{}.pt'.format(args.load_checkpoint)
    else:
        model = seg_model(args.num_seg_class)
        model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    
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
    test_labels = torch.from_numpy((np.load(args.test_label))[:,ind]).to(args.device)
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
        pred_label = torch.argmax(pred_label, 2)
        pred_labels.append(pred_label)
    
    pred_labels = torch.cat(pred_labels)

    total_test_accuracy = pred_labels.eq(test_labels.data).cpu().sum().item() / (test_labels.reshape((-1,1)).size()[0])
    
    if args.indices == None: 
        s_class = []
        s_ind = []
        f_class = []
        f_ind = []
        for i, items in enumerate(zip(test_data, test_labels, pred_labels)):
            data, test_label, pred_label = items
            test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
            
            if test_accuracy>args.s_thresh and len(s_class) < 3:    
                viz_seg(data, test_label, "{}/seg_s_gt_{}.gif".format(args.output_dir, len(s_class)), args)
                viz_seg(data, pred_label, "{}/seg_s_pred_{}.gif".format(args.output_dir, len(s_class)), args)
                s_ind.append(i)
                s_class.append(test_accuracy)
            if test_accuracy<args.f_thresh and len(f_class) < 3:
                viz_seg(data, test_label, "{}/seg_f_gt_{}.gif".format(args.output_dir, len(f_class)), args)
                viz_seg(data, pred_label, "{}/seg_f_pred_{}.gif".format(args.output_dir, len(f_class)), args)
                f_ind.append(i)
                f_class.append(test_accuracy)
        print("Accuracies of success classes: ", s_class)
        print("Accuracies of failure classes: ", f_class)
        print("S indices: ", s_ind)
        print("F indices: ", f_ind)
    else:
        args.indices = args.indices.split(',') 
        accuracies = []
        for i in args.indices:
            i = int(i)
            data = test_data[i]
            test_label = test_labels[i]
            pred_label = pred_labels[i]
            test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
            viz_seg(data, pred_label, "{}/seg_pred_{}_{}.gif".format(args.output_dir, args.exp_name, i), args)
            accuracies.append(test_accuracy)
        print("Accuracies of examples: ", accuracies)
    print ("test accuracy: {}".format(total_test_accuracy))

    # # Visualize Segmentation Result (Pred VS Ground Truth)
    # viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    # viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)
