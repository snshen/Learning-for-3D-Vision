## Data Preparation
Download zip file (~2GB) from https://drive.google.com/file/d/1wXOgwM_rrEYJfelzuuCkRfMmR0J7vLq_/view?usp=sharing. Put the unzipped `data` folder under root directory. There are two folders (`cls` and `seg`) corresponding to two tasks, each of which contains `.npy` files for training and testing.

## Q1. Classification Model (40 points)
Run `python train.py --task cls` to train the model. Note that in my submission, I trained my model for 30 epochs.

Then run `python eval_cls.py` which should automatically print the test accuracy of the best model, generate visualization for a success prediction for each classes (chair, vase and lamp),  generate visualization for a failure prediction for each class, and indicate what indices of the test data was used for visualization.  

Note down the indices printed if you would like to revisualize these specific examples for Q3. For my success examples I used [0, 617, 719] and for my failure examples I used [406, 618, 750].

## Q2. Segmentation Model (40 points) 
Run `python train.py --task seg` to train the model. Note that in my submission, I trained my model for 250 epochs and the best model was saved at epoch 160.
 
Then run `python eval_seg.py` which should automatically print the test accuracy of the best model, generate segmentation visualization and metrics for a 3 success predictions, and generate segmentation visualization and metrics for a 3 failure predictions.  

Thresholds for definig success (default set above 0.9) and failure (default set below 0.7) can be changes with the `--s_thresh` and `--f_thresh` flag respectively.

Note down the indices printed if you would like to revisualize these specific examples for Q3. For my success examples I used [0, 1, 2] and for my failure examples I used [26, 41, 61].

## Q3. Robustness Analysis (20 points) 

### Experiment 1: Rotation Invariance

run `python3 eval_cls.py --rotate [RADIANS]  --indices [INDICES TO VISUALIZE]` for this section, add `--exp_name [UNIQUE PREFIX]` to add a prefix to the saves gif name.

### Experiment 2: Number of Points 


run `python3 eval_cls.py --num_points [NUMBER OF POINTS]  --indices [INDICES TO VISUALIZE]` for this section, add `--exp_name [UNIQUE PREFIX]` to add a prefix to the saves gif name.


## Q4. Expressive architectures (10 points + 20 bonus points)
In this section, I improve the base model performance by implementing transformation blocks and utilizing skip connections in a similar way [Point Transformers](https://arxiv.org/abs/2012.09164). 

To train the classification and prediction models with the tranformer based modifications, use the flags `--task trans_cls` and `--task trans_seg` respectively. In order to evaluate these models include `--tranformer` when running the evaluation scripts.