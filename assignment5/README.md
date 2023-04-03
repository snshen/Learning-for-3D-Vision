## Data Preparation
Download zip file (~2GB) from https://drive.google.com/file/d/1wXOgwM_rrEYJfelzuuCkRfMmR0J7vLq_/view?usp=sharing. Put the unzipped `data` folder under root directory. There are two folders (`cls` and `seg`) corresponding to two tasks, each of which contains `.npy` files for training and testing.

## Q1. Classification Model (40 points)
Run `python train.py --task cls` to train the model. Note that in my submission, I trained my model for 30 epochs.

Then run `python eval_cls.py` which should automatically print the test accuracy of the best model, generate visualization for a success prediction for each classes (chair, vase and lamp),  generate visualization for a failure prediction for each class, and indicate what indices of the test data was used for visualization.  

Note down the indices printed if you would like to revisualize these specific examples for Q3.

## Q2. Segmentation Model (40 points) 
Run `python train.py --task seg` to train the model. Note that in my submission, I trained my model for 250 epochs and the best model was saved at epoch 160.
 
Then run `python eval_seg.py` which should automatically print the test accuracy of the best model, generate segmentation visualization and metrics for a 3 success predictions, and generate segmentation visualization and metrics for a 3 failure predictions.  

Thresholds for definig success (default set above 0.9) and failure (default set below 0.7) can be changes with the `--s_thresh` and `--f_thresh` flag respectively.

Note down the indices printed if you would like to revisualize these specific examples for Q3.

## Q3. Robustness Analysis (20 points) 

### Experiment 1: Rotation Invariance

run `python3 eval_cls.py --rotate [RADIANS]  --indices [INDICES TO VISUALIZE]` for this section, add `--exp_name [UNIQUE PREFIX]` to add a prefix to the saves gif name.

### Experiment 2: Number of Points 


run `python3 eval_cls.py --rotate [RADIANS]  --indices [INDICES TO VISUALIZE]` for this section, add `--exp_name [UNIQUE PREFIX]` to add a prefix to the saves gif name.

Run `python eval_seg.py --num_points 100` for this section, add `--indices` flag to visualize specific examples.

Feel free to try other ways of probing the robustness. Each experiment is worth 10 points.

Deliverables: On your website, for each experiment

- Describe your procedure 
- For each task, report test accuracy and visualization on a few samples, in comparison with your results from Q1 & Q2.
- Provide some interpretation in a few sentences.

## Q4. Expressive architectures (10 points + 20 bonus points)
Instead of using a vanilla PointNet, improve the base model using one of [PointNet++](https://arxiv.org/abs/1706.02413), or [DGCNN](https://arxiv.org/abs/1801.07829), or [Point Transformers](https://arxiv.org/abs/2012.09164). Your implementation need not leverage all details of these models (e.g. you can use different levels of hierarchy), but should borrow the key design principles and should allow some improvement over the base PointNet model.

Deliverables: On your website, 

- Describe the model you have implemented.
- For each task, report the test accuracy of your best model, in comparison with your results from Q1 & Q2.
- Visualize results in comparison to ones obtained in the earlier parts.

Note that you need to implement **at least one** of the above locality methods. That will be of worth 10 points. Each extra implemented method will be of worth 10 bonus points each. 
