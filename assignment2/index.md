# 16-825 Assignment 2: Single View to 3D
number or late days used:
<img src="./data/two.png"  width="2%">

In this assignment, we go through the basics of rendering with PyTorch3D, explore 3D representations, and practice constructing simple geometry.

## 1. Exploring loss functions

### 1.1. Fitting a voxel grid (5 points)

Visualize the optimized voxel grid along-side the ground truth voxel grid using the tools learnt in previous section.

|Ground Truth|Optomized|
|:-:|:-:|
|![First Image](./data/fit/target_vox.gif)|![Second Image](./data/fit/source_vox.gif)|

### 1.2 Fitting a point cloud (10 points)

Visualize the optimized point cloud along-side the ground truth point cloud using the tools learnt in previous section.

|Ground Truth|Optomized|
|:-:|:-:|
|![First Image](./data/fit/target_cloud.gif)|![Second Image](./data/fit/source_cloud.gif)|

### 1.3 Fitting a mesh (5 points)

Visualize the optimized mesh along-side the ground truth mesh using the tools learnt in previous section.

|Ground Truth|Optomized|
|:-:|:-:|
|![First Image](./data/fit/target_mesh.gif)|![Second Image](./data/fit/source_mesh.gif)|

## 2. Reconstructing 3D from single view

### 2.1. Image to voxel grid (15 points)

Visuals of any three examples in the test set. For each example show the input RGB, render of the predicted 3D voxel grid and a render of the ground truth mesh.


### 2.2 Image to point cloud (15 points)

Visuals of any three examples in the test set. For each example show the input RGB, render of the predicted 3D point cloud and a render of the ground truth mesh.

### 2.3 Image to mesh (15 points)

Visuals of any three examples in the test set. For each example show the input RGB, render of the predicted mesh and a render of the ground truth mesh.

### 2.4 Quantitative comparisions(10 points)

Average test F1 score at 0.05 threshold for voxelgrid, pointcloud and the mesh network.

### 2.5 Analyse effects of hyperparms variations (10 points)

Analyse the results, by varying an hyperparameter of your choice. For example n_points or vox_size or w_chamfer or initial mesh(ico_sphere) etc. Try to be unique and conclusive in your analysis.

### 2.6 Interpret your model (15 points)

Simply seeing final predictions and numerical evaluations is not always insightful. Can you create some visualizations that help highlight what your learned model does? Be creative and think of what visualizations would help you gain insights. There is no `right' answer - although reading some papers to get inspiration might give you ideas.


## 3. (Extra Credit) Exploring some recent architectures.
