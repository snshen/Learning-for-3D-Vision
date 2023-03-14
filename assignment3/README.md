Assignment 3
===================================

##  1. Differentiable Volume Rendering

You can run the code for part 1 with:

```bash
python main.py --config-name=box
```
This command should automatically generate the outputs of grid/ray visualization, visualization of the point samples from the first camera, spiral rendering of the box an associated depth image. These results will be written out to `images/part_1.gif`. 

##  2. Optimizing a basic implicit volume

You can run the code for part 2 with:

```bash
python main.py --config-name=train_box
```

This command should optimize the position and side lengths of a box, print out the center of the box and the side lengths of the box after training, and provide a spiral rendering of the optimized volume in `images/part_2.gif`.

##  3. Optimizing a Neural Radiance Field (NeRF) (30 points)

You can run the code for part 3 with:

```bash
python main.py --config-name=nerf_lego
```

This command should produce a spiral rendering of the bulldozer which will be written to `images/part_3.gif`. 

##  4. NeRF Extras (***Choose at least one!*** More than one is extra credit)

###  4.1 View Dependence (10 pts)

I had completed this section while implementing part 3 since I was closely following the NeRF paper according to its supplementary material (Appendix A) which included view dependence, positional encoding, and a skip connection. Therefor, the steps for running this section is identical to part 3.

###  4.3 High Resolution Imagery (10 pts)

You can run the code for part 4.3 with:

```bash
python main.py --config-name=nerf_lego_highres
``` 