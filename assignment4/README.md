Assignment 4
===================================

Here are instructions on how to reproduce my results using the code

##  1. Sphere Tracing (30pts)

You can run the code for part 1 with:

```bash
python -m a4.main --config-name=torus
```

This should save `part_1.gif` in the `images' folder. 

##  2. Optimizing a Neural SDF (30pts)

You can run the code for part 2 with:
```bash
python -m a4.main --config-name=points
```

This should save save `part_2_input.gif` and `part_2.gif` in the `images` folder. 

##  3. VolSDF (20 pts)

You can run the code for part 3 with:

```bash
python -m a4.main --config-name=volsdf
```

This will save `part_3_geometry.gif` and `part_3.gif`. 

## 4. Phong Relighting (20 pts)

You can run the code for part 4 with:

```bash
python -m a4.main --config-name=phong
```

This will save `part_4_geometry.gif` and `part_4.gif`.

## 5. Neural Surface Extras (CHOOSE ONE! More than one is extra credit)

### 5.1. Render a Large Scene with Sphere Tracing (10 pts)

You can run the code for part 5.3 with:

```bash
python -m a4.main --config-name=large
```

This should save `part_1.gif` in the `images' folder. 

### 5.2 Fewer Training Views (10 pts)
In Q3, we relied on 100 training views for a single scene. A benefit of using Surface representations, however, is that the geometry is better regularized and can in principle be inferred from fewer views. Experiment with using fewer training views (say 20) -- you can do this by changing [train_idx in data laoder](https://github.com/learning3d/assignment3/blob/main/dataset.py#L123) to use a smaller random subset of indices). You should also compare the VolSDF solution to a NeRF solution learned using similar views.

### 5.3 Alternate SDF to Density Conversions (10 pts)

You can run the code for part 5.3 with:

```bash
python -m a4.main --config-name=volsdf_naive
```

This will save `part_5_3_geometry.gif` and `part_5_3.gif`.