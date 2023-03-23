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
To reproduce results, uncomment line 125 in dataset.py and change index to desired number of views then run:

```bash
python -m a4.main --config-name=volsdf
```

Be careful as this will overwrite `part_3_geometry.gif` and `part_3.gif`. 

### 5.3 Alternate SDF to Density Conversions (10 pts)

You can run the code for part 5.3 with:

```bash
python -m a4.main --config-name=volsdf_naive
```

This will save `part_5_3_geometry.gif` and `part_5_3.gif`.