import os
import warnings

import hydra
import numpy as np
import torch
import tqdm
import imageio

from omegaconf import DictConfig
from PIL import Image
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform
)
import matplotlib.pyplot as plt

from implicit import volume_dict
from sampler import sampler_dict
from renderer import renderer_dict
from render_functions import render_points
from ray_utils import (
    sample_images_at_xy,
    get_pixels_from_image,
    get_random_pixels_from_image,
    get_rays_from_pixels
)
from data_utils import (
    dataset_from_config,
    create_surround_cameras,
    vis_grid,
    vis_rays,
)
from dataset import (
    get_nerf_datasets,
    trivial_collate,
)


# Model class containing:
#   1) Implicit volume defining the scene
#   2) Sampling scheme which generates sample points along rays
#   3) Renderer which can render an implicit volume given a sampling scheme

class Model(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        # Get implicit function from config
        self.implicit_fn = volume_dict[cfg.implicit_function.type](
            cfg.implicit_function
        )

        # Point sampling (raymarching) scheme
        self.sampler = sampler_dict[cfg.sampler.type](
            cfg.sampler
        )

        # Initialize volume renderer
        self.renderer = renderer_dict[cfg.renderer.type](
            cfg.renderer
        )
    
    def forward(
        self,
        ray_bundle
    ):
        # Call renderer with
        #  a) Implicit volume
        #  b) Sampling routine

        return self.renderer(
            self.sampler,
            self.implicit_fn,
            ray_bundle
        )


def render_images(
    model,
    cameras,
    image_size,
    save=False,
    file_prefix=''
):
    all_images = []
    device = list(model.parameters())[0].device

    for cam_idx, camera in enumerate(cameras):
        print(f'Rendering image {cam_idx}')

        torch.cuda.empty_cache()
        camera = camera.to(device)
        xy_grid = get_pixels_from_image(image_size, camera) # TODO (1.3): implement in ray_utils.py
        ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera) # TODO (1.3): implement in ray_utils.py
        del camera
        torch.cuda.empty_cache()
        # TODO (1.3): Visualize xy grid using vis_grid
        if cam_idx == 0 and file_prefix == '':
            xy_vis = vis_grid(xy_grid, image_size)
            plt.imsave(f'images/xy_vis_{cam_idx}.png',xy_vis)
        del xy_grid
        torch.cuda.empty_cache()
        # TODO (1.3): Visualize rays using vis_rays
        if cam_idx == 0 and file_prefix == '':
            rays = vis_rays(ray_bundle, image_size)
            plt.imsave(f'images/rays_{cam_idx}.png', rays)
        
        # TODO (1.4): Implement point sampling along rays in sampler.py
        model.sampler.forward(ray_bundle)

        # TODO (1.4): Visualize sample points as point cloud
        if cam_idx == 0 and file_prefix == '':
            render_points(f'images/point_samp_{cam_idx}.png', ray_bundle.sample_points.view(-1,3).unsqueeze(0), image_size=256, color=[0.7, 0.7, 1], device=device)

        # TODO (1.5): Implement rendering in renderer.py
        out = model(ray_bundle)
        del ray_bundle
        torch.cuda.empty_cache()

        # Return rendered features (colors)
        image = np.array(
            out['feature'].view(
                image_size[1], image_size[0], 3
            ).detach().cpu()
        )
        all_images.append(image)

        # TODO (1.5): Visualize depth
        if cam_idx == 2 and file_prefix == '':
            depth_image = np.array(out['depth'].view(image_size[1], image_size[0]).detach().cpu())
            plt.imsave(f'images/depth_{cam_idx}.png',  depth_image)

        # Save
        if save:
            plt.imsave(
                f'{file_prefix}_{cam_idx}.png',
                image
            )
    
    return all_images


def render(
    cfg,
):
    # Create model
    model = Model(cfg)
    model = model.cuda(); model.eval()

    # Render spiral
    cameras = create_surround_cameras(3.0, n_poses=20)
    all_images = render_images(
        model, cameras, cfg.data.image_size
    )
    imageio.mimsave('images/part_1.gif', [np.uint8(im * 255) for im in all_images])


def train(
    cfg
):
    # Create model
    model = Model(cfg)
    model = model.cuda(); model.train()

    # Create dataset 
    train_dataset = dataset_from_config(cfg.data)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )
    image_size = cfg.data.image_size

    # Create optimizer 
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr
    )

    # Render images before training
    cameras = [item['camera'] for item in train_dataset]
    render_images(
        model, cameras, image_size,
        save=True, file_prefix='images/part_2_before_training'
    )

    # Train
    t_range = tqdm.tqdm(range(cfg.training.num_epochs))

    for epoch in t_range:
        for iteration, batch in enumerate(train_dataloader):
            image, camera, camera_idx = batch[0].values()
            image = image.cuda()
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(cfg.training.batch_size, image_size, camera).cuda() # TODO (2.1): implement in ray_utils.py
            ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera)
            rgb_gt = sample_images_at_xy(image, xy_grid).cuda()

            # Run model forward
            out = model(ray_bundle)

            # TODO (2.2): Calculate loss
            loss = torch.nn.functional.mse_loss(out['feature'], rgb_gt)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch % 10) == 0:
            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {loss:.06f}')
            t_range.refresh()

    # Print center and side lengths
    print("Box center:", tuple(np.array(model.implicit_fn.sdf.center.data.detach().cpu()).tolist()[0]))
    print("Box side lengths:", tuple(np.array(model.implicit_fn.sdf.side_lengths.data.detach().cpu()).tolist()[0]))

    # Render images after training
    render_images(
        model, cameras, image_size,
        save=True, file_prefix='images/part_2_after_training'
    )
    all_images = render_images(
        model, create_surround_cameras(3.0, n_poses=20), image_size, file_prefix='part_2'
    )
    imageio.mimsave('images/part_2.gif', [np.uint8(im * 255) for im in all_images])


def create_model(cfg):
    # Create model
    model = Model(cfg)
    model.cuda(); model.train()

    # Load checkpoints
    optimizer_state_dict = None
    start_epoch = 0

    checkpoint_path = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.training.checkpoint_path
    )

    if len(cfg.training.checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if cfg.training.resume and os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint {checkpoint_path}.")
            loaded_data = torch.load(checkpoint_path)
            model.load_state_dict(loaded_data["model"])
            start_epoch = loaded_data["epoch"]

            print(f"   => resuming from epoch {start_epoch}.")
            optimizer_state_dict = loaded_data["optimizer"]

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    def lr_lambda(epoch):
        return cfg.training.lr_scheduler_gamma ** (
            epoch / cfg.training.lr_scheduler_step_size
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    return model, optimizer, lr_scheduler, start_epoch, checkpoint_path

def train_nerf(
    cfg
):
    # Create model
    model, optimizer, lr_scheduler, start_epoch, checkpoint_path = create_model(cfg)

    # Load the training/validation data.
    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=trivial_collate,
    )
    del train_dataset
    # Run the main training loop.
    for epoch in range(start_epoch, cfg.training.num_epochs):

        torch.cuda.empty_cache()
        t_range = tqdm.tqdm(enumerate(train_dataloader))

        for iteration, batch in t_range:
            image, camera, camera_idx = batch[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(
                cfg.training.batch_size, cfg.data.image_size, camera
            ).cuda()
            ray_bundle = get_rays_from_pixels(
                xy_grid, cfg.data.image_size, camera
            )
            rgb_gt = sample_images_at_xy(image, xy_grid).cuda()
            # Run model forward
            out = model(ray_bundle)
            # TODO (3.1): Calculate loss
            loss = torch.nn.functional.mse_loss(out['feature'], rgb_gt)

            # Take the training step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {loss:.06f}')
            t_range.refresh()

        # Adjust the learning rate.
        lr_scheduler.step()

        # Checkpoint.
        if (
            epoch % cfg.training.checkpoint_interval == 0
            and len(cfg.training.checkpoint_path) > 0
            and epoch > 0
        ):
            print(f"Storing checkpoint {checkpoint_path}.")

            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(data_to_store, checkpoint_path)

        # Render
        if (
            epoch % cfg.training.render_interval == 0
            and epoch > 0
        ):
            with torch.no_grad():
                test_images = render_images(
                    model, create_surround_cameras(4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0),
                    cfg.data.image_size, file_prefix='nerf'
                )
                imageio.mimsave('images/part_3.gif', [np.uint8(im * 255) for im in test_images])

def train_heir(cfg):
    # Create model
    cfg_c = cfg
    cfg_f = cfg
    cfg_f.sampler.n_pts_per_ray = cfg_f.sampler.n_pts_per_ray*cfg_f.sampler.n_pts_mult
    del cfg
    torch.cuda.empty_cache()

    model_c, optimizer_c, lr_scheduler_c, _, _ = create_model(cfg_c)
    model_f, optimizer_f, lr_scheduler_f, start_epoch, checkpoint_path = create_model(cfg_f)

    # Load the training/validation data.
    train_dataset_c, _ , _ = get_nerf_datasets(
        dataset_name=cfg_c.data.dataset_name,
        image_size=[cfg_c.data.image_size[1], cfg.data.image_size[0]],
    )

    train_dataloader_c = torch.utils.data.DataLoader(
        train_dataset_c,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=trivial_collate,
    )
    del train_dataset_c
    torch.cuda.empty_cache()

    train_dataset_f, _ , _ = get_nerf_datasets(
        dataset_name=cfg_f.data.dataset_name,
        image_size=[cfg_f.data.image_size[1], cfg.data.image_size[0]],
    )

    train_dataloader_f = torch.utils.data.DataLoader(
        train_dataset_f,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=trivial_collate,
    )
    del train_dataset_f
    torch.cuda.empty_cache()

    # Run the main training loop.
    for epoch in range(start_epoch, cfg_c.training.num_epochs):

        torch.cuda.empty_cache()
        t_range = tqdm.tqdm(enumerate(zip(train_dataloader_c, train_dataloader_f)))

        for iteration, (batch_c, batch_f) in t_range:

            image, camera, _ = batch_c[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(cfg_c.training.batch_size, cfg_c.data.image_size, camera).cuda()
            ray_bundle_c = get_rays_from_pixels(xy_grid, cfg_c.data.image_size, camera)
            rgb_gt_c = sample_images_at_xy(image, xy_grid).cuda()
            # Run model forward
            out_c = model_c(ray_bundle_c)

            image, camera, _ = batch_f[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(cfg_f.training.batch_size, cfg_f.data.image_size, camera).cuda()
            ray_bundle_f = get_rays_from_pixels(xy_grid, cfg_f.data.image_size, camera)
            rgb_gt_f = sample_images_at_xy(image, xy_grid).cuda()
            # Run model forward

            out_f = model_f(ray_bundle_f)
            # TODO (3.1): Calculate loss
            loss = torch.nn.functional.mse_loss(out_c['feature'], rgb_gt_c) + torch.nn.functional.mse_loss(out_f['feature'], rgb_gt_f)

            # Take the training step.
            optimizer_c.zero_grad()
            optimizer_f.zero_grad()
            loss.backward()
            optimizer_c.step()
            optimizer_f.step()

            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {loss:.06f}')
            t_range.refresh()

        # Adjust the learning rate.
        lr_scheduler_c.step()
        lr_scheduler_f.step()

        # Render
        if (
            epoch % cfg_f.training.render_interval == 0
            and epoch > 0
        ):
            with torch.no_grad():
                test_images = render_images(
                    model_f, create_surround_cameras(4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0),
                    cfg_f.data.image_size, file_prefix='nerf'
                )
                imageio.mimsave('images/part_4.gif', [np.uint8(im * 255) for im in test_images])


@hydra.main(config_path='./configs', config_name='sphere')
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    if cfg.type == 'render':
        render(cfg)
    elif cfg.type == 'train':
        train(cfg)
    elif cfg.type == 'train_nerf':
        train_nerf(cfg)
    elif cfg.type == 'train_heirarchy':
        train_heir(cfg)


if __name__ == "__main__":
    main()

