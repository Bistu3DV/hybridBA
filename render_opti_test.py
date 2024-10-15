#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, prefilter_voxel
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.pose_utils import get_tensor_from_camera

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    render_org_path = os.path.join(model_path, name, "ours_org_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_org_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        idx = idx
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        rendering = render(
            view, gaussians, pipeline, background, camera_pose=camera_pose
        )["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )

        torchvision.utils.save_image(
            rendering, os.path.join(render_org_path, "{0:05d}".format(idx) + ".png")
        )

def render_set_optimize(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    render_org_path = os.path.join(model_path, name, "ours_org_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_org_path, exist_ok=True)

    gaussians.eval()
    gaussians._anchor.requires_grad_(False)
    gaussians._offset.requires_grad_(False)
    gaussians._anchor_feat.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # num_iter = 200
        num_iter = args.optim_test_pose_iter

        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))

        rendering_org = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, camera_pose=camera_pose)["render"]
        rendering_org = torch.clamp(rendering_org, 0.0, 1.0)

        torchvision.utils.save_image(
            rendering_org, os.path.join(render_org_path, "{0:05d}".format(idx) + ".png")
        )

        
        camera_tensor_T = camera_pose[-3:].requires_grad_()
        camera_tensor_q = camera_pose[:4].requires_grad_()
        pose_optimizer = torch.optim.Adam(
            [
                {
                    "params": [camera_tensor_T],
                    "lr": 0.0003,
                },
                {
                    "params": [camera_tensor_q],
                    "lr": 0.0001,
                },
            ]
        )


        progress_bar = tqdm(
            range(num_iter), desc=f"Tracking Time Step: {idx}", disable=True
        )

        # Keep track of best pose candidate
        candidate_q = camera_tensor_q.clone().detach()
        candidate_T = camera_tensor_T.clone().detach()
        org_pose = torch.cat([candidate_q, candidate_T])
        current_min_loss = float(1e20)
        gt = view.original_image[0:3, :, :]
        for iteration in range(num_iter):
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        
            rendering = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))["render"]
            rendering = torch.clamp(rendering, 0.0, 1.0)

            loss = torch.abs(gt - rendering).mean()
            if iteration%10==0:
                print(iteration, loss.item())
            loss.backward()

            with torch.no_grad():
                pose_optimizer.step()
                pose_optimizer.zero_grad(set_to_none=True)

                if iteration == 0:
                    initial_loss = loss

                if loss < current_min_loss:
                    current_min_loss = loss
                    candidate_q = camera_tensor_q.clone().detach()
                    candidate_T = camera_tensor_T.clone().detach()

                progress_bar.update(1)

        camera_tensor_q = candidate_q
        camera_tensor_T = candidate_T

        progress_bar.close()
        opt_pose = torch.cat([camera_tensor_q, camera_tensor_T])
        print(opt_pose-org_pose)

        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        rendering_opt = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, camera_pose=opt_pose)["render"]
        rendering_opt = torch.clamp(rendering_opt, 0.0, 1.0)

        torchvision.utils.save_image(
            rendering_opt, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )
        

def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args,
):
    with torch.no_grad():
        # gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # if not skip_train:
    #     render_set(
    #         dataset.model_path,
    #         "train",
    #         scene.loaded_iter,
    #         scene.getTrainCameras(),
    #         gaussians,
    #         pipeline,
    #         background,
    #     )

    if not skip_test:
        render_set_optimize(
            dataset.model_path,
            "test",
            scene.loaded_iter,
            scene.getTestCameras(),
            gaussians,
            pipeline,
            background,
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--n_views", default=None, type=int)
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--optim_test_pose_iter", default=500, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args,
    )
