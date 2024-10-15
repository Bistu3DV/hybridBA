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
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding

## add by yxp
from torch_scatter import scatter_min, segment_coo, scatter_mean
from inplace_abn import InPlaceABN
from tqdm import tqdm
from collections import OrderedDict
import torch.nn.functional as F
import gc

from scipy.spatial.transform import Rotation as R
from utils.pose_utils import rotation2quad, get_tensor_from_camera
from utils.graphics_utils import getWorld2View2

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.

    Parameters:
    - quaternion: A tensor of shape (..., 4) representing quaternions.

    Returns:
    - A tensor of shape (..., 3, 3) representing rotation matrices.
    """
    # Ensure quaternion is of float type for computation
    quaternion = quaternion.float()

    # Normalize the quaternion to unit length
    quaternion = quaternion / quaternion.norm(p=2, dim=-1, keepdim=True)

    # Extract components
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]

    # Compute rotation matrix components
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    # Assemble the rotation matrix
    R = torch.stack([
        torch.stack([1 - 2 * (yy + zz),     2 * (xy - zw),     2 * (xz + yw)], dim=-1),
        torch.stack([    2 * (xy + zw), 1 - 2 * (xx + zz),     2 * (yz - xw)], dim=-1),
        torch.stack([    2 * (xz - yw),     2 * (yz + xw), 1 - 2 * (xx + yy)], dim=-1)
    ], dim=-2)

    return R

###################################  feature net  ######################################
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)


    def forward(self, x):
        return self.bn(self.conv(x))

class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, intermediate=False, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.intermediate = intermediate

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        B, V, _, H, W = x.shape
        x = x.reshape(B * V, 3, H, W)

        if self.intermediate:
            x1 = self.conv0(x)  # (B, 8, H, W)
            x2 = self.conv1(x1)  # (B, 16, H//2, W//2)
            x3 = self.conv2(x2)  # (B, 32, H//4, W//4)
            x3 = self.toplayer(x3)  # (B, 32, H//4, W//4)

            return [x, x1, x2, x3]
        else:
            # x: (B, 3, H, W)
            x = self.conv0(x) # (B, 8, H, W)
            x = self.conv1(x) # (B, 16, H//2, W//2)
            x = self.conv2(x) # (B, 32, H//4, W//4)
            x = self.toplayer(x) # (B, 32, H//4, W//4)

            return [x]
    
    def load_networks(self, epoch):
        for name, net in zip(self.model_names, self.get_networks()):
            print('loading', name)
            assert isinstance(name, str)
            load_filename = '{}_net_{}.pth'.format(epoch, name)
            load_path = os.path.join(self.opt.resume_dir, load_filename)

            if not os.path.isfile(load_path):
                print('cannot load', load_path)
                continue

            state_dict = torch.load(load_path, map_location=self.device)
            if isinstance(net, nn.DataParallel):
                net = net.module

            net.load_state_dict(state_dict, strict=False)

    def setup(self, opt):
        '''Creates schedulers if train, Load and print networks if resume'''
        if self.is_train:
            self.schedulers = [
                get_scheduler(optim, opt) for optim in self.optimizers
            ]
        if not self.is_train or opt.resume_dir:
            print("opt.resume_iter!!!!!!!!!", opt.resume_iter)
            self.load_networks(opt.resume_iter)
        self.print_networks(opt.verbose)

    # from kornia.utils import create_meshgrid
    def homo_warp_nongrid(self, c2w, w2c, intrinsic, ref_cam_xyz, HD, WD, filter=True, **kwargs):
        # src_grid: B, 3, D*H*W   xyz
        # import pdb; pdb.set_trace()
        B, M, _ = ref_cam_xyz.shape
        if w2c is not None:
            src_cam_xyz = torch.cat([ref_cam_xyz, torch.ones_like(ref_cam_xyz[:,:,0:1])], dim=-1) @ c2w.transpose(1,2) @ w2c.transpose(1,2)
        else:
            src_cam_xyz = ref_cam_xyz
        src_grid = ((src_cam_xyz[..., :3] / src_cam_xyz[..., 2:3]) @ intrinsic.transpose(1,2))[...,:2]

        mask = torch.prod(torch.cat([torch.ge(src_grid, torch.zeros([1,1,2], device=src_grid.device)), torch.le(src_grid, torch.tensor([[[WD-1,HD-1]]], device=src_grid.device))],dim=-1), dim=-1, keepdim=True, dtype=torch.int8) > 0

        src_grid = src_grid.to(torch.float32)  # grid xy
        hard_id_xy = torch.ceil(src_grid[:,:,:])
        src_grid = torch.masked_select(src_grid, mask).reshape(B, -1, 2) if filter else src_grid

        src_grid[..., 0] = src_grid[..., 0] / ((WD - 1.0) / 2.0) - 1.0  # scale to -1~1
        src_grid[..., 1] = src_grid[..., 1] / ((HD - 1.0) / 2.0) - 1.0  # scale to -1~1
        return src_grid, mask, hard_id_xy

    def homo_warp_nongrid_occ(self, c2w, w2c, intrinsic, ref_cam_xyz, HD, WD, tolerate=0.1, scatter_cpu=True):
        # src_grid: B, 3, D*H*W   xyz
        B, M, _ = ref_cam_xyz.shape
        if w2c is not None:
            src_cam_xyz = torch.cat([ref_cam_xyz, torch.ones_like(ref_cam_xyz[:,:,0:1])], dim=-1) @ c2w.transpose(1,2) @ w2c.transpose(1,2)
        else:
            src_cam_xyz = ref_cam_xyz
        # print("src_cam_xyz",src_cam_xyz.shape, intrinsic.shape)
        src_grid = ((src_cam_xyz[..., :3] / src_cam_xyz[..., 2:3]) @ intrinsic.transpose(1,2))[...,:2]
        # print("src_pix_xy1", src_grid.shape, torch.min(src_grid,dim=-2)[0], torch.max(src_grid,dim=-2)[0])
        mask = torch.prod(torch.cat([torch.ge(src_grid, torch.zeros([1,1,2], device=src_grid.device)), torch.le(torch.ceil(src_grid), torch.tensor([[[WD-1,HD-1]]], device=src_grid.device))],dim=-1), dim=-1, keepdim=True, dtype=torch.int8) > 0
        src_grid = torch.masked_select(src_grid, mask).reshape(B, -1, 2)
        cam_z = torch.masked_select(src_cam_xyz[:,:,2], mask[...,0]).reshape(B, -1)

        src_grid = src_grid.to(torch.float32)  # grid xy
        # print("HD, WD", HD, WD) 512 640
        src_grid_x = src_grid[..., 0:1] / ((WD - 1.0) / 2.0) - 1.0  # scale to -1~1
        src_grid_y = src_grid[..., 1:2] / ((HD - 1.0) / 2.0) - 1.0  # scale to -1~1
        # hard_id_xy: 1, 307405, 2

        hard_id_xy = torch.ceil(src_grid[:,:,:])
        # print("hard_id_xy", hard_id_xy.shape)
        index = (hard_id_xy[...,0] * HD + hard_id_xy[...,1]).long() # 1, 307405
        # print("index", index.shape, torch.min(index), torch.max(index))
        min_depth, argmin = scatter_min(cam_z[:,:].cpu() if scatter_cpu else cam_z[:,:], index[:,:].cpu() if scatter_cpu else index[:,:], dim=1)
        # print("argmin", min_depth.shape, min_depth, argmin.shape)

        queried_depth = min_depth.to(ref_cam_xyz.device)[:, index[0,...]] if scatter_cpu else min_depth[:, index[0,...]]
        block_mask = (cam_z <= (queried_depth + tolerate))
        # print("mask", mask.shape, torch.sum(mask), block_mask.shape, torch.sum(block_mask))
        mask[mask.clone()] = block_mask
        # print("mask", mask.shape, torch.sum(mask), block_mask.shape, torch.sum(block_mask))
        # print("src_grid_x", src_grid_x.shape)
        src_grid_x = torch.masked_select(src_grid_x, block_mask[..., None]).reshape(B, -1, 1)
        src_grid_y = torch.masked_select(src_grid_y, block_mask[..., None]).reshape(B, -1, 1)
        # print("src_grid_x", src_grid_x.shape, src_grid_y.shape, mask.shape)
        return torch.cat([src_grid_x, src_grid_y], dim=-1), mask, hard_id_xy

    def extract_from_2d_grid(self, src_feat, src_grid, mask):
        B, M, _ = src_grid.shape
        warped_src_feat = F.grid_sample(src_feat, src_grid[:, None, ...], mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, C, D, H*W)
        warped_src_feat = warped_src_feat.permute(0,2,3,1).view(B, M, src_feat.shape[1]).cuda() # 1, 224874, 3
        if mask is not None:
            B, N, _ = mask.shape
            full_src_feat = torch.zeros([B, N, src_feat.shape[1]], device=warped_src_feat.device, dtype=warped_src_feat.dtype)
            full_src_feat[0, mask[0,:,0], :] = warped_src_feat
            warped_src_feat = full_src_feat
        return warped_src_feat

    def extract_2d(self, img_feats, view_ids, layer_ids, intrinsics, c2ws, w2cs, cam_xyz, HD, WD, cam_vid=0):
        out_feats = []
        colors = []
        depth_occ = 0
        for vid in view_ids:
            w2c = w2cs[:,vid,...] if vid != cam_vid else None
            warp = self.homo_warp_nongrid_occ if depth_occ > 0 else self.homo_warp_nongrid
            ## modified by yxp
            src_grid, mask, hard_id_xy = warp(c2ws[:,cam_vid,...], w2c, intrinsics[:,vid,...], cam_xyz, HD, WD, tolerate=0.1)

            warped_feats = []
            for lid in layer_ids:
                img_feat = img_feats[lid] # 3, 32, 128, 160
                warped_src_feat = self.extract_from_2d_grid(img_feat[vid:vid+1, ...], src_grid, mask)
                
                if lid == 0:
                    colors.append(warped_src_feat)
                else:
                    warped_feats.append(warped_src_feat)
                
            warped_feats = torch.cat(warped_feats, dim=-1)
            out_feats.append(warped_feats)
        out_feats = torch.cat(out_feats, dim=-1)
        colors = torch.cat(colors, dim=-1) if len(colors) > 0 else None
        return out_feats, colors

    def query_embedding(self, HDWD, cam_xyz, photometric_confidence, img_feats, \
                        c2ws, w2cs, intrinsics, cam_vid, pointdir_w=False):
        # import pdb; pdb.set_trace()
        # img_feats = self.forward(x)
        HD, WD = HDWD
        points_embedding = []
        points_dirs = None
        points_conf = None
        points_colors = None
        # for feat_str in getattr(self.args, feature_str_lst[cam_vid]):
        for feat_str in ['imgfeat_0_0123', 'dir_0', 'point_conf']:
            if feat_str.startswith("imgfeat"):
                _, view_ids, layer_ids = feat_str.split("_")
                view_ids = [int(a) for a in list(view_ids)]
                layer_ids = [int(a) for a in list(layer_ids)]
                twoD_feats_tmp, points_colors_tmp = self.extract_2d(img_feats, view_ids, layer_ids, intrinsics, \
                                                             c2ws, w2cs, cam_xyz, HD, WD, cam_vid=cam_vid)
                twoD_feats = twoD_feats_tmp.clone()
                points_colors = points_colors_tmp.clone()
                points_embedding.append(twoD_feats)
            elif feat_str.startswith("dir"):
                _, view_ids = feat_str.split("_")
                view_ids = torch.as_tensor([int(a) for a in list(view_ids)], dtype=torch.int64, device=cam_xyz.device)
                cam_pos_world = c2ws[:, view_ids, :, 3] # B, V, 4
                cam_trans = w2cs[:, cam_vid, ...] # B, 4, 4
                cam_pos_cam = (cam_pos_world @ cam_trans.transpose(1, 2))[...,:3] # B, V, 4
                points_dirs = cam_xyz[:,:, None, :] - cam_pos_cam[:, None, :, :] # B, N, V, 3 in current cam coord
                points_dirs = points_dirs / (torch.linalg.norm(points_dirs, dim=-1, keepdims=True) + 1e-6)  # B, N, V, 3
                points_dirs = points_dirs.view(cam_xyz.shape[0], -1, 3) @ c2ws[:, cam_vid, :3, :3].transpose(1, 2)
                ## fix, by yxp
                # if not pointdir_w:
                #     points_dirs = points_dirs @ c2ws[:, self.args.ref_vid, :3, :3].transpose(1, 2) # in ref cam coord
                # print("points_dirs", points_dirs.shape)
                points_dirs = points_dirs.view(cam_xyz.shape[0], cam_xyz.shape[1], -1)
            elif feat_str.startswith("point_conf"):
                if photometric_confidence is None:
                    photometric_confidence = torch.ones_like(points_embedding[0][...,0:1])
                points_conf = photometric_confidence
        points_embedding = torch.cat(points_embedding, dim=-1)
        
        # import pdb; pdb.set_trace()
        del twoD_feats_tmp
        del points_colors_tmp
        for img_f in img_feats:
            del img_f
        gc.collect()
        torch.cuda.empty_cache()
        # fix, by yxp
        # if self.args.shading_feature_mlp_layer0 > 0:
        #     points_embedding = self.premlp(torch.cat([points_embedding, points_colors, points_dirs, points_conf], dim=-1))
        return points_embedding, points_colors, points_dirs, points_conf



class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        ## add by yxp
        # self.vision_model = models.vgg16(pretrained=True).cuda()
        self.vision_model = FeatureNet(intermediate=True).cuda()
        ## fixme, load init model parameters.
        
        # self.vision_model.setup()
        load_path = '/data/code0516/HybridNeuralRendering/checkpoints/init/dtu_dgt_d012_img0123_conf_agg2_32_dirclr20/best_net_mvs.pth'
        state_dict = torch.load(load_path)
        new_dict = OrderedDict() 
        for key,value in state_dict.items():
            key = key.replace('FeatureNet.', '')
            new_dict[key] = value

        self.vision_model.load_state_dict(new_dict, strict=False)

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim*2+3+self.opacity_dist_dim, feat_dim*2),
            nn.ReLU(True),
            nn.Linear(feat_dim*2, n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim*2+3+self.cov_dist_dim, feat_dim*2),
            nn.ReLU(True),
            nn.Linear(feat_dim*2, 7*self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim*2+3+self.color_dist_dim+self.appearance_dim, feat_dim*2),
            nn.ReLU(True),
            nn.Linear(feat_dim*2, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()
    

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.P,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._offset,
        self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.P) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def compute_relative_world_to_camera(self, R1, t1, R2, t2):
        # Create a row of zeros with a one at the end, for homogeneous coordinates
        zero_row = np.array([[0, 0, 0, 1]], dtype=np.float32)

        # Compute the inverse of the first extrinsic matrix
        E1_inv = np.hstack([R1.T, -R1.T @ t1.reshape(-1, 1)])  # Transpose and reshape for correct dimensions
        E1_inv = np.vstack([E1_inv, zero_row])  # Append the zero_row to make it a 4x4 matrix

        # Compute the second extrinsic matrix
        E2 = np.hstack([R2, -R2 @ t2.reshape(-1, 1)])  # No need to transpose R2
        E2 = np.vstack([E2, zero_row])  # Append the zero_row to make it a 4x4 matrix

        # Compute the relative transformation
        E_rel = E2 @ E1_inv

        return E_rel

    def init_RT_seq(self, cam_list):
        poses =[]
        for cam in cam_list[1.0]:
            p = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1)) # R T -> quat t
            poses.append(p)
        poses = torch.stack(poses)
        self.P = poses.cuda().requires_grad_(True)


    def get_RT(self, idx):
        pose = self.P[idx]
        return pose

    def get_RT_test(self, idx):
        pose = self.test_P[idx]
        return pose
    
    def voxelize_sample_v1(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        ## input: np.array 
        ## output: torch.tensor
        np.random.shuffle(data)
        # sparse_grid_idx, inv_idx = np.unique(np.floor(data / voxel_size), axis=0, return_inverse=True)
        data_tensor = torch.from_numpy(data)
        # inv_idx_tensor = torch.from_numpy(inv_idx)
        voxel_size_tensor = torch.tensor(voxel_size)
        sparse_grid_idx, inv_idx = torch.unique(torch.floor(data_tensor / voxel_size_tensor[None, ...]).to(torch.int32), dim=0, return_inverse=True)
        data_centroid = scatter_mean(data_tensor, inv_idx, dim=0)
        data_centroid_prop = data_centroid[inv_idx,:]
        data_residual = torch.norm(data_tensor - data_centroid_prop, dim=-1)
        print("data_residual", data_residual.shape)

        _, min_idx = scatter_min(data_residual, inv_idx, dim=0)
        print("min_idx", min_idx.shape)
        sampled_data = data_tensor[min_idx, :]
        return data_centroid, sparse_grid_idx, min_idx, sampled_data.cuda()

    def get_campos_ray(self, cam_list):
        def get_dtu_raydir(pixelcoords, intrinsic, rot, dir_norm):
            # rot is c2w
            ## pixelcoords: H x W x 2
            x = (pixelcoords[..., 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0]
            y = (pixelcoords[..., 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1]
            z = np.ones_like(x)
            dirs = np.stack([x, y, z], axis=-1)
            # dirs = np.sum(dirs[...,None,:] * rot[:,:], axis=-1) # h*w*1*3   x   3*3
            dirs = dirs @ rot[:,:].T #
            if dir_norm:
                # print("dirs",dirs-dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5))
                dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-5)
            # print("dirs", dirs.shape)

            return dirs
        
        # centerpixel = np.asarray(self.img_wh).astype(np.float32)[None, :] // 2
        camposes = []
        centerdirs = []
        # import pdb; pdb.set_trace()
        for cam_info in cam_list:
            centerpixel = np.asarray(cam_info.image.size).astype(np.float32)[None, :] // 2
            # c2w = np.loadtxt(os.path.join(self.data_dir, self.scan, "exported/pose", "{}.txt".format(id))).astype(np.float32)  #@ self.blender2opencv
            c2w = cam_info.extrinsic_c2w
            campos = c2w[:3, 3]
            camrot = c2w[:3, :3]
            cam_intrinsic = cam_info.intrinsic
            raydir = get_dtu_raydir(centerpixel, cam_intrinsic, camrot, True)
            camposes.append(campos)
            centerdirs.append(raydir)
        camposes=np.stack(camposes, axis=0)  # camNum, 3
        centerdirs=np.concatenate(centerdirs, axis=0)  # camNum, 3
        # print("camposes", camposes.shape, centerdirs.shape)
        return torch.as_tensor(camposes, device="cuda", dtype=torch.float32), torch.as_tensor(centerdirs, device="cuda", dtype=torch.float32)
        # return torch.as_tensor(camposes, dtype=torch.float32), torch.as_tensor(centerdirs, dtype=torch.float32)      

    # def nearest_view(self, campos, raydir, xyz):
    def nearest_view(self, xyz):
        cam_ind = torch.zeros([0, 1], device=self.campos.device, dtype=torch.long)
        step = 10000
        for i in range(0, len(xyz), step):
            dists = xyz[i:min(len(xyz), i + step), None, :] - self.campos[None, ...]  # N, M, 3
            dists_norm = torch.norm(dists, dim=-1)  # N, M
            dists_dir = dists / (dists_norm[..., None] + 1e-6)  # N, M, 3
            dists = dists_norm / 200 + (1.1 - torch.sum(dists_dir * self.camdir[None, :], dim=-1))  # N, M
            cam_ind = torch.cat([cam_ind, torch.argmin(dists, dim=1).view(-1, 1)], dim=0)  # N, 1
        return cam_ind
    
    def points2imgfeats(self, points_xyz, train_cameras):
        cam_ind = self.nearest_view(points_xyz)
        unique_cam_ind = torch.unique(cam_ind)
        print("unique_cam_ind", unique_cam_ind.shape)
        
        points_indx = []
        for i in range(len(unique_cam_ind)):
           points_indx.append(cam_ind[:, 0] == unique_cam_ind[i])

        points_xyz = [points_xyz[cam_ind[:, 0] == unique_cam_ind[i], :] for i in range(len(unique_cam_ind))]

        points_embedding_all = torch.zeros([1, 0, self.feat_dim], device="cuda", dtype=torch.float32)
        points_color_all = torch.zeros([1, 0, 3], device="cuda", dtype=torch.float32)
        points_dir_all = torch.zeros([1, 0, 3], device="cuda", dtype=torch.float32)
        points_conf_all = torch.zeros([1, 0, 1], device="cuda", dtype=torch.float32)
 
        print("extract points embeding & colors", )
        with torch.no_grad():
            for i in tqdm(range(len(unique_cam_ind))):
                
                id = unique_cam_ind[i]
                # batch = train_dataset.get_item(id, full_img=True)
                batch = train_cameras[id] ## add in senceinfo
                
                HDWD = [batch.height, batch.width]
                c2w = torch.from_numpy(batch.extrinsic_c2w).cuda().to(torch.float32)
                w2c = torch.inverse(c2w)
                intrinsic = torch.from_numpy(batch.intrinsic).cuda().to(torch.float32)[:,:3]
                # cam_xyz_all 252, 4
                cam_xyz_all = (torch.cat([points_xyz[i], torch.ones_like(points_xyz[i][..., -1:])], dim=-1) @ w2c.transpose(0, 1))[..., :3]
                
                if '000146.jpg' in batch.image_path:
                    import pdb; pdb.set_trace()
                try:
                    if batch.img_feats[0] is None:
                        img_feats = self.vision_model.forward(batch.image_tensor.cuda())
                        for idx, item in enumerate(img_feats):
                            batch.img_feats[idx] = item
                except:
                    import pdb; pdb.set_trace()
                #import pdb; pdb.set_trace()
                embedding, color, dir, conf = self.vision_model.query_embedding(HDWD, cam_xyz_all[None, ...], None, \
                                                                    batch.img_feats, c2w[None, None, ...], \
                                                                    w2c[None, None, ...], intrinsic[None, None, ...], \
                                                                    0, pointdir_w=True)
                    # conf = conf * opt.default_conf if opt.default_conf > 0 and opt.default_conf < 1.0 else conf
                points_embedding_all = torch.cat([points_embedding_all, embedding], dim=1)
                points_color_all = torch.cat([points_color_all, color], dim=1)
                points_dir_all = torch.cat([points_dir_all, dir], dim=1)
                points_conf_all = torch.cat([points_conf_all, conf], dim=1)
                    # visualizer.save_neural_points(id, cam_xyz_all, color, batch, save_ref=True)
            # import pdb; pdb.set_trace()
        
            # visualizer.save_neural_points("init", points_xyz_all, points_color_all, None, save_ref=load_points == 0)
        points_xyz = torch.cat(points_xyz, dim=0)
        return points_xyz, points_embedding_all, points_color_all, points_dir_all, points_conf_all, points_indx

    # def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
    # def create_from_pcd(self, sceneinfo : SceneInfo, spatial_lr_scale : float):
    def create_from_pcd(self, sceneinfo, spatial_lr_scale):
        pcd = sceneinfo.point_cloud
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        ## modified by yxp
        # points = self.voxelize_sample_v1(points, voxel_size=self.voxel_size)
        # center_point, sparse_grid_idx, sampled_pnt_idx, points_xyz = self.voxelize_sample(points, voxel_size=self.voxel_size)
        _, _, _, points_xyz = self.voxelize_sample(points, voxel_size=self.voxel_size)

        

        # import pdb; pdb.set_trace()
        ## fix, only for train
        self.campos, self.camdir = self.get_campos_ray(sceneinfo.train_cameras)

        points_xyz, p_feats, p_colors, p_dirs, p_conf, _ = self.points2imgfeats(points_xyz, sceneinfo.train_cameras)
        # cam_ind = self.nearest_view(campos, camdir, points_xyz)
        # unique_cam_ind = torch.unique(cam_ind)
        # print("unique_cam_ind", unique_cam_ind.shape)
        # points_xyz = [points_xyz[cam_ind[:, 0] == unique_cam_ind[i], :] for i in range(len(unique_cam_ind))]

        # points_embedding_all = torch.zeros([1, 0, self.feat_dim], device="cuda", dtype=torch.float32)
        # points_color_all = torch.zeros([1, 0, 3], device="cuda", dtype=torch.float32)
        # points_dir_all = torch.zeros([1, 0, 3], device="cuda", dtype=torch.float32)
        # points_conf_all = torch.zeros([1, 0, 1], device="cuda", dtype=torch.float32)
 
        # print("extract points embeding & colors", )
        # with torch.no_grad():
        #     for i in tqdm(range(len(unique_cam_ind))):
                
        #         id = unique_cam_ind[i]
        #         # batch = train_dataset.get_item(id, full_img=True)
        #         batch = sceneinfo.train_cameras[id] ## add in senceinfo
                
        #         HDWD = [batch.height, batch.width]
        #         c2w = torch.from_numpy(batch.extrinsic_c2w).cuda().to(torch.float32)
        #         w2c = torch.inverse(c2w)
        #         intrinsic = torch.from_numpy(batch.intrinsic).cuda().to(torch.float32)[:,:3]
        #         # cam_xyz_all 252, 4
        #         cam_xyz_all = (torch.cat([points_xyz[i], torch.ones_like(points_xyz[i][..., -1:])], dim=-1) @ w2c.transpose(0, 1))[..., :3]
                
        #         # import pdb; pdb.set_trace()
        #         embedding, color, dir, conf = self.vision_model.query_embedding(HDWD, cam_xyz_all[None, ...], None, \
        #                                                             batch.image_tensor.cuda(), c2w[None, None, ...], \
        #                                                             w2c[None, None, ...], intrinsic[None, None, ...], \
        #                                                             0, pointdir_w=True)
        #         # conf = conf * opt.default_conf if opt.default_conf > 0 and opt.default_conf < 1.0 else conf
        #         points_embedding_all = torch.cat([points_embedding_all, embedding], dim=1)
        #         points_color_all = torch.cat([points_color_all, color], dim=1)
        #         points_dir_all = torch.cat([points_dir_all, dir], dim=1)
        #         points_conf_all = torch.cat([points_conf_all, conf], dim=1)
        #         # visualizer.save_neural_points(id, cam_xyz_all, color, batch, save_ref=True)
        #     # import pdb; pdb.set_trace()
        
        #     # visualizer.save_neural_points("init", points_xyz_all, points_color_all, None, save_ref=load_points == 0)
        
        print("vis")

        # fused_point_cloud = torch.tensor(np.asarray(points_xyz)).float().cuda()
        fused_point_cloud = points_xyz
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # import pdb; pdb.set_trace()
        anchor_feat_merge = torch.cat([p_feats.squeeze(0), anchors_feat], dim=-1)
        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchor_feat_merge.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        l_cam = [{'params': [self.P],'lr': training_args.rotation_lr*0.1, "name": "pose"},]
        # l_cam = [{'params': [self.P],'lr': training_args.rotation_lr, "name": "pose"},]

        l += l_cam

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)
        self.cam_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr*0.1,
                                                    lr_final=training_args.rotation_lr*0.001,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=1000)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "pose":
                lr = self.cam_scheduler_args(iteration)
                param_group['lr'] = lr
            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name'] or \
                'pose' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

              
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name'] or \
                'pose' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask
        
        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    
    def anchor_growing(self, sceneinfo, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))
                
                # import pdb; pdb.set_trace()
                candidate_anchor, p_feats, p_colors, p_dirs, p_conf, p_indx = self.points2imgfeats(candidate_anchor, sceneinfo.train_cameras)
                # anchors_feat = torch.zeros((candidate_anchor.shape[0], self.feat_dim)).float().cuda()
                # new_feat = torch.cat([p_feats.squeeze(0), anchors_feat], dim=-1)
                
                # new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim*2])[candidate_mask]
                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self._anchor_feat.shape[-1]])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_feat = torch.cat([p_feats.squeeze(0), new_feat[:,self.feat_dim:]], dim=-1)

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
            

    def adjust_anchor(self, sceneinfo, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(sceneinfo, grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim*2+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim*2+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim*2+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
