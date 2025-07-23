import math
import os
import os.path as osp
import numpy as np
import cv2
import random
import torch
import torchvision.utils as vutils
import torchvision.transforms as tt
import torch.nn.functional as F
from PIL import Image


class Meta(object):

    def __init__(self, meta_aa_factor=1, meta_size=None, meta_path=None, device="cpu"):
        super().__init__()
        self.device = device
        self.meta_aa_factor = meta_aa_factor
        # c h w
        self.meta_size = meta_size
        assert meta_size is not None or meta_path is not None
        if meta_path is not None:
            meta = tt.ToTensor()(Image.open(meta_path))[None, 0:3, ...].to(self.device)
            meta = tt.Resize(meta_size[-2:])(meta)
            meta = tt.CenterCrop(meta_size[-2:])(meta)
        else:
            meta = torch.randint(size=meta_size, low=0, high=256, device=self.device)[None, ...] / 255.
        self.meta_mask = self.init_meta_mask()
        self.meta = meta * self.meta_mask

    def init_meta_mask(self, circle_radius=2):
        x = torch.linspace(-1, 1, self.meta_size[2], device=self.device)
        y = torch.linspace(-1, 1, self.meta_size[1], device=self.device) + 1
        y_t, x_t = torch.meshgrid([y, x], indexing='ij')
        eps = 0.005
        circle_constraint = torch.le((self.meta_size[2] / self.meta_size[1]) ** 2 * x_t * x_t + y_t * y_t,
                                     circle_radius ** 2 + eps)
        left_line_constraint = torch.ge(math.sqrt(4 - self.meta_size[2] ** 2 / self.meta_size[1] ** 2) * x_t + y_t,
                                        -eps)
        right_line_constraint = torch.le(math.sqrt(4 - self.meta_size[2] ** 2 / self.meta_size[1] ** 2) * x_t - y_t,
                                         eps)
        meta_mask = (circle_constraint * left_line_constraint * right_line_constraint).float()
        return meta_mask[None, None, ...].repeat(1, 3, 1, 1)

    def forward(self):
        return self.meta * self.meta_mask

    @torch.no_grad()
    def clip_colorset(self, cms, suffix):
        meta_clip = cms.forward(self.meta, suffix)
        self.meta = meta_clip * self.meta_mask
        return

    def _make_fractal_mask(self, top_radius, bottom_radius):
        x = torch.arange(-1, 1, 2.0 / self.meta_size[2]).to(self.device)
        y = torch.arange(-1, 1, 2.0 / self.meta_size[1]).to(self.device) + 1
        y_t, x_t = torch.meshgrid([y, x], indexing='ij')
        eps = 0.005
        circle_top_constraint = torch.ge((self.meta_size[2] / self.meta_size[1]) ** 2 * x_t * x_t + y_t * y_t,
                                         top_radius ** 2 + eps)
        circle_bottom_constraint = torch.le((self.meta_size[2] / self.meta_size[1]) ** 2 * x_t * x_t + y_t * y_t,
                                            bottom_radius ** 2 + eps)
        left_line_constraint = torch.ge(math.sqrt(4 - self.meta_size[2] ** 2 / self.meta_size[1] ** 2) * x_t + y_t,
                                        -eps)
        right_line_constraint = torch.le(math.sqrt(4 - self.meta_size[2] ** 2 / self.meta_size[1] ** 2) * x_t - y_t,
                                         eps)
        meta_mask = (
                    circle_top_constraint * circle_bottom_constraint * left_line_constraint * right_line_constraint).float()
        return meta_mask[None, None, ...].repeat(1, 3, 1, 1)


class TexturePie(object):

    def __init__(self, image_size=(3, 512, 512), num_pies=4, meta_aa_factor=1, meta_path=None,
                 down_up_sample_scale=0.4, device="cpu"):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.num_pies = num_pies
        self.single_pie_angle = 360 / num_pies
        meta_height = image_size[-1] / 2
        meta_width = Utils.calculate_chord_length(meta_height, self.single_pie_angle)
        meta_height_q, meta_width_q = Utils.adaptive_width_height_quantization(meta_height * meta_aa_factor,
                                                                               meta_height / meta_width)
        meta_size = (image_size[0], meta_height_q, meta_width_q)
        self.meta = Meta(meta_aa_factor=meta_aa_factor, meta_size=meta_size, meta_path=meta_path, device=device)
        self.ensemble_mats = self.init_ensemble_mats()
        self.down_up_sample_scale = down_up_sample_scale

    def init_ensemble_mats(self):
        ensemble_mats = []
        # raw circle
        radius_r = 1
        # inside circle
        radius_i = Utils.calculate_chord_length(radius_r, self.single_pie_angle) / 2
        # outside circle
        radius_o = math.sqrt(radius_i ** 2 + radius_r ** 2)
        # target coordinates
        target_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
        for pie_idx in range(self.num_pies):
            angle_a = 90 + pie_idx * self.single_pie_angle
            angle_b = pie_idx * self.single_pie_angle - 90
            angle_bias = math.degrees(math.atan(radius_i / radius_r))
            angle_c = pie_idx * self.single_pie_angle - angle_bias
            angle_d = pie_idx * self.single_pie_angle + angle_bias
            source_coords = np.array([[radius_i * math.cos(math.radians(angle_a)),
                                       radius_i * math.sin(math.radians(angle_a))],
                                      [radius_i * math.cos(math.radians(angle_b)),
                                       radius_i * math.sin(math.radians(angle_b))],
                                      [radius_o * math.cos(math.radians(angle_c)),
                                       radius_o * math.sin(math.radians(angle_c))],
                                      [radius_o * math.cos(math.radians(angle_d)),
                                       radius_o * math.sin(math.radians(angle_d))]
                                      ], dtype=np.float32)
            ensemble_mat = Utils.init_perspective_mats(source_coords, target_coords, device=self.device)
            ensemble_mats.append(ensemble_mat)
        return torch.cat(ensemble_mats, dim=0)

    def forward_texture(self):
        batch_grids = Utils.make_grids(self.ensemble_mats, self.image_size, self.device)
        meta_images = F.grid_sample(self.meta.forward().repeat(self.num_pies, 1, 1, 1), batch_grids, align_corners=True,
                                    mode="bilinear", padding_mode="zeros")
        texture = torch.clamp(meta_images.sum(0), 0., 1.)
        # for idx in range(len(meta_images)):
        #     vutils.save_image(meta_images[idx], f"meta_{idx}.png", normalize=True)
        # vutils.save_image(texture, f"texture.png", normalize=True)
        # B x H x W x C
        return texture[None, ...]

    def forward(self):
        texture = self.forward_texture()
        scale = random.random() * (1 - self.down_up_sample_scale) + self.down_up_sample_scale
        down_sample = F.interpolate(texture, scale_factor=scale, mode='bilinear', align_corners=False)
        up_sample = F.interpolate(down_sample, size=texture.shape[2:], mode='bilinear', align_corners=False)
        # B x H x W x C
        return up_sample.permute(0, 2, 3, 1)

    @torch.no_grad()
    def optimize(self, lr, *args, **kwargs):
        gard = self.meta.meta.grad.clone()
        meta = self.meta.meta - lr * torch.sign(gard)
        self.meta.meta = torch.clamp(meta, 0., 1.) * self.meta.meta_mask

    def reset_grads(self):
        self.meta.meta.requires_grad = True
        self.meta.meta.grad = None

    @torch.no_grad()
    def clip_colorset(self, cms, suffix):
        self.meta.clip_colorset(cms, suffix)

    @torch.no_grad()
    def sync_send(self):
        return self.meta.forward()

    @torch.no_grad()
    def sync_recv(self, meta):
        self.meta.meta = meta * self.meta.meta_mask

    def dump(self, out_fd, suffix):
        os.makedirs(out_fd, exist_ok=True)
        vutils.save_image(self.forward_texture()[0].detach().cpu(), osp.join(out_fd, f"texture_{suffix}.png"))
        vutils.save_image(self.meta.meta.detach().cpu(), osp.join(out_fd, f"meta_{self.num_pies}_{suffix}.png"))


class Utils(object):

    @staticmethod
    def calculate_chord_length(radius, angle_in_degrees):
        # Convert angles to radians
        angle_in_radians = math.radians(angle_in_degrees)
        # Calculate chord length using formula
        chord_length = 2 * radius * math.sin(angle_in_radians / 2)
        return chord_length

    @staticmethod
    def adaptive_width_height_quantization(base_height, ratio):
        candidate_height = np.array([int(base_height) + i for i in range(10)])
        candidate_width = candidate_height / ratio
        differ = np.abs(candidate_width.round() - candidate_width)
        idx = np.argmin(differ)
        return int(candidate_height[idx]), int(candidate_width.round()[idx])

    @staticmethod
    def init_perspective_mats(source_coords, target_coords, device):
        matrix = cv2.getPerspectiveTransform(source_coords, target_coords)
        return torch.from_numpy(matrix[None, ...]).float().to(device)

    @staticmethod
    def make_grids(perspective_mat, graph_size, device):
        num_batch = perspective_mat.shape[0]
        # Generate grid coordinates
        x = torch.arange(-1, 1, 2.0 / graph_size[-1]).to(device)
        y = torch.arange(-1, 1, 2.0 / graph_size[-2]).to(device)
        # Make sure the grid is correct
        y_t, x_t = torch.meshgrid([y, x], indexing='ij')
        x_t_flat = x_t.contiguous().view(-1)
        y_t_flat = y_t.contiguous().view(-1)
        # Apply coordinate transformation
        ones = torch.ones_like(x_t_flat).to(device)
        sampling_grid = torch.stack((x_t_flat, y_t_flat, ones))
        sampling_grid = torch.unsqueeze(sampling_grid, 0)
        sampling_grid = sampling_grid.repeat(num_batch, 1, 1)
        batch_grids = torch.matmul(perspective_mat.float(), sampling_grid)
        # Remove homogeneous terms and map back to the correct coordinate system
        batch_grids_x = torch.div(batch_grids[:, 0, :], batch_grids[:, 2, :])
        batch_grids_y = torch.div(batch_grids[:, 1, :], batch_grids[:, 2, :])
        batch_grids_x = batch_grids_x.view(num_batch, graph_size[-2], graph_size[-1])
        batch_grids_y = batch_grids_y.view(num_batch, graph_size[-2], graph_size[-1])
        batch_grids = torch.stack((batch_grids_x, batch_grids_y), 3)
        return batch_grids

