import math
import random

import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras


class CamerasTwoView(object):

    def __init__(self, dist_range=(1.8, 3), elev_range=(30, 80), azim_range=(0, 360), look_at_surface_radius=0.5,
                 elev_limit_range=(-10, 10), azim_limit_range=(-90, 90), dist_limit_range=(-0.1, 0.1), device="cpu"):
        super().__init__()
        self.device = device
        self.dist_range = dist_range
        # The angle between the vector from the object to the cameras, and the horizontal plane y = 0 (xz-plane).
        self.elev_range = elev_range
        self.azim_range = azim_range
        self.look_at_surface_radius = look_at_surface_radius
        self.elev_limit_range = elev_limit_range
        self.azim_limit_range = azim_limit_range
        self.dist_limit_range = dist_limit_range
        self.azim_delta = 0

    def init_rt_matrix(self):
        r_matrix_list = []
        t_matrix_list = []
        # Random sample a position as at vector
        theta = random.uniform(0, 2 * math.pi)
        r = self.look_at_surface_radius * math.sqrt(random.uniform(0, 1))
        at = ((r * math.cos(theta), 0, r * math.sin(theta)),)
        # view 1
        dist_1 = random.uniform(*self.dist_range)
        elev_1 = random.uniform(*self.elev_range)
        azim_1 = random.uniform(*self.azim_range)
        # Calculate R and T matrix
        R, T = look_at_view_transform(dist_1, elev_1, azim_1, at=at, up=((0, 1, 0),), device=self.device)
        r_matrix_list.append(R)
        t_matrix_list.append(T)
        # view 2
        dist_2 = random.uniform(*self.dist_limit_range) + dist_1
        dist_2 = np.clip(dist_2, a_min=self.dist_range[0], a_max=self.dist_range[1])
        elev_2 = random.uniform(*self.elev_limit_range) + elev_1
        elev_2 = np.clip(elev_2, a_min=self.elev_range[0], a_max=self.elev_range[1])
        azim_2 = random.uniform(*self.azim_limit_range) + azim_1
        azim_2 = np.clip(azim_2, a_min=self.azim_range[0], a_max=self.azim_range[1])
        # Buffer delta of azim in view1 & view2
        self.azim_delta = torch.tensor(azim_2 - azim_1, device=self.device)
        # Calculate R and T matrix
        R, T = look_at_view_transform(dist_2, elev_2, azim_2, at=at, up=((0, 1, 0),), device=self.device)
        r_matrix_list.append(R)
        t_matrix_list.append(T)
        # Combine the R and T
        r_matrix = torch.cat(r_matrix_list, dim=0)
        t_matrix = torch.cat(t_matrix_list, dim=0)
        return r_matrix, t_matrix

    def forward(self, extend=1):
        r_matrix, t_matrix = self.init_rt_matrix()
        r_matrix_extended = r_matrix[None, ...].repeat(extend, 1, 1, 1).reshape(-1, 3, 3)
        t_matrix_extended = t_matrix[None, ...].repeat(extend, 1, 1).reshape(-1, 3)
        cameras = FoVPerspectiveCameras(zfar=100000, R=r_matrix_extended, T=t_matrix_extended, device=self.device)
        return cameras

    @torch.no_grad()
    def get_label(self):
        """
        Camera coordinates x: left; y: down; z: up;
        Image coordinates x: left; y: down;
        """
        d = self.azim_delta
        axis_thetas = torch.tensor([0., 90, -90,
                                    0 + d, 90 + d, -90 + d,
                                    0 - d, 90 - d, -90 - d,
                                    0., 90, -90,], device=self.device)
        axis_vectors = self.axis_direction_vectors(axis_thetas)
        # pair | view | axis | x&y
        axis_vectors = axis_vectors.reshape([2, 2, 2, 3]).permute([1, 2, 3, 0])
        return axis_vectors

    @staticmethod
    def axis_direction_vectors(thetas):
        thetas_rad = torch.deg2rad(thetas)
        axis_vector = torch.stack([torch.cos(thetas_rad), torch.sin(thetas_rad)], dim=0)
        return axis_vector

    def __len__(self):
        return 2


if __name__ == '__main__':
    t = CamerasTwoView(device="cuda")
    cameras = t.forward()
    t.get_label()
    print(cameras)
