from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, camera_position_from_spherical_angles
import torch
import numpy as np
import itertools


class Cameras(object):

    def __init__(self, num_views, dist, pitch_degree, yaw_degree, at, roll_degree, device="cpu"):
        """
        raw_param_list: [[[dist, ...], [pitch_degree, ...], [yaw_degree, ...], [at, ...], [roll_degree, ...]]
        :param param_list: [[dist, pitch_degree, yaw_degree, at, roll_degree], ]
        """
        super().__init__()
        self.device = device
        self.num_views = num_views
        if num_views == max(len(lst) for lst in [dist, pitch_degree, yaw_degree, at, roll_degree]):
            self.param_list = self.broadcast_params([dist, pitch_degree, yaw_degree, at, roll_degree],
                                                    self.num_views)
        else:
            self.param_list = list(itertools.product(*[dist, pitch_degree, yaw_degree, at, roll_degree]))
        self.r_matrix, self.t_matrix = self.init_rt_matrix()
        self.cameras = FoVPerspectiveCameras(zfar=100000, R=self.r_matrix, T=self.t_matrix, device=self.device)

    def init_rt_matrix(self):
        r_matrix_list = []
        t_matrix_list = []
        for camera_idx in range(len(self.param_list)):
            dist, pitch_degree, yaw_degree, at, roll_degree = self.param_list[camera_idx]
            if at is None:
                at = ((0, 0, 0),)
            roll_axis = camera_position_from_spherical_angles(dist, pitch_degree, yaw_degree, degrees=True)[0].numpy()
            up = self.rotate_vector_around_axis(np.array((0, 1, 0), dtype=np.float32), roll_axis, roll_degree)
            # Calculate R and T matrix
            R, T = look_at_view_transform(dist, pitch_degree, yaw_degree, at=at, up=(tuple(up.tolist()),),
                                          device=self.device)
            r_matrix_list.append(R)
            t_matrix_list.append(T)
        r_matrix = torch.cat(r_matrix_list, dim=0)
        t_matrix = torch.cat(t_matrix_list, dim=0)
        return r_matrix, t_matrix

    def forward(self, idx=None, **kwargs):
        if idx is None:
            return self.cameras
        return self.cameras[idx]

    @staticmethod
    def rotate_vector_around_axis(v, axis, angle_degrees):
        """
        Rotate vector v around axis by angle_degrees.
        """
        angle_radians = np.radians(angle_degrees)
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)
        # Rodrigues' rotation formula
        term1 = v * cos_theta
        term2 = np.cross(axis, v) * sin_theta
        term3 = axis * np.dot(axis, v) * (1 - cos_theta)
        return term1 + term2 + term3

    @staticmethod
    def broadcast_params(raw_params, num_groups):
        param_list = []
        for params in raw_params:
            repeat_count = num_groups // len(params) + (0 if num_groups % len(params) == 0 else 1)
            tmp_params = sum([[i] * repeat_count for i in params], [])
            param_list.append(tmp_params)
        transposed_list = [list(row) for row in zip(*param_list)]
        return transposed_list

    def __len__(self):
        return len(self.cameras)


if __name__ == '__main__':
    # [[dist, pitch_degree, yaw_degree, at, roll_degree], ]
    t = Cameras(6, [2], [30, 60], [i for i in range(0, 360, 60)], [((0, 0, 0),)], [0], device="cuda")
    cameras = t.forward()
    print(cameras)
