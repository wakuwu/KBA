import os
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, SoftPhongShader, Materials
from pytorch3d.structures import join_meshes_as_batch

from renderer.lights import *
from renderer.carrier import *
from renderer.cameras import *
from renderer.environments import *
from renderer.victims import *


class DataLoaderTest(object):

    def __init__(self, lights, meshes, cameras, view_params, render_params, device="cpu"):
        """
        :param aa_factor: antialiasing factor
        """
        super().__init__()
        self.device = device
        self.render_size = render_params["render_size"]
        self.aa_factor = render_params["aa_factor"]
        self.lights = self.instantiate(lights)
        self.meshes = self.instantiate(meshes)
        self.cameras = self.instantiate(dict(camera=cameras))["camera"]
        default_view_params = view_params.pop("default", None)
        if default_view_params is not None:
            for i in range(len(self.cameras)):
                view_params[i] = view_params.get(i, dict())
            for key, value in view_params.items():
                new_dict = self.update_dict(value, default_view_params)
                view_params[key] = new_dict
        self.view_params = view_params
        self.raster_settings = self.init_raster_settings()
        # Buffer render results
        self.outputs = None

    def instantiate(self, params_dict):
        objects = dict()
        for key, value in params_dict.items():
            objects[key] = eval(value.pop("type"))(device=self.device, **value)
        return objects

    @staticmethod
    def update_dict(dic, base_dic):
        base_dic = base_dic.copy()
        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic and base_dic.get(key) is not None:
                base_dic[key] = DataLoaderTest.update_dict(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def init_raster_settings(self):
        raster_settings = RasterizationSettings(
            image_size=(np.array(self.render_size, dtype=np.int32) * int(self.aa_factor)).tolist(),
            faces_per_pixel=1,
            bin_size=256,
            max_faces_per_bin=1000000
        )
        return raster_settings

    def vis_render(self, out_fd):
        images = self.outputs.get("images_synthesis", None)
        if images is None:
            return
        os.makedirs(out_fd, exist_ok=True)
        for image_idx in range(len(images)):
            vutils.save_image(images[image_idx][0:3, ...].detach().cpu(), osp.join(out_fd, f"{image_idx:05d}.png"))

    def forward(self):
        self.outputs = self.forward_batch()
        return self.outputs

    def forward_batch(self):
        images_synthesis_list = []
        for view_idx in range(len(self.cameras)):
            view_params = self.view_params[view_idx]
            # Combine the meshes
            combined_meshes_list = []
            for obj in view_params["meshes_groups"]:
                meshes_transform = view_params["meshes_transforms"].get(obj, dict())
                combined_meshes_list.append(self.meshes[obj].forward(**meshes_transform))
            combined_meshes = join_meshes_as_batch(combined_meshes_list)
            # Prepare the batch of cameras
            camera = self.cameras.forward(idx=view_idx)
            # Render
            rasterizer = MeshRasterizer(cameras=camera, raster_settings=self.raster_settings)
            shader = SoftPhongShader(
                device=self.device,
                cameras=camera,
                lights=self.lights[view_params["light"]].forward(),
                materials=Materials(ambient_color=((1, 1, 1),), diffuse_color=((1, 1, 1),),
                                    specular_color=((1, 1, 1),),
                                    shininess=500, device=self.device)
            )
            fragments = rasterizer(combined_meshes)
            # B x C x H x W
            images = shader(fragments, combined_meshes).permute([0, 3, 1, 2])
            # Fused all images
            masks = (fragments.pix_to_face != -1).any(dim=-1).to(torch.float32)
            images_synthesis = images[0] * masks[0]
            for i in range(len(images) - 1):
                images_synthesis = images_synthesis * (1 - masks[i+1]) + images[i+1] * masks[i+1]
            # anti-aliasing
            images_synthesis_aa = F.avg_pool2d(images_synthesis[None, ...], kernel_size=self.aa_factor,
                                               stride=self.aa_factor)[:, 0:3, ...]
            images_synthesis_list.append(images_synthesis_aa.cpu())
        images_synthesis_outputs = torch.cat(images_synthesis_list, dim=0)
        return dict(images_synthesis=images_synthesis_outputs)

    def __len__(self):
        return len(self.cameras)


if __name__ == '__main__':
    device = "cuda"

