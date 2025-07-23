import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from pytorch3d.renderer import RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, Materials
from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch

from renderer.cameras.cameras_two_view import CamerasTwoView
from renderer.carrier.carrier import Carrier
from renderer.carrier.texture.color_management_system import ColorManagementSystem
from renderer.environments.environments import Environments
from renderer.lights.point_light import PointLight
from renderer.postprocess.post_process import PostProcess
from renderer.victims.victims import Victims


class DataLoaderAttack(object):

    def __init__(self, carrier_params, victim_params, lights_params, cameras_params, environment_params=None,
                 cms_params=None, post_process_params=None,
                 render_size=(384, 512), aa_factor=1, require_grads=False, forward_type="scene", device="cpu"):
        """
        :param aa_factor: antialiasing factor
        """
        super().__init__()
        self.device = device
        self.forward_type = forward_type
        self.render_size = render_size
        self.aa_factor = aa_factor
        self.require_grads = require_grads
        self.carrier = Carrier(device=device, **carrier_params)
        self.victim = Victims(device=device, **victim_params)
        self.lights = PointLight(device=device, **lights_params)
        self.cameras = CamerasTwoView(device=device, **cameras_params)
        if environment_params is not None:
            self.environment = Environments(device=device, **environment_params)
        else:
            self.environment = None
        if cms_params is not None:
            self.cms = ColorManagementSystem(**cms_params, tmp_dir=osp.join(os.environ.get("out_dir", "./"), "tmp"))
        else:
            self.cms = None
        if post_process_params is not None:
            self.post_process = PostProcess(post_process_params, device=device)
        else:
            self.post_process = None
        self.raster_settings = self.init_raster_settings()

    def init_raster_settings(self):
        raster_settings = RasterizationSettings(
            image_size=(np.array(self.render_size, dtype=np.int32) * int(self.aa_factor)).tolist(),
            faces_per_pixel=1,
            bin_size=256,
            max_faces_per_bin=1700000
        )
        return raster_settings

    def forward(self):
        results = None
        if "scene" == self.forward_type:
            results = self.forward_scene()
        elif "batch" == self.forward_type:
            while results is None:
                results = self.forward_batch()
        elif "gt" == self.forward_type:
            results = self.forward_gt()
        else:
            raise NotImplementedError
        results["axis_direction"] = self.cameras.get_label()
        return results

    def forward_scene(self):
        if self.require_grads:
            self.carrier.reset_grads()
        # Combine the meshes
        combined_meshes_list = []
        for obj in [self.victim, self.carrier, self.environment]:
            if obj is not None:
                combined_meshes_list.append(obj.forward())
        combined_meshes = join_meshes_as_scene(combined_meshes_list)
        # Prepare cameras
        cameras = self.cameras.forward()
        # Render
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=self.lights.forward(),
                materials=Materials(ambient_color=((1, 1, 1),), diffuse_color=((1, 1, 1),), specular_color=((0, 0, 0),),
                                    shininess=1000, device=self.device)
            )
        )
        images = renderer(combined_meshes.extend(len(renderer.rasterizer.cameras)))
        # B x C x H x W
        images_synthesis = images.permute([0, 3, 1, 2])
        images_synthesis_aa = F.avg_pool2d(images_synthesis, kernel_size=self.aa_factor, stride=self.aa_factor)[:, 0:3,
                              ...]
        if self.post_process is not None:
            images_output = self.post_process.forward(images_synthesis_aa)
        else:
            images_output = images_synthesis_aa
        return dict(images_synthesis=images_output)

    def forward_batch(self):
        if self.require_grads:
            self.carrier.reset_grads()
        # Combine the meshes
        combined_meshes_list = []
        for obj in [self.victim, self.carrier, self.environment]:
            if obj is not None:
                combined_meshes_list.append(obj.forward())
        combined_meshes = join_meshes_as_batch(combined_meshes_list)
        # Prepare cameras
        cameras = self.cameras.forward(extend=len(combined_meshes))
        # Render
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        shader = SoftPhongShader(
            device=self.device,
            cameras=cameras,
            lights=self.lights.forward(),
            materials=Materials(ambient_color=((1, 1, 1),), diffuse_color=((1, 1, 1),),
                                specular_color=((1, 1, 1),),
                                shininess=500, device=self.device)
        )
        combined_meshes_extended = combined_meshes.extend(len(self.cameras))
        fragments = rasterizer(combined_meshes_extended)
        # B x C x H x W
        images = shader(fragments, combined_meshes_extended).permute([0, 3, 1, 2])
        b, c, h, w = images.shape
        images = images.reshape(len(combined_meshes), len(self.cameras), c, h, w).permute([0, 2, 3, 4, 1])
        # Fused all images
        # with torch.no_grad():
        masks = (fragments.pix_to_face != -1).any(dim=-1).to(torch.float32)
        masks = masks.reshape(len(combined_meshes), len(self.cameras), h, w).permute([0, 2, 3, 1])
        images_synthesis = images[0] * masks[0] + images[1] * (1 - masks[0])
        masks_victim = masks[0:1]
        masks_victim_carrier = (masks[0:1].bool() | masks[1:2].bool()).float()
        masks_carrier_wo_victim = (1 - masks[0:1]) * masks[1:2]
        # Check if the victims completely covers the carrier
        ratio = masks_carrier_wo_victim.reshape(-1, len(self.cameras)).sum(0) / masks[1:2].reshape(-1,
                                                                                                   len(self.cameras)).sum(
            0)
        print(f"Object occlusion ratio: {(1 - ratio).cpu().numpy().tolist()}")
        if (ratio < 0.1).sum() > 0:
            print(f"Victim is too large which completely covers the carrier! {ratio.cpu().numpy().tolist()}")
            return None
        if self.environment is not None:
            images_synthesis = images_synthesis * masks_victim_carrier + images[2] * (1 - masks_victim_carrier)
        # anti-aliasing
        images_synthesis_aa = F.avg_pool2d(images_synthesis.permute([3, 0, 1, 2]), kernel_size=self.aa_factor,
                                           stride=self.aa_factor)[:, 0:3, ...]
        if self.post_process is not None:
            images_output = self.post_process.forward(images_synthesis_aa)
        else:
            images_output = images_synthesis_aa
        masks_victim_t = F.avg_pool2d(masks_victim.permute([3, 0, 1, 2]), kernel_size=self.aa_factor,
                                      stride=self.aa_factor)
        masks_carrier_wo_victim_t = F.avg_pool2d(masks_carrier_wo_victim.permute([3, 0, 1, 2]), kernel_size=self.aa_factor,
                                                 stride=self.aa_factor)
        # B x C x H x W
        results = dict(images_synthesis=images_output,
                       masks_victim=masks_victim_t.bool().float(),
                       masks_carrier_wo_victim=masks_carrier_wo_victim_t.bool().float())
        return results

    def forward_gt(self):
        if self.require_grads:
            self.carrier.reset_grads()
        # Combine the meshes
        combined_meshes_list = []
        for obj in [self.victim, self.carrier, self.environment]:
            if obj is not None:
                combined_meshes_list.append(obj.forward())
        combined_meshes = join_meshes_as_batch(combined_meshes_list)
        # Prepare cameras
        cameras = self.cameras.forward(extend=len(combined_meshes))
        # Render
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        shader = SoftPhongShader(
            device=self.device,
            cameras=cameras,
            lights=self.lights.forward(),
            materials=Materials(ambient_color=((1, 1, 1),), diffuse_color=((1, 1, 1),),
                                specular_color=((1, 1, 1),),
                                shininess=500, device=self.device)
        )
        combined_meshes_extended = combined_meshes.extend(len(self.cameras))
        fragments = rasterizer(combined_meshes_extended)
        # B x C x H x W
        images = shader(fragments, combined_meshes_extended).permute([0, 3, 1, 2])
        b, c, h, w = images.shape
        images = images.reshape(len(combined_meshes), len(self.cameras), c, h, w).permute([0, 2, 3, 4, 1])
        # Fused all images
        # with torch.no_grad():
        masks = (fragments.pix_to_face != -1).any(dim=-1).to(torch.float32)
        masks = masks.reshape(len(combined_meshes), len(self.cameras), h, w).permute([0, 2, 3, 1])
        if self.environment is not None:
            images_synthesis = images[1] * masks[1] + images[2] * (1 - masks[1])
        else:
            images_synthesis = images[1] * masks[1]
        images_gt = images_synthesis.detach()
        images_synthesis = images[0] * masks[0] + images_synthesis * (1 - masks[0])
        masks_victim = masks[0:1]
        masks_carrier = masks[1:2]
        # anti-aliasing
        images_synthesis_aa = F.avg_pool2d(images_synthesis.permute([3, 0, 1, 2]), kernel_size=self.aa_factor,
                                           stride=self.aa_factor)[:, 0:3, ...]
        images_gt_aa = F.avg_pool2d(images_gt.permute([3, 0, 1, 2]), kernel_size=self.aa_factor,
                                    stride=self.aa_factor)[:, 0:3, ...]
        if self.post_process is not None:
            images_output = self.post_process.forward(images_synthesis_aa)
            images_output_gt = self.post_process.forward(images_gt_aa)
        else:
            images_output = images_synthesis_aa
            images_output_gt = images_gt_aa
        masks_victim_t = F.avg_pool2d(masks_victim.permute([3, 0, 1, 2]), kernel_size=self.aa_factor,
                                      stride=self.aa_factor)
        masks_carrier_t = F.avg_pool2d(masks_carrier.permute([3, 0, 1, 2]), kernel_size=self.aa_factor,
                                                 stride=self.aa_factor)
        # B x C x H x W
        results = dict(images_synthesis=images_output,
                       images_synthesis_gt=images_output_gt,
                       masks_victim=masks_victim_t.bool().float(),
                       masks_carrier=masks_carrier_t.bool().float())
        return results

    @torch.no_grad()
    def optimize(self, lr, *args, **kwargs):
        self.carrier.optimize(lr, *args, **kwargs)

    def dump(self, out_fd, suffix):
        self.carrier.dump(out_fd, suffix)

    @staticmethod
    def vis_render(images, out_fd, save_alpha=False):
        os.makedirs(out_fd, exist_ok=True)
        for image_idx in range(len(images)):
            if save_alpha:
                vutils.save_image(images[image_idx].detach().cpu(), osp.join(out_fd, f"{image_idx:05d}.png"))
            else:
                vutils.save_image(images[image_idx][0:3, ...].detach().cpu(), osp.join(out_fd, f"{image_idx:05d}.png"))

