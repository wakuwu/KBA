import os
import os.path as osp
import random

import torch
from iopath.common.file_io import PathManager
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io.utils import _read_image
from pytorch3d.transforms import Translate, Scale, Transform3d

class Environments(object):

    def __init__(self, environments_dir, sphere_radius=10000, translate_y=0., item_list=None, device="cpu"):
        super().__init__()
        self.device = device
        self.environments_dir = environments_dir
        self.sphere_radius = sphere_radius
        # -1 ~ 1
        self.translate_y = translate_y
        if item_list is None:
            item_list = []
        self.item_list = item_list
        self.sphere_mesh = self.init_sphere_object()
        self.texture_images = self.load_all_texture_images_to_mem()

    def init_sphere_object(self):
        mesh = load_objs_as_meshes([osp.join(self.environments_dir, "environment.obj")], device=self.device)
        mesh_t = self.prefix_transform(mesh, self.device)
        mesh_t.textures._maps_padded = None
        return mesh_t

    def load_all_texture_images_to_mem(self):
        device = "cpu"
        HDRIs_dir = osp.join(self.environments_dir, "HDRIs")
        texture_images = {}
        for hdri in os.listdir(HDRIs_dir):
            if len(self.item_list) > 0:
                if hdri not in self.item_list:
                    continue
            image = (_read_image(osp.join(HDRIs_dir, hdri), path_manager=PathManager(), format="RGB") / 255.0)
            image = torch.from_numpy(image).to(device)
            texture_images[hdri] = image[None, ...]
        return texture_images

    def forward(self, **kwargs):
        texture_image = self.texture_images[random.choice(list(self.texture_images.keys()))]
        self.sphere_mesh.textures._maps_padded = texture_image.to(self.device)
        return self.sphere_mesh

    def prefix_transform(self, mesh, device="cpu"):
        scaling = Scale(self.sphere_radius, device=device)
        # Object position aligned with the origin.
        translation = Translate(0, self.translate_y, 0, device=device)
        transform = Transform3d(device=device).compose(translation, scaling)
        vertices = mesh.verts_packed()
        mesh_t = mesh.update_padded(transform.transform_points(vertices)[None, ...])
        return mesh_t


if __name__ == "__main__":
    t = Environments("../../data/environments", device="cuda")
    t.forward()
