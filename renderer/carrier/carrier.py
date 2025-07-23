import torch
from .mesh import *
from .texture import *

class Carrier(object):

    def __init__(self, mesh_type, texture_type, mesh_params, texture_params, device="cpu"):
        super().__init__()
        self.device = device
        self.texture = eval(texture_type)(device=device, **texture_params)
        self.mesh = eval(mesh_type)(device=device, **mesh_params)

    def forward(self, **kwargs):
        mesh = self.mesh.forward(**kwargs)
        setattr(mesh.textures, "_maps_padded", self.texture.forward())
        return mesh

    def reset_grads(self):
        self.texture.reset_grads()

    @torch.no_grad()
    def optimize(self, *args, **kwargs):
        self.texture.optimize(*args, **kwargs)

    def dump(self, out_fd, suffix):
        self.texture.dump(out_fd, suffix)

