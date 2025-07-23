from renderer.carrier.carrier import Carrier


class CarrierAdversarial(Carrier):

    def __init__(self, mesh_class, texture_class, mesh_params, texture_params, device="cpu"):
        super().__init__(mesh_class, texture_class, mesh_params, texture_params, device)

    def forward(self, *args, **kwargs):
        mesh_dict = self.mesh.forward(*args, **kwargs)
        texture_dict = self.texture.forward()
        for texture_name, texture in texture_dict.items():
            for mesh_name, mesh in mesh_dict.items():
                if texture_name in mesh_name:
                    setattr(mesh.textures, "_maps_padded", texture.forward())
        return mesh_dict
