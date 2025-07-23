import random
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import Scale, Transform3d, RotateAxisAngle, Translate


class MeshDisc(object):

    def __init__(self, disc_path, disc_radius_range=(1.0, 1.0), rotate_range=(0, 360),
                 translate_range=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)), device="cpu"):
        super().__init__()
        self.device = device
        self.disc_path = disc_path
        self.disc_radius_range = disc_radius_range
        self.disc_mesh = self.init_disc_mesh()
        self.rotate_range = rotate_range
        self.translate_range = translate_range

    def forward(self, disc_radius=None, rotate_angle=None, translate=None):
        if disc_radius is None:
            disc_radius = random.uniform(*self.disc_radius_range)
        if rotate_angle is None:
            rotate_angle = random.uniform(*self.rotate_range)
        if translate is None:
            translate = (random.uniform(*self.translate_range[0]),
                         random.uniform(*self.translate_range[1]),
                         random.uniform(*self.translate_range[2]))
        return self.suffix_transform(self.disc_mesh, disc_radius, rotate_angle, translate)

    def init_disc_mesh(self):
        disc_mesh = load_objs_as_meshes([self.disc_path], device=self.device)
        return disc_mesh

    def suffix_transform(self, mesh, scale, angle, translate):
        scaling = Scale(scale, device=self.device)
        rotation = RotateAxisAngle(angle, axis="Y", device=self.device)
        translation = Translate(*translate, device=self.device)
        transform = Transform3d(device=self.device).compose(scaling, rotation, translation)
        vertices = mesh.verts_packed()
        mesh_t = mesh.update_padded(transform.transform_points(vertices)[None, ...])
        return mesh_t

    @staticmethod
    def calculate_rt(yaw_degrees, self_rotation, translate_angle, translate_radius):
        import math
        for idx in range(len(yaw_degrees)):
            r = yaw_degrees[idx] + self_rotation
            t = (translate_radius * math.cos(math.radians(translate_angle-yaw_degrees[idx])), 0,
                 translate_radius * math.sin(math.radians(translate_angle-yaw_degrees[idx])))
            print(f"{idx}: dict(meshes_transforms=dict(CarrierMovable=dict(rotate_angle={r}, translate={t}))),")
        return


if __name__ == "__main__":
    MeshDisc.calculate_rt([i for i in range(0, 360, 60)], 60, 60, 0.7)
