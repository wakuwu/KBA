import random
import os
import os.path as osp

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import RotateAxisAngle, Translate, Scale, Transform3d


class Victims(object):

    def __init__(self, dataset_dir, scale_range=(0.5, 1.1),
                 rotate_range=((0, 360), (0, 360), (0, 360)),
                 translate_range=((-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25)),
                 target_objects=None, device="cpu"):
        super().__init__()
        self.device = device
        self.dataset_dir = dataset_dir
        self.file_suffix = "Scan/Scan.obj"
        if target_objects is None:
            target_objects = dict()
        self.target_objects = target_objects
        self.classes, self.dataset = self.init_dataset()
        self.load_all_objects_to_mem()
        self.scale_range = scale_range
        self.rotate_range = rotate_range
        self.translate_range = translate_range

    def init_dataset(self):
        dataset = dict()
        classes = os.listdir(self.dataset_dir)
        for cls in classes:
            if len(self.target_objects) > 0:
                if cls not in self.target_objects.keys():
                    continue
            items = os.listdir(osp.join(self.dataset_dir, cls))
            for item in items:
                if len(self.target_objects.get(cls, [])) > 0:
                    if item not in self.target_objects[cls]:
                        continue
                cls_dict = dataset.get(cls, dict())
                cls_dict[item] = None
                dataset[cls] = cls_dict
        return classes, dataset

    def load_all_objects_to_mem(self):
        def process_item(cls, item):
            device = self.device
            mesh = load_objs_as_meshes([osp.join(self.dataset_dir, cls, item, self.file_suffix)], device=device)
            mesh_t = self.prefix_transform(mesh, device=device)
            return cls, item, mesh_t
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=8)(
            delayed(process_item)(cls, item)
            for cls, items in self.dataset.items()
            for item in items.keys()
        )
        for cls, item, mesh_t in results:
            self.dataset[cls][item] = mesh_t

    def forward(self, **kwargs):
        target_cls = random.choice(list(self.dataset.keys()))
        target_item = random.choice(list(self.dataset[target_cls].keys()))
        target_object = self.dataset[target_cls][target_item].to(self.device)
        target_object_t = self.suffix_transform(target_object)
        return target_object_t

    @staticmethod
    def prefix_transform(mesh, device="cpu"):
        # Object position aligned with the origin.
        translation = Translate(-(mesh.verts_packed()[..., 0].min() + mesh.verts_packed()[..., 0].max()) / 2,
                                -(mesh.verts_packed()[..., 1].min() + mesh.verts_packed()[..., 1].max()) / 2,
                                -(mesh.verts_packed()[..., 2].min() + mesh.verts_packed()[..., 2].max()) / 2,
                                device=device)
        transform = Transform3d(device=device).compose(translation)
        vertices = mesh.verts_packed()
        mesh_t = mesh.update_padded(transform.transform_points(vertices)[None, ...])
        return mesh_t

    def suffix_transform(self, mesh):
        transform_list = []
        rotation_x = RotateAxisAngle(random.uniform(*self.rotate_range[0]), axis="X", device=self.device)
        rotation_y = RotateAxisAngle(random.uniform(*self.rotate_range[1]), axis="Y", device=self.device)
        rotation_z = RotateAxisAngle(random.uniform(*self.rotate_range[2]), axis="Z", device=self.device)
        transform_list.extend([rotation_x, rotation_y, rotation_z])
        translation = Translate(random.uniform(*self.translate_range[0]),
                                random.uniform(*self.translate_range[1]),
                                random.uniform(*self.translate_range[2]), device=self.device)
        transform_list.append(translation)
        transform = Transform3d(device=self.device).compose(*transform_list)
        mesh_t = mesh.update_padded(transform.transform_points(mesh.verts_packed())[None, ...])
        # Rescale
        object_size = mesh_t.verts_packed().max(0).values - mesh_t.verts_packed().min(0).values
        scale = random.uniform(*self.scale_range) / object_size.max()
        scaling = Scale(scale, device=self.device)
        transform = Transform3d(device=self.device).compose(scaling)
        mesh_t = mesh_t.update_padded(transform.transform_points(mesh_t.verts_packed())[None, ...])
        # Correcting the position of a 3D object
        l = -1 - mesh_t.verts_packed().min(0).values
        r = 1 - mesh_t.verts_packed().max(0).values
        if l[0] <= 0 <= r[0]:
            correction_x = 0
        else:
            correction_x = l[0] if l[0] >= -r[0] else r[0]
        correction_y = -mesh_t.verts_packed()[..., 1].min()
        if l[2] <= 0 <= r[2]:
            correction_z = 0
        else:
            correction_z = l[2] if l[2] >= -r[2] else r[2]
        correction_translate = Translate(correction_x, correction_y, correction_z, device=self.device)
        correction_transform = Transform3d(device=self.device).compose(correction_translate)
        mesh_t_c = mesh_t.update_padded(correction_transform.transform_points(mesh_t.verts_packed())[None, ...])
        return mesh_t_c


if __name__ == "__main__":
    t = Victims("../../data/dataset", device="cuda")
    t.forward()
