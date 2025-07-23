import torch

from attack import Attack
from loss import LossAttacks
from utils import init_seed


def get_adv_dataloader_params():
    params = dict(
        carrier_params=dict(mesh_type="MeshDisc",
                            texture_type="TexturePie",
                            mesh_params=dict(disc_path="data/carriers/disc/carrier.obj",
                                             disc_radius_range=(1.0, 1.0),
                                             rotate_range=(0, 360)),
                            texture_params=dict(
                                image_size=(3, 1000, 1000), num_pies=12, meta_aa_factor=2, down_up_sample_scale=0.5
                            )),
        victim_params=dict(dataset_dir="data/dataset",
                           scale_range=(0.4, 1.0),
                           rotate_range=((-90, 90), (-180, 180), (-90, 90)),
                           translate_range=((-0.1, 0.1), (-0.0, 0.0), (-0.1, 0.1)),
                           target_objects=dict()),
        lights_params=dict(color_scale_range=((0.6, 1.0), (0.0, 0.1), (0.0, 0.1)),
                           color_delta=(-0.1, 0.1),
                           location_rage=((-10, 10), (0, 10), (-10, 10))),
        cameras_params=dict(dist_range=(2.5, 3.0), elev_range=(15, 85),
                            azim_range=(0, 360), look_at_surface_radius=0.2, dist_limit_range=(-0.1, 0.1),
                            elev_limit_range=(-10, 10), azim_limit_range=(-180, 180)),
        environment_params=dict(environments_dir="data/environments", sphere_radius=10000, translate_y=0),
        cms_params=dict(icc_fp="data/cms/CMYK/USWebCoatedSWOP.icc"),
        post_process_params=dict(GaussianNoise=dict(mean=0.0, stddev=5.0)),
        render_size=(384, 512),
        aa_factor=2,
        require_grads=True,
        forward_type="batch",
    )
    return params

def get_adv_optim_params():
    params = dict(num_steps=20001, lr=1 / 255,
                  loss_coefficients=dict(loss_chaos_direction=-1.0),
                  loss_func="loss_chaos_direction", loss_params=dict(), clip_colorset_frequency=500)
    return params

class AttackDust3r(Attack):

    def __init__(self, dataloader_params=None, optim_params=None, device="cpu"):
        super().__init__(dataloader_params.copy(), optim_params.copy(), device)

    def loss(self, batch, out):
        loss_combined = 0
        loss_coefficients = self.attack_params["loss_coefficients"]
        loss_dict = getattr(LossAttacks, self.attack_params["loss_func"])(batch, out,
                                                                          **self.attack_params["loss_params"])
        for loss_name, loss in loss_dict.items():
            coefficient = loss_coefficients.get(loss_name, 0)
            loss_combined += coefficient * loss
        loss_dict["loss_combined"] = loss_combined
        return loss_dict



if __name__ == '__main__':
    init_seed(2024)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    optim_obj = AttackDust3r(get_adv_dataloader_params(), get_adv_optim_params(), device=device)
    optim_obj.run(0)
