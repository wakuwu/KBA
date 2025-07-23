import functools as ft
import math
import os
import os.path as osp
import torch
import third_party.interface_dust3r as tpi
from utils import get_utc8_time, print_to_console_and_file, print_params_to_console_and_file, serialize_loss
from torchvision import transforms
from renderer.data_loader_attack import DataLoaderAttack


class Attack(object):

    def __init__(self, dataloader_params, optim_params, device):
        super().__init__()
        self.device = device
        self.out_dir = self.init_logger()
        self.logger = ft.partial(print_to_console_and_file, file_path=osp.join(self.out_dir, "log.txt"))
        # Log params of running
        print_params_to_console_and_file(dict(renderer_loader_params=dataloader_params),
                                         file_path=osp.join(self.out_dir, "log.txt"))
        print_params_to_console_and_file(dict(attack_params=optim_params),
                                         file_path=osp.join(self.out_dir, "log.txt"))
        self.attack_params = optim_params
        self.renderer_loader_params = dataloader_params
        self.dataloader = DataLoaderAttack(**dataloader_params, device=self.device)
        self.model = self.init_model()
        self.transform = transforms.Compose([
            transforms.Resize(size=(512, 512)),
            transforms.CenterCrop(size=(384, 512)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.loss_trend = []

    def init_logger(self):
        if eval(os.getenv('DEBUG', 'False')):
            out_dir = osp.join("./data/results/attack", "debug_" + get_utc8_time())
        else:
            out_dir = osp.join("./data/results/attack", get_utc8_time())
        os.makedirs(out_dir, exist_ok=True)
        os.environ["out_dir"] = out_dir
        checkpoint_dir = osp.join(out_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return out_dir

    def init_model(self):
        net = tpi.load_model(device=self.device)
        net.eval()
        # Disable postprocess
        net.downstream_head1.postprocess = None
        net.downstream_head2.postprocess = None
        return net

    def loss(self, batch, out):
        return out

    def attack_step(self, batch):
        render_images_t = self.transform(batch.pop("images_synthesis"))
        imgs = tpi.to_dust3r_input(render_images_t)
        pairs = tpi.make_pairs(imgs, scene_graph="complete")
        outputs = tpi.inference(pairs, self.model, self.device, batch_size=1)
        if eval(os.getenv('DEBUG', 'False')):
            with torch.no_grad():
                outputs_debug = torch.stack(outputs, dim=0)
                from utils import heatmap
                for pair in range(len(outputs_debug)):
                    for view in range(2):
                        heatmap(outputs_debug[pair, view, 2:3], out_fp=osp.join(os.environ.get("OUT_DIR", "./"),
                                                                                f"tmp/outputs_z_{pair}_{view}.png"))
        loss_dict = self.loss(batch, outputs)
        loss_dict["loss_combined"].backward()
        self.dataloader.optimize(self.attack_params["lr"])
        self.logger(serialize_loss(loss_dict))
        self.loss_trend.append(loss_dict["loss_combined"].detach().cpu())
        self.logger(serialize_loss(dict(loss_trend=torch.stack(self.loss_trend).mean())))

    def run(self, rank, **kwargs):
        for step in range(self.attack_params["num_steps"]):
            self.logger(
                "================================================================================================")
            self.logger(f"Steps: {step}")
            self.before_step_hook(rank=rank, **kwargs)
            batch = self.dataloader.forward()
            if eval(os.getenv('DEBUG', 'False')):
                with torch.no_grad():
                    import torchvision.utils as vutils
                    vutils.save_image(batch["images_synthesis"][0], osp.join(os.environ.get("OUT_DIR", "./"),
                                                                             f"tmp/images_synthesis_1.png"),
                                      normalize=True)
                    vutils.save_image(batch["images_synthesis"][1], osp.join(os.environ.get("OUT_DIR", "./"),
                                                                             f"tmp/images_synthesis_2.png"),
                                      normalize=True)
            self.attack_step(batch)
            if step % 100 == 0:
                self.dataloader.dump(out_fd=osp.join(self.out_dir, "checkpoint"), suffix=f"{step:06d}")
            if ((step + 1) % self.attack_params.get("clip_colorset_frequency", math.inf) == 0 or
                    step == self.attack_params["num_steps"] - 1):
                self.dataloader.carrier.texture.clip_colorset(cms=self.dataloader.cms, suffix=rank)
            # torch.cuda.empty_cache()
        return

    def before_step_hook(self, **kwargs):
        return
