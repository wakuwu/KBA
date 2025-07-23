import os
import os.path as osp
import random
import torch
import torchvision.utils as vutils
import torchvision.transforms as tt
import torch.nn.functional as F
from PIL import Image

class TextureDisc(object):
    def __init__(self, image_size=(3, 512, 512), image_path=None, down_up_sample_scale=0.4, device="cpu"):
        """
        :param image_size:  C x H x W
        """
        super().__init__()
        self.device = device
        assert image_size is not None or image_path is not None
        if image_path is not None:
            image = tt.ToTensor()(Image.open(image_path))[None, ...].to(self.device)
            self.image_size = image.shape[-3:]
        else:
            image = torch.randint(size=image_size, low=0, high=256, device=self.device)[None, ...] / 255.
            # image = torch.ones_like(image)
            if image.shape[-3] < 3:
                image = image.repeat(1, 3, 1, 1)
            self.image_size = image_size
        # B x H x W x C
        self.texture_mask = self.init_texture_mask()
        self.texture = self.texture_mask * image[:, 0:image_size[0]]
        self.down_up_sample_scale = down_up_sample_scale

    def forward_texture(self):
        return self.texture * self.texture_mask

    def forward(self):
        texture = self.forward_texture()
        scale = random.random() * (1 - self.down_up_sample_scale) + self.down_up_sample_scale
        down_sample = F.interpolate(texture, scale_factor=scale, mode='bilinear', align_corners=False)
        up_sample = F.interpolate(down_sample, size=texture.shape[2:], mode='bilinear', align_corners=False)
        # B x H x W x C
        return up_sample.permute(0, 2, 3, 1)

    def init_texture_mask(self):
        x = torch.linspace(-1, 1, self.image_size[2], device=self.device)
        y = torch.linspace(-1, 1, self.image_size[1], device=self.device)
        y_t, x_t = torch.meshgrid([y, x], indexing='ij')
        eps = 0.001
        mask = torch.le(x_t*x_t + y_t*y_t, 1 + eps).float()
        return mask[None, None, ...].repeat(1, 3, 1, 1)

    @torch.no_grad()
    def clip_colorset(self, cms, suffix):
        self.texture = cms.forward(self.forward_texture(), suffix) * self.texture_mask

    @torch.no_grad()
    def optimize(self, lr, *args, **kwargs):
        gard = self.texture.grad.clone()
        texture = self.texture - lr * torch.sign(gard)
        self.texture = torch.clamp(texture, min=0., max=1.) * self.texture_mask

    def reset_grads(self):
        self.texture.requires_grad = True
        self.texture.grad = None

    @torch.no_grad()
    def sync_send(self):
        return self.forward_texture()

    @torch.no_grad()
    def sync_recv(self, texture):
        self.texture = texture * self.texture_mask

    def dump(self, out_fd, suffix):
        os.makedirs(out_fd, exist_ok=True)
        vutils.save_image(self.forward_texture()[0].detach().cpu(), osp.join(out_fd, f"texture_{suffix}.png"))


if __name__ == "__main__":
    t = TextureDisc(image_size=(1, 512, 512))
