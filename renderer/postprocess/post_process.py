from renderer.postprocess.gaussian_noise import GaussianNoise
from renderer.postprocess.straightforward import StraightForward


class PostProcess(object):

    def __init__(self, post_process_params=None, device="cpu"):
        super().__init__()
        self.device = device
        if post_process_params is None:
            post_process_params = dict(StraightForward=dict())
        self.post_process_list = []
        for cls, params in post_process_params.items():
            self.post_process_list.append(eval(cls)(device=self.device, **params))

    def forward(self, image):
        """image: B x C x H x W"""
        for post_process in self.post_process_list:
            image = post_process.forward(image)
        return image

if __name__ == "__main__":
    post_process_params = dict(GaussianNoise=dict(mean=0.0, stddev=5.0))
    pp = PostProcess(post_process_params, "cuda")
    import torch
    image = torch.ones([2, 3, 100, 100], device="cuda") * 0.5
    import torchvision.utils as vutils
    vutils.save_image(image[0], "test_raw.png")
    image_t = pp.forward(image)
    print(image_t.shape)
    vutils.save_image(image_t[0], "test_process.png")