import os
import os.path as osp
import torch
import torchvision.utils as vutils
import torchvision.transforms as tt
from PIL import Image


class ColorManagementSystem(object):

    def __init__(self, icc_fp, tmp_dir=None):
        super().__init__()
        self.icc_fp = icc_fp
        if tmp_dir is None:
            tmp_dir = "./tmp"
        self.tmp_dir = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)

    @torch.no_grad()
    def forward(self, image_tensor, suffix):
        vutils.save_image(image_tensor, osp.join(self.tmp_dir, f"tmp_{suffix}.tif"))
        os.system(
            f"tificc -t 3 -p {self.icc_fp} -m 3 -c2 "
            f"{osp.join(self.tmp_dir, f'tmp_{suffix}.tif')} "
            f"{osp.join(self.tmp_dir, f'tmp_t_{suffix}.tif')}")
        image_tensor_t = tt.ToTensor()(Image.open(osp.join(self.tmp_dir, f"tmp_t_{suffix}.tif")))[None, 0:3, ...].to(
            image_tensor.device)
        return image_tensor_t


if __name__ == '__main__':
    cms = ColorManagementSystem(icc_fp="../../../data/cms/Adobe_ICC_Profiles/CMYK/USWebCoatedSWOP.icc")
    meta = tt.ToTensor()(Image.open("../../../test.png"))[None, 0:3, ...]
    meta_t = cms.forward(meta, "0")
    vutils.save_image(meta_t[0], osp.join("tmp2.png"))
    print(meta_t.shape)
