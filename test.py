import os.path as osp
import os
from renderer.data_loader_test import DataLoaderTest
from configs.view_5 import configs

if __name__ == '__main__':
    device = "cuda"
    # texture_path = "data/backgrounds/worn_planks_diff_2k.png"
    texture_path = "data/results/demo/texture_adv.png"
    outputs_dir = f"data/results/visual/{texture_path.split('/')[-1].split('.')[0]}"
    os.makedirs(outputs_dir, exist_ok=True)
    configs["meshes"]["Carrier"]["texture_params"]["image_path"] = texture_path
    t = DataLoaderTest(**configs, device=device)
    import torchvision.utils as vutils
    vutils.save_image(t.meshes['Carrier'].texture.texture, osp.join(outputs_dir, "texture.png"))
    t.forward()
    t.vis_render(outputs_dir)
