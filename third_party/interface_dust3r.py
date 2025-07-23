import os.path as osp
import sys
from tqdm import trange
import torch
import numpy as np
sys.path.append(osp.join(osp.dirname(__file__), "dust3r"))
inf = float('inf')

from third_party.dust3r.dust3r.model import AsymmetricCroCo3DStereo
import third_party.dust3r.dust3r.inference as infer
import third_party.dust3r.dust3r.image_pairs as image_pairs
import third_party.dust3r.dust3r.utils.device as utils_device

make_pairs = image_pairs.make_pairs

class AsymmetricCroCo3DStereoAttack(AsymmetricCroCo3DStereo):
    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        return res1, res2

def load_model(device, verbose=True):
    model_path = "./third_party/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    # Init different models for attack
    args = args.replace("AsymmetricCroCo3DStereo", "AsymmetricCroCo3DStereoAttack")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


def to_dust3r_input(render_images_t):
    inputs = []
    for idx in range(len(render_images_t)):
        img = render_images_t[idx:idx+1]
        ipt = dict(img=img,
                   true_shape=np.array(img.shape[-2:])[None, ...],
                   idx=idx,
                   instance=str(idx))
        inputs.append(ipt)
    return inputs


def inference(pairs, model, device, batch_size=8, verbose=False):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (infer.check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1
    for i in trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(utils_device.collate_with_cat(pairs[i:i + batch_size]), model, device)
        # result.append(utils_device.to_cpu(res))
        result.append(res)
    # Sort pairs and views by idxes
    result.reverse()
    results = []
    for pair in result:
        pair_sorted = {key: pair[key] for key in sorted(pair.keys())}
        results.append(torch.cat(list(pair_sorted.values()), dim=0))
    return results

def loss_of_one_batch(batch, model, device, symmetrize_batch=False, use_amp=False):
    view1, view2 = batch
    for view in batch:
        for name in 'img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres'.split():  # pseudo_focal
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = infer.make_batch_symmetric(batch)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        pred1, pred2 = model(view1, view2)
    return {view1['idx'][0]: pred1, view2['idx'][0]: pred2}
