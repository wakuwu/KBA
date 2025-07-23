import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import pprint
from datetime import datetime, timezone, timedelta


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_utc8_time():
    UTF8_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_now = utc_now.astimezone(UTF8_TZ)
    return beijing_now.strftime('%Y-%m-%d-%H-%M-%f')


def print_to_console_and_file(text, file_path):
    # Open the file in append mode
    with open(file_path, "a") as f:
        # Print to console
        print(text)
        # Print to file
        f.write(str(text) + '\n')

def print_params_to_console_and_file(text, file_path):
    text_format = pprint.pformat(text, sort_dicts=False)
    # Open the file in append mode
    with open(file_path, "a") as f:
        # Print to console
        print(text_format)
        # Print to file
        f.write(text_format + '\n')

def serialize_loss(loss_dict):
    results = []
    keys_list = sorted(loss_dict.keys())
    if "loss_combined" in keys_list:
        keys_list.remove("loss_combined")
        keys_list.insert(0, "loss_combined")
    loss_dict_sorted = {key: loss_dict[key] for key in keys_list}
    for loss_name, loss_value in loss_dict_sorted.items():
        loss_string = f"{loss_name}: {loss_value.data.item(): >8.5f}"
        results.append(loss_string)
    return ",    ".join(results)

def heatmap(sample, normalized=False, is_show=False, out_fp=None, resolution_rate=2):
    """
    from utils import heatmap
    heatmap(feat1.detach().cpu(), normalized=True, is_show=True)
    The deeper the color, the smaller the value; the lighter and brighter the color, the larger the value.
    """
    sample = sample.squeeze()
    if normalized:
        sample = (sample-sample.min()) / (sample.max()-sample.min())
    fig = plt.figure("", frameon=False)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    width, height = sample.shape[1], sample.shape[0]
    fig.set_size_inches((width + 1e-2) / dpi, (height + 1e-2) / dpi)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    sns.heatmap(sample.cpu().numpy(), ax=ax, vmin=0, vmax=1, cbar=False,
                cmap=sns.color_palette("rocket", n_colors=20))
    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    if sys.platform == 'darwin':
        width, height = canvas.get_width_height(physical=True)
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')

    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute((2, 0, 1))[:, :, None, :, None]
    img_tensor = img_tensor.expand(-1, -1, resolution_rate, -1, resolution_rate)
    img_tensor = img_tensor.reshape(3, height * resolution_rate, width * resolution_rate)
    img_tensor = img_tensor.permute((1, 2, 0))
    img = img_tensor.numpy()

    if is_show:
        plt.show(dpi=100)
        plt.close()
        return
    if out_fp is not None:
        plt.imsave(out_fp, img)
        plt.close()
        return
    return img

def heatmap_direction(sample, normalized=False, is_show=False, out_fp=None, start_point=None, direction_vector=None, length=20, arrow_width=2):
    """
    from utils import heatmap
    heatmap(feat1.detach().cpu(), normalized=True, is_show=True, start_point=(50, 50), direction_vector=(1, 1), length=20)
    """
    sample = sample.squeeze()
    if normalized:
        sample = (sample - sample.min()) / (sample.max() - sample.min())
    fig = plt.figure("", frameon=False)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    width, height = sample.shape[1], sample.shape[0]
    fig.set_size_inches((width + 1e-2) / dpi, (height + 1e-2) / dpi)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    sns.heatmap(sample.cpu().numpy(), ax=ax, vmin=0, vmax=1, cbar=False,
                cmap=sns.color_palette("rocket", n_colors=20))

    # Draw arrow
    if start_point is not None and direction_vector is not None:
        x0, y0 = start_point
        dx, dy = direction_vector
        direction = torch.tensor([dx, dy], dtype=torch.float32)
        direction = direction / torch.norm(direction) * length
        x1, y1 = x0 + direction[0].item(), y0 + direction[1].item()
        plt.arrow(x0, y0, direction[0], direction[1], head_width=arrow_width, head_length=arrow_width*2,
                  fc='#43a047', ec='#43a047', width=arrow_width/10, overhang=0.4)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    if sys.platform == 'darwin':
        width, height = canvas.get_width_height(physical=True)
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')

    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute((2, 0, 1))[:, :, None, :, None]
    resolution_rate_w = 2000 // width
    resolution_rate_h = 720 // height
    img_tensor = img_tensor.expand(-1, -1, resolution_rate_h, -1, resolution_rate_w)
    img_tensor = img_tensor.reshape(3, height * resolution_rate_h, width * resolution_rate_w)
    img_tensor = img_tensor.permute((1, 2, 0))
    img = img_tensor.numpy()

    if is_show:
        plt.show(dpi=100)
    if out_fp is not None:
        plt.imsave(out_fp, img)
    plt.close()