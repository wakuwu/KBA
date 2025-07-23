<div align="center">
  <img src="https://iccv.thecvf.com/static/core/img/iccv-navbar-logo.svg" alt="conference_logo" height="50">
  <h2>Kaleidoscopic Background Attack: Disrupting Pose Estimation <br> with Multi-Fold Radial Symmetry Textures</h2>
  <p align="center">
    <div style="line-height: 3;">
      <a href="https://scholar.google.com/citations?user=JY9oXVIAAAAJ&hl=en"><strong>Xinlong Ding</strong></a>
      ·
      <a href="https://scholar.google.com/citations?user=cDidt64AAAAJ&hl=en"><strong>Hongwei Yu</strong></a>
      ·
      <a href="https://scholar.google.com/citations?user=xWy8RZEAAAAJ&hl=en"><strong>Jiawei Li</strong></a>
      ·
      <a href=""><strong>Feifan Li</strong></a>
      ·
      <a href="https://scholar.google.com/citations?user=2HFt8mkAAAAJ&hl=en"><strong>Yu Shang</strong></a>
      ·
      <a href="https://scholar.google.com/citations?user=Cb29A3cAAAAJ&hl=en"><strong>Bochao Zou</strong></a>
      ·
      <a href="https://scholar.google.com/citations?user=32hwVLEAAAAJ&hl=en"><strong>Huimin Ma</strong></a>
      ·
      <a href="https://scholar.google.com/citations?user=A1gA9XIAAAAJ&hl=en"><strong>Jiansheng Chen</strong></a>
    </div>
    <br>
    <div style="line-height: 3;">
      <a href="https://arxiv.org/pdf/2507.10265">
        <img src='https://img.shields.io/badge/arXiv-2507.10265-b31b1b?logo=arxiv&logoColor=white' alt='arXiv'>
      </a>
      <a href='https://wakuwu.github.io/KBA/'>
        <img src='https://img.shields.io/badge/Project_Page-KBA-green?logo=googlechrome&logoColor=white' alt='Project Page'>
      </a>
      <img src="https://www.easycounter.com/counter.php?umiskky" border="0" alt="Hit Counter">
    </div>
    <div style="line-height: 3;">
      <b>University of Science and Technology Beijing &nbsp; | &nbsp;  Tsinghua University</b>
    </div>
  </p>
</div>

## 🏆 Overview
Official PyTorch implementation for ICCV 2025 Paper: **Kaleidoscopic Background Attack: Disrupting Pose Estimation with Multi-Fold Radial Symmetry Textures**.

## 🚀 Installation

To get started, please make sure your environment meets the following requirements:

- GPU with **at least 24GB** memory (we recommend ~33GB)
- **Ubuntu 22.04**, CUDA **12.4**
- Python >= **3.10**

### Step 1: Clone and Set Up Python Environment

```bash
git clone --recursive https://github.com/wakuwu/KBA
cd KBA

# Install uv (https://docs.astral.sh/uv/)
uv sync

# Install PyTorch3D (CUDA 12.4 compatible)
uv pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.4.1cu124
```

### Step 2: Color Management Setup

Install required tools:

```bash
sudo apt-get update
sudo apt-get install liblcms2-dev liblcms2-utils
```

Then download the [Adobe ICC Profiles](https://www.adobe.com/support/downloads/iccprofiles/icc_eula_win_dist.html), accept the license, and unzip the archive `AdobeICCProfilesCS4Win_bundler`. Copy the `CMYK` folder into the following directory:

```bash
data/cms/
```

### Step 3: Prepare Data

Download our preprocessed attack dataset:

```bash
wget https://huggingface.co/datasets/umiskky/KBA/resolve/main/data.tar
tar -xf data.tar
```

You can also optionally download:

- **OmniObject3D models** from [OpenXLab](https://openxlab.org.cn/datasets/OpenXDLab/OmniObject3D-New/tree/main/raw/raw_scans), placed under `data/dataset/`
- **HDRI environment maps** from [PolyHaven](https://polyhaven.com/hdris), placed under `data/environments/`

### Step 4: DUSt3R Configuration

Download the DUSt3R pretrained weights:

[DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth](https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth)

Place the downloaded file in the checkpoint directory:

```bash
third_party/dust3r/checkpoints/
```

### 🐳 Optional: Run with Docker

We also provide a pre-built Docker image for convenience:

```bash
docker pull ghcr.io/wakuwu/kba:latest
```

## 🔬 Evaluation & Rendering

After setup, you can test the system using the following commands:

```bash
# Run DUSt3R pose estimation and 3D reconstruction
python third_party/dust3r/demo.py \
    --weights third_party/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# Render multi-view images with specified kaleidoscopic background
python test.py
```

## 🛡️ Launch Attack

To launch the kaleidoscopic background attack:

```bash
python attack_dust3r.py
```

## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{ding2025kba,
    title   = {Kaleidoscopic Background Attack: Disrupting Pose Estimation with Multi-Fold Radial Symmetry Textures},
    author  = {Xinlong Ding, Hongwei Yu, Jiawei Li, Feifan Li, Yu Shang, Bochao Zou, Huimin Ma and Jiansheng Chen},
    journal = {arXiv preprint arXiv:2507.10265},
    year    = {2025}
}
```

## 📄 License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). See the [LICENSE](./LICENSE) file for more details.


## 🌠 Star History

<div style="width:600px;">
  <a href="https://www.star-history.com/#wakuwu/KBA&Date">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=wakuwu/KBA&type=Date&theme=dark" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=wakuwu/KBA&type=Date" />
      <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=wakuwu/KBA&type=Date" style="width:100%;" />
    </picture>
  </a>
</div>
