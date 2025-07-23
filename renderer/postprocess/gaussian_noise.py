import torch


class GaussianNoise:
    def __init__(self, mean=0.0, stddev=5.0, device="cpu", **kwargs):
        """
        初始化高斯噪声添加器
        :param mean: 高斯噪声的均值
        :param stddev: 高斯噪声的标准差
        """
        self.device = device
        self.mean = mean
        self.stddev = stddev / 255.0  # 将标准差缩放到0-1范围内

    def forward(self, image):
        # 生成高斯噪声
        noise = torch.randn(image.size(), device=self.device) * self.stddev + self.mean
        # 添加噪声到图像
        noisy_image = image + noise
        # 确保图像值在有效范围内
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
        return noisy_image
