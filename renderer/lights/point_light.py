import random
import numpy as np
from pytorch3d.renderer import PointLights


class PointLight(object):
    def __init__(self, color_scale_range=((0.5, 1.0), (0.0, 0.1), (0.0, 0.1)), color_delta=(-0.1, 0.1),
                 location_rage=((-10, 10), (0, 10), (-10, 10)), device="cpu"):
        """
        :param color_scale_range: ((ambient_color), (specular_color), (diffuse_color))
        """
        super().__init__()
        self.device = device
        self.color_scale_range = color_scale_range
        self.color_delta = color_delta
        self.location_rage = location_rage

    def forward(self, **kwargs):
        light_color = np.array([random.uniform(*self.color_scale_range[0]),
                                random.uniform(*self.color_scale_range[1]),
                                random.uniform(*self.color_scale_range[2])])[..., None].repeat(3, axis=-1) * \
                      np.array(((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)), dtype=np.float32) \
                      + np.random.random(size=(3, 3)) * (self.color_delta[1] - self.color_delta[0]) + self.color_delta[
                          0]
        light_color[0] = np.clip(light_color[0], a_min=0.0, a_max=self.color_scale_range[0][-1])
        light_color[1] = np.clip(light_color[1], a_min=0.0, a_max=self.color_scale_range[1][-1])
        light_color[2] = np.clip(light_color[2], a_min=0.0, a_max=self.color_scale_range[2][-1])
        location = ((random.uniform(*self.location_rage[0]), random.uniform(*self.location_rage[1]),
                     random.uniform(*self.location_rage[2])),)
        light = PointLights(light_color[0:1], light_color[1:2], light_color[2:], location, self.device)
        return light


if __name__ == '__main__':
    p = PointLight()
    for _ in range(10):
        p.forward()
