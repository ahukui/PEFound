from torch import nn
from .spatial_pooling_projector import SpatialPoolingProjector

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, *args, **kwargs):
        return x
    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class FullLinear(nn.Module):
    def __init__(self, config):
        super(FullLinear, self).__init__()
        self.linear = nn.Linear(config.mm_hidden_size, config.hidden_size)
    def forward(self, x):
        x = self.linear(x)
        return x
    @property
    def proj_out_num(self):
        num = 2048
        return num


def build_mm_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type')

    if projector_type == 'linear':
        return FullLinear(config)

    elif projector_type == 'spp':
        return SpatialPoolingProjector(image_size=config.image_size,
                                        patch_size=config.patch_size,
                                        in_dim=config.mm_hidden_size,
                                        out_dim=config.hidden_size,
                                        layer_type=config.proj_layer_type,
                                        layer_num=config.proj_layer_num,
                                        pooling_type=config.proj_pooling_type,
                                        pooling_size=config.proj_pooling_size)


    elif projector_type == 'identity':
        return IdentityMap()
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')