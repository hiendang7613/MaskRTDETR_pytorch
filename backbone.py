import torch
from torch import nn
from utils import reset_bn, _freeze_norm, kaiming_normal_, ones_, zeros_
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, use_act=True, use_lab=False, lr_mult=1.0):
        super(ConvBNAct, self).__init__()
        self.use_act = use_act
        self.use_lab = use_lab

        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding if isinstance(padding, str) else (kernel_size - 1) // 2,
            groups=groups,
            bias=False)  # PyTorch does not include bias in BatchNorm, set bias=False

        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels)

        # Activation function (ReLU)
        self.act = nn.ReLU() if use_act else None

        # Learnable affine block
        if self.use_lab:
            self.lab = LearnableAffineBlock(lr_mult=lr_mult)
        else:
            self.lab = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        if self.use_act:
            x = self.act(x)

        if self.use_lab and self.lab is not None:
            x = self.lab(x)

        return x

class MaxPool2dSamePadding(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super(MaxPool2dSamePadding, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        pad_h = max(self.kernel_size - self.stride, 0)
        pad_w = max(self.kernel_size - self.stride, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return F.max_pool2d(x, self.kernel_size, self.stride, padding=0)


class StemBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, use_lab=False, lr_mult=1.0):
        super(StemBlock, self).__init__()
        self.stem1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.stem2a = ConvBNAct(
            in_channels=mid_channels,
            out_channels=mid_channels // 2,
            kernel_size=2,
            stride=1,
            padding='same',  # For PyTorch, padding="SAME" is equivalent to padding=0
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.stem2b = ConvBNAct(
            in_channels=mid_channels // 2,
            out_channels=mid_channels,
            kernel_size=2,
            stride=1,
            padding='same',
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.stem3 = ConvBNAct(
            in_channels=mid_channels * 2,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.stem4 = ConvBNAct(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.pool = MaxPool2dSamePadding(
            kernel_size=2, stride=1)  # ceil_mode=True is not available in PyTorch

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2a(x)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], 1)  # torch.cat instead of paddle.concat
        x = self.stem3(x)
        x = self.stem4(x)

        return x

class HG_Stage(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 block_num,
                 layer_num=6,
                 downsample=True,
                 light_block=True,
                 kernel_size=3,
                 use_lab=False,
                 lr_mult=1.0):
        super(HG_Stage, self).__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                use_act=False,
                use_lab=use_lab,
                lr_mult=lr_mult)

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_channels=in_channels if i == 0 else out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    layer_num=layer_num,
                    identity=False if i == 0 else True,
                    light_block=light_block,
                    use_lab=use_lab,
                    lr_mult=lr_mult))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 use_lab=False,
                 lr_mult=1.0):
        super(LightConvBNAct, self).__init__()
        self.conv1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.conv2 = ConvBNAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=out_channels,
            use_act=True,
            use_lab=use_lab,
            lr_mult=lr_mult)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class HG_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 layer_num=6,
                 identity=False,
                 light_block=True,
                 use_lab=False,
                 lr_mult=1.0):
        super(HG_Block, self).__init__()
        self.identity = identity

        self.layers = nn.ModuleList()
        block_type = LightConvBNAct if light_block else ConvBNAct
        for i in range(layer_num):
            self.layers.append(
                block_type(
                    in_channels=in_channels if i == 0 else mid_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    use_lab=use_lab,
                    lr_mult=lr_mult))

        # Feature aggregation
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(
            in_channels=total_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult)
        self.aggregation_excitation_conv = ConvBNAct(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult)

    def forward(self, x):
        identity = x
        outputs = [x]

        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        x = torch.cat(outputs, dim=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)

        if self.identity:
            x += identity

        return x


class PPHGNetV2(nn.Module):
    """
    PPHGNetV2
    Args:
        arch: str. Specifies the architecture configuration ('L', 'X', 'H').
        use_lab: bool. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Controls the learning rate of different stages.
        return_idx: list. Indices of stages to return outputs from.
        freeze_stem_only: bool. Whether to freeze only the stem block.
        freeze_at: int. Indicates up to which stage (inclusive) to freeze layers.
        freeze_norm: bool. Whether to freeze batch normalization layers.
    Returns:
        model: nn.Module. PPHGNetV2 model based on the specified args.
    """

    arch_configs = {
        'L': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            }
        },
        'X': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            }
        },
        'H': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            }
        }
    }

    def __init__(self,
                 arch,
                 use_lab=False,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],
                 return_idx=[1, 2, 3],
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[arch]['stem_channels']
        stage_config = self.arch_configs[arch]['stage_config']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # stem
        self.stem = StemBlock(
            in_channels=stem_channels[0],
            mid_channels=stem_channels[1],
            out_channels=stem_channels[2],
            use_lab=use_lab,
            lr_mult=lr_mult_list[0])

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    lr_mult=lr_mult_list[i + 1]))

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            reset_bn(self, reset_func=_freeze_norm)


        self._init_weights()

    def _freeze_parameters(self, m):
        for p in m.parameters():
            p.requires_grad = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                ones_(m.weight)
                zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                zeros_(m.bias)

    @property
    def out_shape(self):
        return [
            {'channels': self._out_channels[i], 'stride': self._out_strides[i]}
            for i in self.return_idx
        ]

    def forward(self, inputs):
        x = inputs['image']
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


if __name__ == '__main__':
  arch= 'L'
  use_lab= False
  lr_mult_list= [0.0, 0.05, 0.05, 0.05, 0.05]
  return_idx= [0, 1, 2, 3]
  freeze_stem_only= True
  freeze_at= 0
  freeze_norm= True

  pPHGNetV2 = PPHGNetV2(arch,
    use_lab= use_lab,
    lr_mult_list= lr_mult_list,
    return_idx= return_idx,
    freeze_stem_only= freeze_stem_only,
    freeze_at= freeze_at,
    freeze_norm= freeze_norm)
