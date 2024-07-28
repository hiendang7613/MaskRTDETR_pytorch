import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
import numpy as np

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', alpha=False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = BaseConv(ch_in, ch_out, ksize=3, stride=1, bias=False, act=None)
        self.conv2 = BaseConv(ch_in, ch_out, ksize=1, stride=1, bias=False, act=None)
        self.act = nn.ReLU() if act == "relu" else nn.SiLU()  # Assuming "silu" is SiLU activation
        if alpha:
            self.alpha = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        else:
            self.alpha = None

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        if self.alpha:
            y = y1 + self.alpha * y2
        else:
            y = y1 + y2
        y = self.act(y)
        return y

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(in_channels=self.ch_in, out_channels=self.ch_out, kernel_size=3, stride=1, padding=1, groups=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        if bias is not None:
            self.conv.bias.data = bias
        delattr(self, 'conv1')
        delattr(self, 'conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        if self.alpha is not None:
            alpha = self.alpha.item()
            return kernel3x3 + alpha * self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + alpha * bias1x1
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, (1, 1, 1, 1))

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        std = torch.sqrt(running_var + eps)
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

class BaseConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.bn(self.conv(x))
        y = x * torch.sigmoid(x)
        return y


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(* [
            RepVggBlock(
                hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(
                hidden_channels,
                out_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output



class MultiHeadAttention(nn.Module):
   

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.Parameter(torch.Tensor(embed_dim, 3 * embed_dim))
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
            self.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            self.in_proj_bias.data.zero_()
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(self.kdim, embed_dim)
            self.v_proj = nn.Linear(self.vdim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                p.data.zero_()

    def compute_qkv(self, tensor, index):
        if self._qkv_same_embed_dim:
            tensor = F.linear(
                tensor,
                self.in_proj_weight[:, index * self.embed_dim:(index + 1) * self.embed_dim],
                self.in_proj_bias[index * self.embed_dim:(index + 1) * self.embed_dim] if self.in_proj_bias is not None else None)
        else:
            tensor = getattr(self, self._type_list[index])(tensor)
        tensor = tensor.view(tensor.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        return tensor

    def forward(self, query, key=None, value=None, attn_mask=None):
        key = query if key is None else key
        value = query if value is None else value

        q, k, v = (self.compute_qkv(t, i) for i, t in enumerate([query, key, value]))

        product = torch.matmul(q, k.transpose(-2, -1))
        scaling = float(self.head_dim)**-0.5
        product = product * scaling

        if attn_mask is not None:
            product += attn_mask

        weights = F.softmax(product, dim=-1)
        if self.dropout > 0:
            weights = F.dropout(weights, p=self.dropout, training=self.training)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(out.size(0), -1, self.embed_dim)
        out = self.out_proj(out)

        if self.need_weights:
            return out, weights
        else:
            return out

class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=attn_dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(act_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = getattr(F, activation)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor + pos_embed if pos_embed is not None else tensor

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        src = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(src, src, src, attn_mask=src_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = residual + src
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 eval_size=None):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)))

        # encoder transformer
        self.encoder = nn.ModuleList([
            TransformerEncoder(encoder_layer, num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = torch.matmul(grid_w.flatten()[:, None], omega[None])
        out_h = torch.matmul(grid_h.flatten()[:, None], omega[None])

        return torch.cat(
            [
                torch.sin(out_w), torch.cos(out_w), torch.sin(out_h),
                torch.cos(out_h)
            ],
            dim=1).unsqueeze(0)

    def forward(self, feats, for_mot=False, is_teacher=False):
        assert len(feats) == len(self.in_channels)
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).transpose(1, 2)
                if self.training or self.eval_size is None or is_teacher:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed.to(src_flatten))
                proj_feats[enc_ind] = memory.transpose(1, 2).view(
                    -1, self.hidden_dim, h, w)

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = F.interpolate(
                feat_heigh, scale_factor=2., mode="nearest")
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.cat(
                    [upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat(
                [downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'feat_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            {'channels': self.hidden_dim, 'stride': self.feat_strides[idx]}
            for idx in range(len(self.in_channels))
        ]


class MaskFeatFPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 256, 256],
                 fpn_strides=[32, 16, 8],
                 feat_channels=256,
                 dropout_ratio=0.0,
                 out_channels=256,
                 align_corners=False,
                 act='swish'):
        super(MaskFeatFPN, self).__init__()
        assert len(in_channels) == len(fpn_strides)
        reorder_index = np.argsort(fpn_strides, axis=0)
        in_channels = [in_channels[i] for i in reorder_index]
        fpn_strides = [fpn_strides[i] for i in reorder_index]
        assert min(fpn_strides) == fpn_strides[0]
        self.reorder_index = reorder_index
        self.fpn_strides = fpn_strides
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)

        self.scale_heads = nn.ModuleList()  # Use ModuleList instead of LayerList
        for i in range(len(fpn_strides)):
            head_length = max(1, int(np.log2(fpn_strides[i]) - np.log2(fpn_strides[0])))
            scale_head = []
            for k in range(head_length):
                in_c = in_channels[i] if k == 0 else feat_channels
                scale_head.append(
                    nn.Sequential(
                        BaseConv(in_c, feat_channels, 3, 1, act=act)
                    )
                )
                if fpn_strides[i] != fpn_strides[0]:
                    scale_head.append(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners)
                    )

            self.scale_heads.append(nn.Sequential(*scale_head))

        self.output_conv = BaseConv(feat_channels, out_channels, 3, 1, act=act)

    def forward(self, inputs):
        x = [inputs[i] for i in self.reorder_index]

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.fpn_strides)):
            output = output + F.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )

        if self.dropout_ratio > 0:
            output = self.dropout(output)
        output = self.output_conv(output)
        return output

class MaskHybridEncoder(HybridEncoder):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size', 'num_prototypes']
    __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 feat_strides=[4, 8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[3],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 num_prototypes=32,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 mask_feat_channels=[64, 64],
                 act='silu',
                 trt=False,
                 eval_size=None):
        assert len(in_channels) == len(feat_strides)
        x4_feat_dim = in_channels.pop(0)
        x4_feat_stride = feat_strides.pop(0)
        use_encoder_idx = [i - 1 for i in use_encoder_idx]
        assert x4_feat_stride == 4

        super(MaskHybridEncoder, self).__init__(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            use_encoder_idx=use_encoder_idx,
            num_encoder_layers=num_encoder_layers,
            encoder_layer=encoder_layer,
            pe_temperature=pe_temperature,
            expansion=expansion,
            depth_mult=depth_mult,
            act=act,
            trt=trt,
            eval_size=eval_size)

        self.mask_feat_head = MaskFeatFPN(
            [hidden_dim] * len(feat_strides),
            feat_strides,
            feat_channels=mask_feat_channels[0],
            out_channels=mask_feat_channels[1],
            act=act)
        self.enc_mask_lateral = BaseConv(
            x4_feat_dim, mask_feat_channels[1], 3, 1, act=act)
        self.enc_mask_output = nn.Sequential(
            BaseConv(
                mask_feat_channels[1],
                mask_feat_channels[1], 3, 1, act=act),
            nn.Conv2d(mask_feat_channels[1], num_prototypes, 1))

    def forward(self, feats, for_mot=False, is_teacher=False):
        x4_feat = feats.pop(0)

        enc_feats = super(MaskHybridEncoder, self).forward(
            feats, for_mot=for_mot, is_teacher=is_teacher)

        mask_feat = self.mask_feat_head(enc_feats)
        mask_feat = F.interpolate(
            mask_feat,
            scale_factor=2,
            mode='bilinear',
            align_corners=False)
        mask_feat += self.enc_mask_lateral(x4_feat)
        mask_feat = self.enc_mask_output(mask_feat)

        return enc_feats, mask_feat

if __name__ == '__main__':
  d_model= 256
  nhead= 8
  dim_feedforward= 1024
  dropout= 0.0
  activation= 'gelu'
  attn_dropout= None
  act_dropout= None
  normalize_before= False

  encoder_layer= TransformerLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=dim_feedforward,
      dropout=dropout,
      activation=activation,
      attn_dropout=attn_dropout,
      act_dropout=act_dropout,
      normalize_before=normalize_before
  )

  in_channels= [128, 512, 1024, 2048]
  feat_strides= [4, 8, 16, 32]
  hidden_dim= 256
  use_encoder_idx= [3]
  num_encoder_layers= 1
  # encoder_layer= encoder_layer
  num_prototypes= 32
  pe_temperature= 10000
  expansion= 1.0
  depth_mult= 1.0
  mask_feat_channels= [64, 64]
  act= 'silu'
  trt= False
  eval_size= [640, 640]

  maskHybridEncoder = MaskHybridEncoder(
      in_channels=in_channels,
      feat_strides=feat_strides,
      hidden_dim=hidden_dim,
      use_encoder_idx=use_encoder_idx,
      num_encoder_layers=num_encoder_layers,
      encoder_layer=encoder_layer,
      num_prototypes=num_prototypes,
      pe_temperature=pe_temperature,
      expansion=expansion,
      depth_mult=depth_mult,
      mask_feat_channels=mask_feat_channels,
      act=act,
      trt=trt,
      eval_size=eval_size
  )
