import torch
import torch.nn as nn
import math
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
import copy
import numpy as np
from torch.nn import init
from utils import gather_nd

class MaskTransformerDecoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 decoder_layer,
                 num_layers,
                 eval_idx=-1,
                 eval_topk=100):
        super(MaskTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.eval_topk = eval_topk

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                mask_feat,
                bbox_head,
                score_head,
                query_pos_head,
                mask_query_head,
                dec_norm,
                attn_mask=None,
                memory_mask=None,
                query_pos_head_inv_sig=False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_masks = []
        ref_points_detach = torch.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            if not query_pos_head_inv_sig:
                query_pos_embed = query_pos_head(ref_points_detach)
            else:
                query_pos_embed = query_pos_head(inverse_sigmoid(ref_points_detach))

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = torch.sigmoid(bbox_head(output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                logits_, masks_ = _get_pred_class_and_mask(
                    output, mask_feat, dec_norm, score_head, mask_query_head)
                dec_out_logits.append(logits_)
                dec_out_masks.append(masks_)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(torch.sigmoid(bbox_head(output) + inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                logits_, masks_ = _get_pred_class_and_mask(
                    output, mask_feat, dec_norm, score_head, mask_query_head)
                dec_out_logits.append(logits_)
                dec_out_masks.append(masks_)
                dec_out_bboxes.append(inter_ref_bbox)
                return (torch.stack(dec_out_bboxes),
                        torch.stack(dec_out_logits),
                        torch.stack(dec_out_masks))

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return (torch.stack(dec_out_bboxes),
                torch.stack(dec_out_logits),
                torch.stack(dec_out_masks))

def deformable_attention_core_func(value, value_spatial_shapes,
                                   value_level_start_index, sampling_locations,
                                   attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [int(h * w) for h, w in value_spatial_shapes]
    value_list = torch.split(value, split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).reshape(bs * n_head, Len_q, n_points, 2)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * n_head, 1, Len_q, n_levels * n_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs, n_head * c, Len_q)

    return output.transpose(1, 2)


class MSDeformableAttention(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 lr_mult=0.1):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # try:
        #     # use cuda op
        #     from deformable_detr_ops import ms_deformable_attn
        # except ImportError:
        #     # use fallback implementation
        #     from .utils import deformable_attention_core_func as ms_deformable_attn
        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).view(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data.copy_(grid_init.view(-1))

        # attention_weights
        constant_(self.attention_weights.weight, 0.0)
        constant_(self.attention_weights.bias, 0.0)

        # proj
        xavier_uniform_(self.value_proj.weight)
        constant_(self.value_proj.bias, 0.0)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.output_proj.bias, 0.0)

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_level_start_index,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1,1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (Tensor): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (Tensor(int64)): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]
        assert int(torch.prod(torch.tensor(value_spatial_shapes, dtype=torch.int32), 1).sum()) == Len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.to(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.view(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.flip(value_spatial_shapes, [1]).view(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.view(
                bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer.to(sampling_offsets.dtype)
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but got {} instead.".format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight)
        xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        if attn_mask is not None:
            attn_mask = torch.where(
                attn_mask.bool(),
                torch.zeros_like(attn_mask, dtype=tgt.dtype),
                torch.full_like(attn_mask, float("-inf"), dtype=tgt.dtype)).to(tgt)
            
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed), reference_points, memory,
            memory_spatial_shapes, memory_level_start_index, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


def constant_(tensor, value=0.):
    return _no_grad_fill_(tensor, value)

def _no_grad_fill_(tensor, value=0.):
    with torch.no_grad():
        tensor.fill_(value)
    return tensor


def _get_pred_class_and_mask(query_embed,
                             mask_feat,
                             dec_norm,
                             score_head,
                             mask_query_head):
    out_query = dec_norm(query_embed)
    out_logits = score_head(out_query)
    mask_query_embed = mask_query_head(out_query)
    batch_size, mask_dim, _ = mask_query_embed.shape
    _, _, mask_h, mask_w = mask_feat.shape
    out_mask = torch.bmm(
        mask_query_embed, mask_feat.flatten(2)).reshape(
        [batch_size, mask_dim, mask_h, mask_w])
    return out_logits, out_mask



def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def bbox_xyxy_to_cxcywh(x):

    x1, y1, x2, y2 = x.chunk(4, dim=-1)
    return torch.cat(
        [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], dim=-1)

def mask_to_box_coordinate(mask,
                           normalize=False,
                           format="xyxy",
                           dtype=torch.int32):
    assert mask.ndim == 4
    assert format in ["xyxy", "xywh"]

    h, w = mask.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(
            end=h, dtype=dtype), torch.arange(
                end=w, dtype=dtype))
    x = x.to(mask)
    y = y.to(mask)

    x_mask = x * mask.to(x.dtype)
    x_max = x_mask.flatten(-2).max(-1)[0] + 1
    x_min = torch.where(mask.to(bool), x_mask,
                         torch.tensor(1e8)).flatten(-2).min(-1)[0]

    y_mask = y * mask.to(y.dtype)
    y_max = y_mask.flatten(-2).max(-1)[0] + 1
    y_min = torch.where(mask.to(bool), y_mask,
                         torch.tensor(1e8)).flatten(-2).min(-1)[0]
    out_bbox = torch.stack([x_min, y_min, x_max, y_max], axis=-1)
    mask = mask.any(axis=[2, 3]).unsqueeze(2)
    out_bbox = out_bbox * mask.to(out_bbox.dtype)
    if normalize:
        out_bbox /= torch.tensor([w, h, w, h]).to(dtype).to(mask)

    return out_bbox if format == "xyxy" else bbox_xyxy_to_cxcywh(out_bbox)


def get_denoising_training_group(targets,
                                 num_classes,
                                 num_queries,
                                 class_embed,
                                 num_denoising=100,
                                 label_noise_ratio=0.5,
                                 box_noise_scale=1.0):
    if num_denoising <= 0:
        return None, None, None, None
    num_gts = [len(t) for t in targets["gt_class"]]
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(targets["gt_class"])
    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4])
    pad_gt_mask = torch.zeros([bs, max_gt_num])
    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets["gt_class"][i].squeeze(-1)
            input_query_bbox[i, :num_gt] = targets["gt_bbox"][i]
            pad_gt_mask[i, :num_gt] = 1

    input_query_class = input_query_class.repeat(1, num_group)
    input_query_bbox = input_query_bbox.repeat(1, num_group, 1)
    pad_gt_mask = pad_gt_mask.repeat(1, num_group)

    dn_positive_idx = pad_gt_mask.nonzero(as_tuple=False)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * num_group)

    if label_noise_ratio > 0:
        input_query_class = input_query_class.flatten()
        pad_gt_mask = pad_gt_mask.flatten()
        # half of bbox prob
        mask = torch.rand(input_query_class.shape) < (label_noise_ratio * 0.5)
        chosen_idx = torch.nonzero(mask * pad_gt_mask, as_tuple=False).squeeze(-1)
        # randomly put a new one here
        new_label = torch.randint(0, num_classes, chosen_idx.shape, dtype=input_query_class.dtype)
        input_query_class.index_copy_(0, chosen_idx, new_label)
        input_query_class = input_query_class.reshape([bs, num_denoising])
        pad_gt_mask = pad_gt_mask.reshape([bs, num_denoising])

    if box_noise_scale > 0:
        diff = torch.cat([input_query_bbox[..., 2:] * 0.5, input_query_bbox[..., 2:]], dim=-1) * box_noise_scale
        diff *= (torch.rand(input_query_bbox.shape) * 2.0 - 1.0)
        input_query_bbox = input_query_bbox + diff
        input_query_bbox = inverse_sigmoid(input_query_bbox)

    class_embed = torch.cat([class_embed, torch.zeros([1, class_embed.shape[-1]]).to(class_embed)])
    input_query_class = class_embed[input_query_class.flatten()]
    input_query_class = input_query_class.reshape([bs, num_denoising, -1])

    tgt_size = num_denoising + num_queries
    attn_mask = torch.ones([tgt_size, tgt_size], dtype=torch.bool)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = False
    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * i:max_gt_num * (i + 1), max_gt_num * (i + 1):num_denoising] = False
        if i == num_group - 1:
            attn_mask[max_gt_num * i:max_gt_num * (i + 1), :max_gt_num * i] = False
        else:
            attn_mask[max_gt_num * i:max_gt_num * (i + 1), max_gt_num * (i + 1):num_denoising] = False
            attn_mask[max_gt_num * i:max_gt_num * (i + 1), :max_gt_num * i] = False
    attn_mask = ~attn_mask
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_class, input_query_bbox, attn_mask, dn_meta


class MaskRTDETR(nn.Module):
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_prototypes=32,
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.4,
                 box_noise_scale=0.4,
                 learnt_init_query=False,
                 query_pos_head_inv_sig=False,
                 mask_enhanced=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        super(MaskRTDETR, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.mask_enhanced = mask_enhanced
        self.eval_size = eval_size

        # backbone feature projection
        self._build_input_proj_layer(backbone_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points)
        self.decoder = MaskTransformerDecoder(hidden_dim, decoder_layer,
                                              num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.query_pos_head_inv_sig = query_pos_head_inv_sig

        # mask embedding
        self.mask_query_head = MLP(hidden_dim, hidden_dim, num_prototypes, num_layers=3)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim))

        # decoder norm layer
        self.dec_norm = nn.LayerNorm(hidden_dim)

        # shared prediction head
        self.score_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self._reset_parameters()

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.score_head)
        constant_(self.score_head.bias, bias_cls)
        constant_(self.bbox_head.layers[-1].weight)
        constant_(self.bbox_head.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape],
                'feat_strides': [i.stride for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.hidden_dim)))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(self.hidden_dim)))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose(1, 2))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.cat(feat_flatten, 1)
        level_start_index.pop()
        return feat_flatten, spatial_shapes, level_start_index

    def forward(self, feats, pad_mask=None, gt_meta=None, is_teacher=False):
        enc_feats, mask_feat = feats
        # input projection and embedding
        memory, spatial_shapes, level_start_index = self._get_encoder_input(enc_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = get_denoising_training_group(
                gt_meta, self.num_classes, self.num_queries, self.denoising_class_embed.weight,
                self.num_denoising, self.label_noise_ratio, self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_out, init_out = self._get_decoder_input(
            memory, mask_feat, spatial_shapes, denoising_class, denoising_bbox_unact, is_teacher)

        # decoder
        out_bboxes, out_logits, out_masks = self.decoder(
            target, init_ref_points_unact, memory, spatial_shapes,
            level_start_index, mask_feat, self.bbox_head, self.score_head,
            self.query_pos_head, self.mask_query_head, self.dec_norm,
            attn_mask=attn_mask, memory_mask=None, query_pos_head_inv_sig=self.query_pos_head_inv_sig)

        return out_logits, out_bboxes, out_masks, enc_out, init_out, dn_meta

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_size[0] / s), int(self.eval_size[1] / s)] for s in self.feat_strides]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, dtype=dtype),
                torch.arange(w, dtype=dtype))
            grid_xy = torch.stack([grid_x, grid_y], -1)

            valid_WH = torch.tensor([h, w], dtype=dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.cat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = torch.cat(anchors, 1)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self, memory, mask_feat, spatial_shapes, denoising_class=None,
                           denoising_bbox_unact=None, is_teacher=False):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_size is None or is_teacher:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = torch.where(valid_mask.to(memory).bool(), memory, torch.tensor(0).to(memory))
        output_memory = self.enc_output(memory)

        enc_logits_unact = self.score_head(output_memory)
        enc_bboxes_unact = self.bbox_head(output_memory) + anchors.to(memory)

        # get topk index
        _, topk_ind = torch.topk(enc_logits_unact.max(-1)[0], self.num_queries, dim=1)
        batch_ind = torch.arange(bs).unsqueeze(-1).repeat(1, self.num_queries).to(memory)
        topk_ind = torch.stack([batch_ind, topk_ind], dim=-1)

        # # extract content and position query embedding
        # target = torch.gather(output_memory, 1, topk_ind)
        # reference_points_unact = torch.gather(enc_bboxes_unact, 1, topk_ind)  # unsigmoided.
        
        # extract content and position query embedding
        # output_memory = paddle.to_tensor(output_memory.detach().numpy())
        # topk_ind = paddle.to_tensor(topk_ind.detach().numpy())
        # enc_bboxes_unact = paddle.to_tensor(enc_bboxes_unact.detach().numpy())
        
        target = gather_nd(output_memory, topk_ind)
        # print(target.shape)
        reference_points_unact = gather_nd(enc_bboxes_unact,
                                                  topk_ind)  # unsigmoided.

        # target = torch.tensor(target.numpy())
        # reference_points_unact = torch.tensor(reference_points_unact.numpy())

        
        
        
        
        # get encoder output: {logits, bboxes, masks}
        enc_out_logits, enc_out_masks = _get_pred_class_and_mask(
            target, mask_feat, self.dec_norm, self.score_head, self.mask_query_head)
        enc_out_bboxes = torch.sigmoid(reference_points_unact)
        enc_out = (enc_out_logits, enc_out_bboxes, enc_out_masks)

        # concat denoising query
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            target = target.detach()
        if denoising_class is not None:
            target = torch.cat([denoising_class, target], 1)
        if self.mask_enhanced:
            # use mask-enhanced anchor box initialization
            reference_points = mask_to_box_coordinate(enc_out_masks > 0, normalize=True, format="xywh")
            reference_points_unact = inverse_sigmoid(reference_points)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.cat([denoising_bbox_unact.to(memory), reference_points_unact], 1)

        # direct prediction from the matching and denoising part in the beginning
        if self.training and denoising_class is not None:
            init_out_logits, init_out_masks = _get_pred_class_and_mask(
                target, mask_feat, self.dec_norm, self.score_head, self.mask_query_head)
            init_out_bboxes = torch.sigmoid(reference_points_unact)
            init_out = (init_out_logits, init_out_bboxes, init_out_masks)
        else:
            init_out = None

        return target, reference_points_unact.detach(), enc_out, init_out

def inverse_sigmoid(x, eps=1e-5):
    x = torch.clamp(x, min=0., max=1.)
    return torch.log(torch.clamp(x, min=eps) / torch.clamp(1 - x, min=eps))

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        self._reset_parameters()

    def _reset_parameters(self):
        for l in self.layers:
            self.linear_init_(l)

    def linear_init_(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def linear_init_(module):
    """Initialize weights and biases for a linear layer."""
    if isinstance(module, nn.Linear):
        # Tính toán bound cho phân phối đồng đều
        bound = 1 / math.sqrt(module.weight.shape[1])
        # Khởi tạo trọng số với phân phối đồng đều
        init.uniform_(module.weight, -bound, bound)
        # Khởi tạo bias (nếu có) với phân phối đồng đều
        if module.bias is not None:
            init.uniform_(module.bias, -bound, bound)


if __name__ == '__main__':

  num_classes=80
  hidden_dim=256
  num_queries=300
  position_embed_type='sine'
  backbone_feat_channels=[256, 256, 256]
  feat_strides=[8, 16, 32]
  num_prototypes=32
  num_levels=3
  num_decoder_points=4
  nhead=8
  num_decoder_layers=6
  dim_feedforward=1024
  dropout=0.
  activation="relu"
  num_denoising=100
  label_noise_ratio=0.4
  box_noise_scale=0.4
  learnt_init_query=False
  query_pos_head_inv_sig=False
  mask_enhanced=True
  eval_size=None
  eval_idx=-1
  eps=1e-2

  maskRTDETR = MaskRTDETR(
      num_classes=num_classes,
      hidden_dim=hidden_dim,
      num_queries=num_queries,
      position_embed_type=position_embed_type,
      backbone_feat_channels=backbone_feat_channels,
      feat_strides=feat_strides,
      num_prototypes=num_prototypes,
      num_levels=num_levels,
      num_decoder_points=num_decoder_points,
      nhead=nhead,
      num_decoder_layers=num_decoder_layers,
      dim_feedforward=dim_feedforward,
      dropout=dropout,
      activation=activation,
      num_denoising=num_denoising,
      label_noise_ratio=label_noise_ratio,
      box_noise_scale=box_noise_scale,
      learnt_init_query=learnt_init_query,
      query_pos_head_inv_sig=query_pos_head_inv_sig,
      mask_enhanced=mask_enhanced,
      eval_size=eval_size,
      eval_idx=eval_idx,
      eps=eps
  )
