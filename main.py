from backbone import PPHGNetV2
from transformer_module import MaskRTDETR
from neck import MaskHybridEncoder, TransformerLayer
from head import MaskDINOHead
from loss import HungarianMatcher, MaskDINOLoss
from head import MaskDINOHead
from MRTDETR import MRTDETR

import torch
import numpy as np
import torch.nn.functional as F


# backbone
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




#head

matcher_coeff= {'class': 4, 'bbox': 5, 'giou': 2, 'mask': 5, 'dice': 5}
use_focal_loss= True
with_mask= True
num_sample_points= 12544
alpha= 0.25
gamma= 2.0


hungarian_matcher = HungarianMatcher(matcher_coeff, use_focal_loss, with_mask, num_sample_points, alpha, gamma)

num_classes= 80
matcher= hungarian_matcher
loss_coeff= {'class': 4, 'bbox': 5, 'giou': 2, 'mask': 5, 'dice': 5}
aux_loss= True
use_focal_loss= True
use_vfl= True
vfl_iou_type= 'mask'
num_sample_points= 12544
oversample_ratio= 3.0
important_sample_ratio= 0.75

loss = MaskDINOLoss(num_classes, matcher, loss_coeff, aux_loss, use_focal_loss, num_sample_points, oversample_ratio, important_sample_ratio)

maskDINOHead = MaskDINOHead(loss)


#neck
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

# transformer module
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

# MRTDETR

backbone = pPHGNetV2
transformer = maskRTDETR
detr_head = maskDINOHead
neck = maskHybridEncoder
post_process= None
post_process_semi= None
with_mask= True
exclude_post_process= False




model = MRTDETR(
    backbone=backbone,
    transformer=transformer,
    detr_head=detr_head,
    neck=neck,
    post_process=post_process,
    post_process_semi=post_process_semi,
    with_mask=with_mask,
    exclude_post_process=exclude_post_process
)


training = True
pytorch_model = model
pytorch_model.training = training
pytorch_model.transformer.training = training
pytorch_model.exclude_post_process = False
pytorch_model.detr_head.training = training

image_size = 224
label_id = 0
num_instance = 1
batch_size = 2
num_channels = 3

inputs = {
    'image': torch.rand((batch_size, num_channels, image_size, image_size)),
    'gt_bbox': [torch.rand((num_instance, 4)) for _ in range(batch_size)],
    'gt_class': [torch.tensor(np.array(num_instance*[label_id])) for _ in range(batch_size)],
    'gt_segm': [torch.randint(0,1,(num_instance, image_size, image_size), dtype=torch.float) for _ in range(batch_size)],
}
out = pytorch_model(inputs)
