import json
import torch
import numpy as np
from backbone import PPHGNetV2
from transformer_module import MaskRTDETR
from neck import MaskHybridEncoder, TransformerLayer
from head import MaskDINOHead
from loss import HungarianMatcher, MaskDINOLoss
from MRTDETR import MRTDETR

# Load configuration from JSON file
with open('/content/config.json', 'r') as config_file:
    config = json.load(config_file)

# Backbone configuration
backbone_config = config['backbone']
pPHGNetV2 = PPHGNetV2(
    arch=backbone_config['arch'],
    use_lab=backbone_config['use_lab'],
    lr_mult_list=backbone_config['lr_mult_list'],
    return_idx=backbone_config['return_idx'],
    freeze_stem_only=backbone_config['freeze_stem_only'],
    freeze_at=backbone_config['freeze_at'],
    freeze_norm=backbone_config['freeze_norm']
)

# Matcher configuration
matcher_config = config['matcher']
hungarian_matcher = HungarianMatcher(
    matcher_config['matcher_coeff'],
    matcher_config['use_focal_loss'],
    matcher_config['with_mask'],
    matcher_config['num_sample_points'],
    matcher_config['alpha'],
    matcher_config['gamma']
)

# Loss configuration
loss_config = config['loss']
loss = MaskDINOLoss(
    loss_config['num_classes'],
    hungarian_matcher,
    loss_config['loss_coeff'],
    loss_config['aux_loss'],
    loss_config['use_focal_loss'],
    loss_config['num_sample_points'],
    loss_config['oversample_ratio'],
    loss_config['important_sample_ratio']
)

maskDINOHead = MaskDINOHead(loss)

# Neck configuration
neck_config = config['neck']
encoder_layer = TransformerLayer(
    d_model=neck_config['d_model'],
    nhead=neck_config['nhead'],
    dim_feedforward=neck_config['dim_feedforward'],
    dropout=neck_config['dropout'],
    activation=neck_config['activation'],
    attn_dropout=neck_config['attn_dropout'],
    act_dropout=neck_config['act_dropout'],
    normalize_before=neck_config['normalize_before']
)

maskHybridEncoder = MaskHybridEncoder(
    in_channels=neck_config['in_channels'],
    feat_strides=neck_config['feat_strides'],
    hidden_dim=neck_config['hidden_dim'],
    use_encoder_idx=neck_config['use_encoder_idx'],
    num_encoder_layers=neck_config['num_encoder_layers'],
    encoder_layer=encoder_layer,
    num_prototypes=neck_config['num_prototypes'],
    pe_temperature=neck_config['pe_temperature'],
    expansion=neck_config['expansion'],
    depth_mult=neck_config['depth_mult'],
    mask_feat_channels=neck_config['mask_feat_channels'],
    act=neck_config['act'],
    trt=neck_config['trt'],
    eval_size=neck_config['eval_size']
)

# Transformer module configuration
transformer_config = config['transformer']
maskRTDETR = MaskRTDETR(
    num_classes=transformer_config['num_classes'],
    hidden_dim=transformer_config['hidden_dim'],
    num_queries=transformer_config['num_queries'],
    position_embed_type=transformer_config['position_embed_type'],
    backbone_feat_channels=transformer_config['backbone_feat_channels'],
    feat_strides=transformer_config['feat_strides'],
    num_prototypes=transformer_config['num_prototypes'],
    num_levels=transformer_config['num_levels'],
    num_decoder_points=transformer_config['num_decoder_points'],
    nhead=transformer_config['nhead'],
    num_decoder_layers=transformer_config['num_decoder_layers'],
    dim_feedforward=transformer_config['dim_feedforward'],
    dropout=transformer_config['dropout'],
    activation=transformer_config['activation'],
    num_denoising=transformer_config['num_denoising'],
    label_noise_ratio=transformer_config['label_noise_ratio'],
    box_noise_scale=transformer_config['box_noise_scale'],
    learnt_init_query=transformer_config['learnt_init_query'],
    query_pos_head_inv_sig=transformer_config['query_pos_head_inv_sig'],
    mask_enhanced=transformer_config['mask_enhanced'],
    eval_size=transformer_config['eval_size'],
    eval_idx=transformer_config['eval_idx'],
    eps=transformer_config['eps']
)

# MRTDETR configuration
mRTDETR_config = config['MRTDETR']
model = MRTDETR(
    backbone=pPHGNetV2,
    transformer=maskRTDETR,
    detr_head=maskDINOHead,
    neck=maskHybridEncoder,
    post_process=mRTDETR_config['post_process'],
    post_process_semi=mRTDETR_config['post_process_semi'],
    with_mask=mRTDETR_config['with_mask'],
    exclude_post_process=mRTDETR_config['exclude_post_process']
)

# Training configuration
training_config = config['training']
training = training_config['training']
pytorch_model = model
pytorch_model.training = training
pytorch_model.transformer.training = training
pytorch_model.exclude_post_process = False
pytorch_model.detr_head.training = training

image_size = training_config['image_size']
label_id = training_config['label_id']
num_instance = training_config['num_instance']
batch_size = training_config['batch_size']
num_channels = training_config['num_channels']

inputs = {
    'image': torch.rand((batch_size, num_channels, image_size, image_size)),
    'gt_bbox': [torch.rand((num_instance, 4)) for _ in range(batch_size)],
    'gt_class': [torch.tensor(np.array(num_instance * [label_id])) for _ in range(batch_size)],
    'gt_segm': [torch.randint(0, 1, (num_instance, image_size, image_size), dtype=torch.float) for _ in range(batch_size)],
}

out = pytorch_model(inputs)
