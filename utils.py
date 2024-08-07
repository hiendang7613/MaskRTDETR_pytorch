import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import typing


def gather_nd(x, index):
    index_shape = index.shape
    last_dim = index_shape[-1]

    flat_index = index.reshape(-1, last_dim)

    gathered = x[flat_index.T.tolist()]

    output_shape = list(index_shape[:-1]) + list(x.shape[last_dim:])
    output = gathered.view(output_shape)
    
    return output
    
def sigmoid_focal_loss(logit, label, normalizer=1.0, alpha=0.25, gamma=2.0):
    prob = torch.sigmoid(logit)
    label = label.float()
    ce_loss = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    p_t = prob * label + (1 - prob) * (1 - label)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        loss = alpha_t * loss
    return loss.mean(1).sum() / normalizer

# Kaiming Normal initialization
def kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu'):
    init.kaiming_normal_(tensor, mode='fan_out', nonlinearity='relu')

# Constant initialization to 0
def zeros_(tensor):
    init.constant_(tensor, 0.)

# Constant initialization to 1
def ones_(tensor):
    init.constant_(tensor, 1.)

def _freeze_norm(m: nn.BatchNorm2d):
    # Create a new BatchNorm2d with frozen parameters
    frozen_norm = nn.BatchNorm2d(
        m.num_features,
        eps=m.eps,
        momentum=m.momentum,
        affine=m.affine,
        track_running_stats=m.track_running_stats)

    # Freeze the parameters
    for param in frozen_norm.parameters():
        param.requires_grad = False

    return frozen_norm

def reset_bn(model: nn.Module, reset_func=_freeze_norm):
    if isinstance(model, nn.BatchNorm2d):
        return reset_func(model)
    else:
        for name, child in model.named_children():
            new_child = reset_bn(child, reset_func)
            if new_child is not child:
                setattr(model, name, new_child)
        return model


class BaseArch(nn.Module):
    def __init__(self, data_format='NCHW', use_extra_data=False):
        super(BaseArch, self).__init__()
        self.data_format = data_format
        self.inputs = {}
        self.fuse_norm = False
        self.use_extra_data = use_extra_data

    def load_meanstd(self, cfg_transform):
        scale = 1.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for item in cfg_transform:
            if 'NormalizeImage' in item:
                mean = np.array(item['NormalizeImage']['mean'], dtype=np.float32)
                std = np.array(item['NormalizeImage']['std'], dtype=np.float32)
                if item['NormalizeImage'].get('is_scale', True):
                    scale = 1. / 255.
                break
        if self.data_format == 'NHWC':
            self.scale = torch.tensor(scale / std).reshape((1, 1, 1, 3))
            self.bias = torch.tensor(-mean / std).reshape((1, 1, 1, 3))
        else:
            self.scale = torch.tensor(scale / std).reshape((1, 3, 1, 1))
            self.bias = torch.tensor(-mean / std).reshape((1, 3, 1, 1))

    def forward(self, inputs):
        if self.data_format == 'NHWC':
            image = inputs['image']
            inputs['image'] = image.permute(0, 2, 3, 1)

        if self.fuse_norm:
            image = inputs['image']
            self.inputs['image'] = image * self.scale + self.bias
            self.inputs['im_shape'] = inputs['im_shape']
            self.inputs['scale_factor'] = inputs['scale_factor']
        else:
            self.inputs = inputs

        self.model_arch()

        if self.training:
            out = self.get_loss()
        else:
            inputs_list = []
            # multi-scale input
            if not isinstance(inputs, typing.Sequence):
                inputs_list.append(inputs)
            else:
                inputs_list.extend(inputs)
            outs = []
            for inp in inputs_list:
                if self.fuse_norm:
                    self.inputs['image'] = inp['image'] * self.scale + self.bias
                    self.inputs['im_shape'] = inp['im_shape']
                    self.inputs['scale_factor'] = inp['scale_factor']
                else:
                    self.inputs = inp
                outs.append(self.get_pred())

            # multi-scale test
            if len(outs) > 1:
                out = self.merge_multi_scale_predictions(outs)
            else:
                out = outs[0]
        return out

    def merge_multi_scale_predictions(self, outs):
        # default values for architectures not included in following list
        num_classes = 80
        nms_threshold = 0.5
        keep_top_k = 100

        if self.__class__.__name__ in ('CascadeRCNN', 'FasterRCNN', 'MaskRCNN'):
            num_classes = self.bbox_head.num_classes
            keep_top_k = self.bbox_post_process.nms.keep_top_k
            nms_threshold = self.bbox_post_process.nms.nms_threshold
        else:
            raise Exception("Multi scale test only supports CascadeRCNN, FasterRCNN and MaskRCNN for now")

        final_boxes = []
        all_scale_outs = torch.cat([o['bbox'] for o in outs]).cpu().numpy()
        for c in range(num_classes):
            idxs = all_scale_outs[:, 0] == c
            if np.count_nonzero(idxs) == 0:
                continue
            r = nms(all_scale_outs[idxs, 1:], nms_threshold)
            final_boxes.append(np.concatenate([np.full((r.shape[0], 1), c), r], 1))
        out = np.concatenate(final_boxes)
        out = np.concatenate(sorted(out, key=lambda e: e[1])[-keep_top_k:]).reshape((-1, 6))
        out = {
            'bbox': torch.tensor(out),
            'bbox_num': torch.tensor(np.array([out.shape[0]]))
        }

        return out

    def build_inputs(self, data, input_def):
        inputs = {}
        for i, k in enumerate(input_def):
            inputs[k] = data[i]
        return inputs

    def model_arch(self):
        pass

    def get_loss(self):
        raise NotImplementedError("Should implement get_loss method!")

    def get_pred(self):
        raise NotImplementedError("Should implement get_pred method!")

