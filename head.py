# MaskDINOHead

class MaskDINOHead(nn.Module):
    def __init__(self, loss='DINOLoss'):
        super(MaskDINOHead, self).__init__()
        self.loss = loss

    def forward(self, out_transformer, body_feats, inputs=None):
        dec_out_logits, dec_out_bboxes, dec_out_masks, enc_out, init_out, dn_meta = out_transformer
        
        if self.training:
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs
            assert 'gt_segm' in inputs

            if dn_meta is not None:
                dn_out_logits, dec_out_logits = torch.split(
                    dec_out_logits, dn_meta['dn_num_split'], dim=2)
                dn_out_bboxes, dec_out_bboxes = torch.split(
                    dec_out_bboxes, dn_meta['dn_num_split'], dim=2)
                dn_out_masks, dec_out_masks = torch.split(
                    dec_out_masks, dn_meta['dn_num_split'], dim=2)
                if init_out is not None:
                    init_out_logits, init_out_bboxes, init_out_masks = init_out
                    init_out_logits_dn, init_out_logits = torch.split(
                        init_out_logits, dn_meta['dn_num_split'], dim=1)
                    init_out_bboxes_dn, init_out_bboxes = torch.split(
                        init_out_bboxes, dn_meta['dn_num_split'], dim=1)
                    init_out_masks_dn, init_out_masks = torch.split(
                        init_out_masks, dn_meta['dn_num_split'], dim=1)

                    dec_out_logits = torch.cat(
                        [init_out_logits.unsqueeze(0), dec_out_logits])
                    dec_out_bboxes = torch.cat(
                        [init_out_bboxes.unsqueeze(0), dec_out_bboxes])
                    dec_out_masks = torch.cat(
                        [init_out_masks.unsqueeze(0), dec_out_masks])

                    dn_out_logits = torch.cat(
                        [init_out_logits_dn.unsqueeze(0), dn_out_logits])
                    dn_out_bboxes = torch.cat(
                        [init_out_bboxes_dn.unsqueeze(0), dn_out_bboxes])
                    dn_out_masks = torch.cat(
                        [init_out_masks_dn.unsqueeze(0), dn_out_masks])
            else:
                dn_out_bboxes, dn_out_logits = None, None
                dn_out_masks = None

            enc_out_logits, enc_out_bboxes, enc_out_masks = enc_out
            out_logits = torch.cat(
                [enc_out_logits.unsqueeze(0), dec_out_logits])
            out_bboxes = torch.cat(
                [enc_out_bboxes.unsqueeze(0), dec_out_bboxes])
            out_masks = torch.cat(
                [enc_out_masks.unsqueeze(0), dec_out_masks])

            return self.loss(
                out_bboxes,
                out_logits,
                inputs['gt_bbox'],
                inputs['gt_class'],
                masks=out_masks,
                gt_mask=inputs['gt_segm'],
                dn_out_logits=dn_out_logits,
                dn_out_bboxes=dn_out_bboxes,
                dn_out_masks=dn_out_masks,
                dn_meta=dn_meta)
        else:
            return (dec_out_bboxes[-1], dec_out_logits[-1], dec_out_masks[-1])