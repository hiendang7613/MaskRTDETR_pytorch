

class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1., eps=1e-10, reduction='none'):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xkis1 = torch.maximum(x1, x1g)
        ykis1 = torch.maximum(y1, y1g)
        xkis2 = torch.minimum(x2, x2g)
        ykis2 = torch.minimum(y2, y2g)
        w_inter = (xkis2 - xkis1).clamp(min=0)
        h_inter = (ykis2 - ykis1).clamp(min=0)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union

        return iou, overlap, union

    def __call__(self, pbox, gbox, iou_weight=1., loc_reweight=None):
        x1, y1, x2, y2 = torch.chunk(pbox, 4, dim=-1)
        x1g, y1g, x2g, y2g = torch.chunk(gbox, 4, dim=-1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = torch.minimum(x1, x1g)
        yc1 = torch.minimum(y1, y1g)
        xc2 = torch.maximum(x2, x2g)
        yc2 = torch.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = loc_reweight.view(-1, 1)
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == 'none':
            loss = giou
        elif self.reduction == 'sum':
            loss = torch.sum(giou * iou_weight)
        else:
            loss = torch.mean(giou * iou_weight)
        return loss * self.loss_weight

    def _get_loss_class(self,
                        logits,
                        gt_class,
                        match_indices,
                        bg_index,
                        num_gts,
                        postfix="",
                        iou_score=None,
                        gt_score=None):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = "loss_class" + postfix

        target_label = torch.full(logits.shape[:2], bg_index, dtype=torch.int64)
        bs, num_query_objects = target_label.shape
        num_gt = sum(len(a) for a in gt_class)
        if num_gt > 0:
            index, updates = self._get_index_updates(num_query_objects, gt_class, match_indices)
            target_label = target_label.view(-1, 1).scatter_(0, index, updates.to(torch.int64)).view(bs, num_query_objects)
        if self.use_focal_loss:
            target_label = F.one_hot(target_label, self.num_classes + 1)[..., :-1].to(logits.device)
            if iou_score is not None and self.use_vfl:
                if gt_score is not None:
                    target_score = torch.zeros([bs, num_query_objects], device=logits.device)
                    target_score = target_score.view(-1, 1).scatter_(0, index, gt_score).view(bs, num_query_objects, 1) * target_label

                    target_score_iou = torch.zeros([bs, num_query_objects], device=logits.device)
                    target_score_iou = target_score_iou.view(-1, 1).scatter_(0, index, iou_score).view(bs, num_query_objects, 1) * target_label
                    target_score = target_score * target_score_iou
                    loss_ = self.loss_coeff['class'] * varifocal_loss_with_logits(logits, target_score, target_label, num_gts / num_query_objects)
                else:
                    target_score = torch.zeros([bs, num_query_objects], device=logits.device)
                    if num_gt > 0:
                        target_score = target_score.view(-1, 1).scatter_(0, index, iou_score).view(bs, num_query_objects, 1) * target_label
                    loss_ = self.loss_coeff['class'] * varifocal_loss_with_logits(logits, target_score, target_label, num_gts / num_query_objects)
            else:
                loss_ = self.loss_coeff['class'] * sigmoid_focal_loss(logits, target_label, num_gts / num_query_objects)
        else:
            loss_ = F.cross_entropy(logits, target_label, weight=self.loss_coeff['class'])
        return {name_class: loss_}

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts, postfix=""):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix

        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = torch.tensor([0.], device=boxes.device)
            loss[name_giou] = torch.tensor([0.], device=boxes.device)
            return loss

        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox, match_indices)
        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(src_bbox, target_bbox, reduction='sum') / num_gts
        loss[name_giou] = self.giou_loss(bbox_cxcywh_to_xyxy(src_bbox), bbox_cxcywh_to_xyxy(target_bbox))
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]
        return loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts, postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = torch.tensor([0.], device=masks.device)
            loss[name_dice] = torch.tensor([0.], device=masks.device)
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask, match_indices)
        src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode="bilinear")[0]
        loss[name_mask] = self.loss_coeff['mask'] * sigmoid_focal_loss(src_masks, target_masks, torch.tensor([num_gts], dtype=torch.float32, device=masks.device))
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
        return loss

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self,
                      boxes,
                      logits,
                      gt_bbox,
                      gt_class,
                      bg_index,
                      num_gts,
                      dn_match_indices=None,
                      postfix="",
                      masks=None,
                      gt_mask=None,
                      gt_score=None):
        loss_class = []
        loss_bbox, loss_giou = [], []
        loss_mask, loss_dice = [], []
        
        if dn_match_indices is not None:
            match_indices = dn_match_indices
        elif self.use_uni_match:
            match_indices = self.matcher(
                boxes[self.uni_match_ind],
                logits[self.uni_match_ind],
                gt_bbox,
                gt_class,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask
            )
        
        for i, (aux_boxes, aux_logits) in enumerate(zip(boxes, logits)):
            aux_masks = masks[i] if masks is not None else None
            if not self.use_uni_match and dn_match_indices is None:
                match_indices = self.matcher(
                    aux_boxes,
                    aux_logits,
                    gt_bbox,
                    gt_class,
                    masks=aux_masks,
                    gt_mask=gt_mask
                )
            if self.use_vfl:
                if sum(len(a) for a in gt_bbox) > 0:
                    src_bbox, target_bbox = self._get_src_target_assign(aux_boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(bbox_cxcywh_to_xyxy(src_bbox).split(4, -1), bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
                else:
                    iou_score = None
                if gt_score is not None:
                    _, target_score = self._get_src_target_assign(logits[-1].detach(), gt_score, match_indices)
            else:
                iou_score = None
            
            loss_class.append(
                self._get_loss_class(
                    aux_logits,
                    gt_class,
                    match_indices,
                    bg_index,
                    num_gts,
                    postfix,
                    iou_score,
                    gt_score=target_score if gt_score is not None else None
                )['loss_class' + postfix]
            )
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices, num_gts, postfix)
            loss_bbox.append(loss_['loss_bbox' + postfix])
            loss_giou.append(loss_['loss_giou' + postfix])
            
            if masks is not None and gt_mask is not None:
                loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, num_gts, postfix)
                loss_mask.append(loss_['loss_mask' + postfix])
                loss_dice.append(loss_['loss_dice' + postfix])
        
        loss = {
            "loss_class_aux" + postfix: torch.stack(loss_class).sum(),
            "loss_bbox_aux" + postfix: torch.stack(loss_bbox).sum(),
            "loss_giou_aux" + postfix: torch.stack(loss_giou).sum()
        }
        if masks is not None and gt_mask is not None:
            loss["loss_mask_aux" + postfix] = torch.stack(loss_mask).sum()
            loss["loss_dice_aux" + postfix] = torch.stack(loss_dice).sum()
        
        return loss

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = torch.cat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)
        target_assign = torch.cat([
            t.gather(0, dst) for t, (_, dst) in zip(target, match_indices)
        ])
        return src_idx, target_assign

    def _get_src_target_assign(self, src, target, match_indices):
        src_assign = torch.cat([
            t.gather(0, I) if len(I) > 0 else torch.zeros([0, t.shape[-1]], device=t.device)
            for t, (I, _) in zip(src, match_indices)
        ])
        target_assign = torch.cat([
            t.gather(0, J) if len(J) > 0 else torch.zeros([0, t.shape[-1]], device=t.device)
            for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, target_assign

    def _get_num_gts(self, targets, dtype=torch.float32):
        num_gts = sum(len(a) for a in targets)
        num_gts = torch.tensor([num_gts], dtype=dtype, device=targets[0].device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_gts)
            num_gts /= torch.distributed.get_world_size()
        num_gts = torch.clamp(num_gts, min=1.)
        return num_gts

    def _get_prediction_loss(self,
                            boxes,
                            logits,
                            gt_bbox,
                            gt_class,
                            masks=None,
                            gt_mask=None,
                            postfix="",
                            dn_match_indices=None,
                            num_gts=1,
                            gt_score=None):
        if dn_match_indices is None:
            match_indices = self.matcher(
                boxes, logits, gt_bbox, gt_class, masks=masks, gt_mask=gt_mask)
        else:
            match_indices = dn_match_indices

        if self.use_vfl:
            if gt_score is not None:  # ssod
                _, target_score = self._get_src_target_assign(
                    logits[-1].detach(), gt_score, match_indices)
            elif sum(len(a) for a in gt_bbox) > 0:
                if self.vfl_iou_type == 'bbox':
                    src_bbox, target_bbox = self._get_src_target_assign(
                        boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(
                        bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                        bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
                elif self.vfl_iou_type == 'mask':
                    assert masks is not None and gt_mask is not None, 'Make sure the input has `mask` and `gt_mask`'
                    assert sum(len(a) for a in gt_mask) > 0
                    src_mask, target_mask = self._get_src_target_assign(
                        masks.detach(), gt_mask, match_indices)
                    src_mask = F.interpolate(
                        src_mask.unsqueeze(0),
                        scale_factor=2,
                        mode='bilinear',
                        align_corners=False).squeeze(0)
                    target_mask = F.interpolate(
                        target_mask.unsqueeze(0),
                        size=src_mask.shape[-2:],
                        mode='bilinear',
                        align_corners=False).squeeze(0)
                    src_mask = src_mask.flatten(1)
                    src_mask = torch.sigmoid(src_mask)
                    src_mask = torch.where(src_mask > 0.5, 1., 0.).to(masks.dtype)
                    target_mask = target_mask.flatten(1)
                    target_mask = torch.where(target_mask > 0.5, 1., 0.).to(masks.dtype)
                    inter = (src_mask * target_mask).sum(1)
                    union = src_mask.sum(1) + target_mask.sum(1) - inter
                    iou_score = (inter + 1e-2) / (union + 1e-2)
                    iou_score = iou_score.unsqueeze(-1)
                else:
                    iou_score = None
            else:
                iou_score = None
        else:
            iou_score = None

        loss = dict()
        loss.update(
            self._get_loss_class(
                logits,
                gt_class,
                match_indices,
                self.num_classes,
                num_gts,
                postfix,
                iou_score,
                gt_score=target_score if gt_score is not None else None))
        loss.update(
            self._get_loss_bbox(boxes, gt_bbox, match_indices, num_gts, postfix))
        if masks is not None and gt_mask is not None:
            loss.update(
                self._get_loss_mask(masks, gt_mask, match_indices, num_gts, postfix))
        return loss

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                gt_score=None,
                **kwargs):
        r"""
        Args:
            boxes (Tensor): [l, b, query, 4]
            logits (Tensor): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [l, b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
            postfix (str): postfix of loss name
        """

        dn_match_indices = kwargs.get("dn_match_indices", None)
        num_gts = kwargs.get("num_gts", None)
        if num_gts is None:
            num_gts = self._get_num_gts(gt_class)

        total_loss = self._get_prediction_loss(
            boxes[-1],
            logits[-1],
            gt_bbox,
            gt_class,
            masks=masks[-1] if masks is not None else None,
            gt_mask=gt_mask,
            postfix=postfix,
            dn_match_indices=dn_match_indices,
            num_gts=num_gts,
            gt_score=gt_score if gt_score is not None else None
        )

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    boxes[:-1],
                    logits[:-1],
                    gt_bbox,
                    gt_class,
                    self.num_classes,
                    num_gts,
                    dn_match_indices,
                    postfix,
                    masks=masks[:-1] if masks is not None else None,
                    gt_mask=gt_mask,
                    gt_score=gt_score if gt_score is not None else None
                )
            )

        return total_loss


# class GIoULoss(object):
#     """
#     Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
#     Args:
#         loss_weight (float): giou loss weight, default as 1
#         eps (float): epsilon to avoid divide by zero, default as 1e-10
#         reduction (string): Options are "none", "mean" and "sum". default as none
#     """

#     def __init__(self, loss_weight=1., eps=1e-10, reduction='none'):
#         self.loss_weight = loss_weight
#         self.eps = eps
#         assert reduction in ('none', 'mean', 'sum')
#         self.reduction = reduction

#     def bbox_overlap(self, box1, box2, eps=1e-10):
#         """calculate the iou of box1 and box2
#         Args:
#             box1 (Tensor): box1 with the shape (..., 4)
#             box2 (Tensor): box1 with the shape (..., 4)
#             eps (float): epsilon to avoid divide by zero
#         Return:
#             iou (Tensor): iou of box1 and box2
#             overlap (Tensor): overlap of box1 and box2
#             union (Tensor): union of box1 and box2
#         """
#         x1, y1, x2, y2 = box1
#         x1g, y1g, x2g, y2g = box2

#         xkis1 = torch.maximum(x1, x1g)
#         ykis1 = torch.maximum(y1, y1g)
#         xkis2 = torch.minimum(x2, x2g)
#         ykis2 = torch.minimum(y2, y2g)
#         w_inter = (xkis2 - xkis1).clamp(min=0)
#         h_inter = (ykis2 - ykis1).clamp(min=0)
#         overlap = w_inter * h_inter

#         area1 = (x2 - x1) * (y2 - y1)
#         area2 = (x2g - x1g) * (y2g - y1g)
#         union = area1 + area2 - overlap + eps
#         iou = overlap / union

#         return iou, overlap, union

#     def __call__(self, pbox, gbox, iou_weight=1., loc_reweight=None):
#         x1, y1, x2, y2 = torch.chunk(pbox, 4, dim=-1)
#         x1g, y1g, x2g, y2g = torch.chunk(gbox, 4, dim=-1)
#         box1 = [x1, y1, x2, y2]
#         box2 = [x1g, y1g, x2g, y2g]
#         iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
#         xc1 = torch.minimum(x1, x1g)
#         yc1 = torch.minimum(y1, y1g)
#         xc2 = torch.maximum(x2, x2g)
#         yc2 = torch.maximum(y2, y2g)

#         area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
#         miou = iou - ((area_c - union) / area_c)
#         if loc_reweight is not None:
#             loc_reweight = loc_reweight.view(-1, 1)
#             loc_thresh = 0.9
#             giou = 1 - (1 - loc_thresh) * miou - loc_thresh * miou * loc_reweight
#         else:
#             giou = 1 - miou
#         if self.reduction == 'none':
#             loss = giou
#         elif self.reduction == 'sum':
#             loss = torch.sum(giou * iou_weight)
#         else:
#             loss = torch.mean(giou * iou_weight)
#         return loss * self.loss_weight


def bbox_cxcywh_to_xyxy(x):
    cxcy, wh = torch.split(x, 2, dim=-1)
    return torch.cat([cxcy - 0.5 * wh, cxcy + 0.5 * wh], dim=-1)


class HungarianMatcher(nn.Module):
    __shared__ = ['use_focal_loss', 'with_mask', 'num_sample_points']

    def __init__(self,
                 matcher_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'mask': 1,
                     'dice': 1
                 },
                 use_focal_loss=False,
                 with_mask=False,
                 num_sample_points=12544,
                 alpha=0.25,
                 gamma=2.0):
        r"""
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super(HungarianMatcher, self).__init__()
        self.matcher_coeff = matcher_coeff
        self.use_focal_loss = use_focal_loss
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

        self.giou_loss = GIoULoss()

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None):
        r"""
        Args:
            boxes (Tensor): [b, query, 4]
            logits (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor|None): [b, query, h, w]
            gt_mask (List(Tensor)): list[[n, H, W]]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = boxes.shape[:2]

        num_gts = [len(a) for a in gt_class]
        if sum(num_gts) == 0:
            return [(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        logits = logits.detach()
        out_prob = torch.sigmoid(logits.flatten(0, 1)) if self.use_focal_loss else F.softmax(logits.flatten(0, 1), dim=-1)
        # [batch_size * num_queries, 4]
        out_bbox = boxes.detach().flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat(gt_class).flatten()
        tgt_bbox = torch.cat(gt_bbox)
      
        # Compute the classification cost
        out_prob = torch.gather(out_prob, 1, tgt_ids.unsqueeze(0).repeat(out_prob.shape[0],1))
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob

        # Compute the L1 cost between boxes
        cost_bbox = (out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        # Compute the giou cost between boxes
        giou_loss = self.giou_loss(bbox_cxcywh_to_xyxy(out_bbox.unsqueeze(1)), bbox_cxcywh_to_xyxy(tgt_bbox.unsqueeze(0))).squeeze(-1)
        cost_giou = giou_loss - 1

        # Final cost matrix
        C = self.matcher_coeff['class'] * cost_class + \
            self.matcher_coeff['bbox'] * cost_bbox + \
            self.matcher_coeff['giou'] * cost_giou

        # Compute the mask cost and dice cost
        if self.with_mask:
            assert masks is not None and gt_mask is not None, 'Make sure the input has `mask` and `gt_mask`'
            # all masks share the same set of points for efficient matching
            sample_points = torch.rand([bs, 1, self.num_sample_points, 2], device=masks.device)
            sample_points = 2.0 * sample_points - 1.0

            out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
            out_mask = out_mask.flatten(0, 1)

            tgt_mask = torch.cat(gt_mask).unsqueeze(1)
            sample_points = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
            tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze([1, 2])

            with torch.cuda.amp.autocast(enabled=False):
                # binary cross entropy cost
                pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
                neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
                cost_mask = torch.matmul(pos_cost_mask, tgt_mask.t()) + torch.matmul(neg_cost_mask, (1 - tgt_mask).t())
                cost_mask /= self.num_sample_points

                # dice cost
                out_mask = torch.sigmoid(out_mask)
                numerator = 2 * torch.matmul(out_mask, tgt_mask.t())
                denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
                cost_dice = 1 - (numerator + 1) / (denominator + 1)

                C = C + self.matcher_coeff['mask'] * cost_mask + self.matcher_coeff['dice'] * cost_dice

        C = C.view(bs, num_queries, -1)
        C = [a.squeeze(0) for a in C.chunk(bs)]
        sizes = [a.shape[0] for a in gt_bbox]
        indices = [linear_sum_assignment(c[:, :size].cpu().numpy()) for c, size in zip(C, sizes)]
        return [(torch.tensor(i, dtype=torch.int64), torch.tensor(j, dtype=torch.int64)) for i, j in indices]


class DETRLoss(nn.Module):
    __shared__ = ['num_classes', 'use_focal_loss']
    __inject__ = ['matcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 loss_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'no_object': 0.1,
                     'mask': 1,
                     'dice': 1
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 use_vfl=False,
                 use_uni_match=False,
                 uni_match_ind=0):
        r"""
        Args:
            num_classes (int): The number of classes.
            matcher (HungarianMatcher): It computes an assignment between the targets
                and the predictions of the network.
            loss_coeff (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_focal_loss (bool): Use focal loss or not.
        """
        super(DETRLoss, self).__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_coeff = loss_coeff
        self.aux_loss = aux_loss
        self.use_focal_loss = use_focal_loss
        self.use_vfl = use_vfl
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind

        if not self.use_focal_loss:
            self.loss_coeff['class'] = torch.full([num_classes + 1], loss_coeff['class'])
            self.loss_coeff['class'][-1] = loss_coeff['no_object']
        self.giou_loss = GIoULoss()


    def _get_loss_class(self,
                        logits,
                        gt_class,
                        match_indices,
                        bg_index,
                        num_gts,
                        postfix="",
                        iou_score=None,
                        gt_score=None):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = "loss_class" + postfix

        target_label = torch.full(logits.shape[:2], bg_index, dtype=torch.int64)
        bs, num_query_objects = target_label.shape
        num_gt = sum(len(a) for a in gt_class)
        if num_gt > 0:
            index, updates = self._get_index_updates(num_query_objects, gt_class, match_indices)
            target_label = target_label.view(-1, 1).scatter_(0, index.unsqueeze(-1), updates.to(torch.int64)).view(bs, num_query_objects)
        if self.use_focal_loss:
            target_label = F.one_hot(target_label, self.num_classes + 1)[..., :-1].to(logits.device)
            if iou_score is not None and self.use_vfl:
                if gt_score is not None:
                    target_score = torch.zeros([bs, num_query_objects], device=logits.device)
                    target_score = target_score.view(-1, 1).scatter_(0, index, gt_score).view(bs, num_query_objects, 1) * target_label

                    target_score_iou = torch.zeros([bs, num_query_objects], device=logits.device)
                    target_score_iou = target_score_iou.view(-1, 1).scatter_(0, index, iou_score).view(bs, num_query_objects, 1) * target_label
                    target_score = target_score * target_score_iou
                    loss_ = self.loss_coeff['class'] * varifocal_loss_with_logits(logits, target_score, target_label, num_gts / num_query_objects)
                else:
                    target_score = torch.zeros([bs, num_query_objects], device=logits.device)
                    if num_gt > 0:
                        target_score = target_score.view(-1, 1).scatter_(0, index, iou_score).view(bs, num_query_objects, 1) * target_label
                    loss_ = self.loss_coeff['class'] * varifocal_loss_with_logits(logits, target_score, target_label, num_gts / num_query_objects)
            else:
                loss_ = self.loss_coeff['class'] * sigmoid_focal_loss(logits, target_label, num_gts / num_query_objects)
        else:
            loss_ = F.cross_entropy(logits, target_label, weight=self.loss_coeff['class'])
        return {name_class: loss_}

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts, postfix=""):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix

        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = torch.tensor([0.], device=boxes.device)
            loss[name_giou] = torch.tensor([0.], device=boxes.device)
            return loss

        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox, match_indices)
        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(src_bbox, target_bbox, reduction='sum') / num_gts
        loss[name_giou] = self.giou_loss(bbox_cxcywh_to_xyxy(src_bbox), bbox_cxcywh_to_xyxy(target_bbox))
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]
        return loss

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts, postfix=""):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix

        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = torch.tensor([0.], device=boxes.device)
            loss[name_giou] = torch.tensor([0.], device=boxes.device)
            return loss

        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox, match_indices)
        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(src_bbox, target_bbox, reduction='sum') / num_gts
        loss[name_giou] = self.giou_loss(bbox_cxcywh_to_xyxy(src_bbox), bbox_cxcywh_to_xyxy(target_bbox))
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]
        return loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts, postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = torch.tensor([0.], device=masks.device)
            loss[name_dice] = torch.tensor([0.], device=masks.device)
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask, match_indices)
        src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode="bilinear")[0]
        loss[name_mask] = self.loss_coeff['mask'] * sigmoid_focal_loss(
            src_masks,
            target_masks,
            torch.tensor([num_gts], dtype=torch.float32, device=masks.device))
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self,
                      boxes,
                      logits,
                      gt_bbox,
                      gt_class,
                      bg_index,
                      num_gts,
                      dn_match_indices=None,
                      postfix="",
                      masks=None,
                      gt_mask=None,
                      gt_score=None):
        loss_class = []
        loss_bbox, loss_giou = [], []
        loss_mask, loss_dice = [], []
        if dn_match_indices is not None:
            match_indices = dn_match_indices
        elif self.use_uni_match:
            match_indices = self.matcher(
                boxes[self.uni_match_ind],
                logits[self.uni_match_ind],
                gt_bbox,
                gt_class,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask)
        for i, (aux_boxes, aux_logits) in enumerate(zip(boxes, logits)):
            aux_masks = masks[i] if masks is not None else None
            if not self.use_uni_match and dn_match_indices is None:
                match_indices = self.matcher(
                    aux_boxes,
                    aux_logits,
                    gt_bbox,
                    gt_class,
                    masks=aux_masks,
                    gt_mask=gt_mask)
            if self.use_vfl:
                if sum(len(a) for a in gt_bbox) > 0:
                    src_bbox, target_bbox = self._get_src_target_assign(
                        aux_boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(
                        bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                        bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
                else:
                    iou_score = None
                if gt_score is not None:
                    _, target_score = self._get_src_target_assign(
                        logits[-1].detach(), gt_score, match_indices)
            else:
                iou_score = None
            loss_class.append(
                self._get_loss_class(
                    aux_logits,
                    gt_class,
                    match_indices,
                    bg_index,
                    num_gts,
                    postfix,
                    iou_score,
                    gt_score=target_score
                    if gt_score is not None else None)['loss_class' + postfix])
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices, num_gts, postfix)
            loss_bbox.append(loss_['loss_bbox' + postfix])
            loss_giou.append(loss_['loss_giou' + postfix])
            if masks is not None and gt_mask is not None:
                loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, num_gts, postfix)
                loss_mask.append(loss_['loss_mask' + postfix])
                loss_dice.append(loss_['loss_dice' + postfix])
        loss = {
            "loss_class_aux" + postfix: torch.stack(loss_class).sum(),
            "loss_bbox_aux" + postfix: torch.stack(loss_bbox).sum(),
            "loss_giou_aux" + postfix: torch.stack(loss_giou).sum()
        }
        if masks is not None and gt_mask is not None:
            loss["loss_mask_aux" + postfix] = torch.stack(loss_mask).sum()
            loss["loss_dice_aux" + postfix] = torch.stack(loss_dice).sum()
        return loss

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = torch.cat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)

        target_assign = torch.cat([
            t.gather(0, dst) for t, (_, dst) in zip(target, match_indices)
        ])
        return src_idx, target_assign

    def _get_src_target_assign(self, src, target, match_indices):
        src_assign = torch.cat([
            t.gather(0, I) if len(I) > 0 else torch.zeros([0, t.shape[-1]], device=t.device)
            for t, (I, _) in zip(src, match_indices)
        ])
        target_assign = torch.cat([
            t.gather(0, J) if len(J) > 0 else torch.zeros([0, t.shape[-1]], device=t.device)
            for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, target_assign

    def _get_num_gts(self, targets, dtype=torch.float32):
        num_gts = sum(len(a) for a in targets)
        num_gts = torch.tensor([num_gts], dtype=dtype, device=targets[0].device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_gts)
            num_gts /= torch.distributed.get_world_size()
        num_gts = torch.clamp(num_gts, min=1.)
        return num_gts
    
    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = torch.cat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)
        target_assign = torch.cat([
            t.gather(0, dst.unsqueeze(1)) for t, (_, dst) in zip(target, match_indices)
        ])
        return src_idx, target_assign

    def _get_src_target_assign(self, src, target, match_indices):
        num_dim_expand = len(src[0].shape) - 1
        new_expand_shape = (-1,) + (1,) * num_dim_expand
        match_indices = [(i.view(new_expand_shape), j.view(new_expand_shape)) for i,j in match_indices]
        new_src_shape = (match_indices[0][0].shape[0],) + tuple(src[0].shape[1:])
        new_target_shape = (match_indices[0][1].shape[0],) + tuple(target[0].shape[1:])
        match_indices = [(i.broadcast_to(new_src_shape), j.broadcast_to(new_target_shape)) for i,j in match_indices]
        src_assign = torch.cat([
            t.gather(0, I) if len(I) > 0 else torch.zeros([0, t.shape[-1]], device=t.device)
            for t, (I, _) in zip(src, match_indices)
        ])
        target_assign = torch.cat([
            t.gather(0, J) if len(J) > 0 else torch.zeros([0, t.shape[-1]], device=t.device)
            for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, target_assign

    def _get_num_gts(self, targets, dtype=torch.float32):
        num_gts = sum(len(a) for a in targets)
        num_gts = torch.tensor([num_gts], dtype=dtype, device=targets[0].device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_gts)
            num_gts /= torch.distributed.get_world_size()
        num_gts = torch.clamp(num_gts, min=1.)
        return num_gts

    def _get_prediction_loss(self,
                            boxes,
                            logits,
                            gt_bbox,
                            gt_class,
                            masks=None,
                            gt_mask=None,
                            postfix="",
                            dn_match_indices=None,
                            num_gts=1,
                            gt_score=None):
        if dn_match_indices is None:
            match_indices = self.matcher(
                boxes, logits, gt_bbox, gt_class, masks=masks, gt_mask=gt_mask)
        else:
            match_indices = dn_match_indices

        if self.use_vfl:
            if gt_score is not None:  # ssod
                _, target_score = self._get_src_target_assign(
                    logits[-1].detach(), gt_score, match_indices)
            elif sum(len(a) for a in gt_bbox) > 0:
                src_bbox, target_bbox = self._get_src_target_assign(
                    boxes.detach(), gt_bbox, match_indices)
                iou_score = bbox_iou(
                    bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                    bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
            else:
                iou_score = None
        else:
            iou_score = None

        loss = dict()
        loss.update(
            self._get_loss_class(
                logits,
                gt_class,
                match_indices,
                self.num_classes,
                num_gts,
                postfix,
                iou_score,
                gt_score=target_score if gt_score is not None else None))
        loss.update(
            self._get_loss_bbox(boxes, gt_bbox, match_indices, num_gts, postfix))
        if masks is not None and gt_mask is not None:
            loss.update(
                self._get_loss_mask(masks, gt_mask, match_indices, num_gts, postfix))
        return loss

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                gt_score=None,
                **kwargs):
        r"""
        Args:
            boxes (Tensor): [l, b, query, 4]
            logits (Tensor): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [l, b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
            postfix (str): postfix of loss name
        """

        dn_match_indices = kwargs.get("dn_match_indices", None)
        num_gts = kwargs.get("num_gts", None)
        if num_gts is None:
            num_gts = self._get_num_gts(gt_class)

        total_loss = self._get_prediction_loss(
            boxes[-1],
            logits[-1],
            gt_bbox,
            gt_class,
            masks=masks[-1] if masks is not None else None,
            gt_mask=gt_mask,
            postfix=postfix,
            dn_match_indices=dn_match_indices,
            num_gts=num_gts,
            gt_score=gt_score if gt_score is not None else None)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    boxes[:-1],
                    logits[:-1],
                    gt_bbox,
                    gt_class,
                    self.num_classes,
                    num_gts,
                    dn_match_indices,
                    postfix,
                    masks=masks[:-1] if masks is not None else None,
                    gt_mask=gt_mask,
                    gt_score=gt_score if gt_score is not None else None))

        return total_loss


class DINOLoss(DETRLoss):
    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_meta=None,
                gt_score=None,
                **kwargs):
        num_gts = self._get_num_gts(gt_class)
        total_loss = super(DINOLoss, self).forward(
            boxes,
            logits,
            gt_bbox,
            gt_class,
            num_gts=num_gts,
            gt_score=gt_score)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = self.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(DINOLoss, self).forward(
                dn_out_bboxes,
                dn_out_logits,
                gt_bbox,
                gt_class,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts,
                gt_score=gt_score)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': torch.tensor([0.], device=boxes.device)
                 for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(labels, dn_positive_idx, dn_num_group):
        dn_match_indices = []
        for i in range(len(labels)):
            num_gt = len(labels[i])
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.int64, device=labels[i].device)
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(
                    [0], dtype=torch.int64, device=labels[i].device), torch.zeros(
                        [0], dtype=torch.int64, device=labels[i].device)))
        return dn_match_indices


class MaskDINOLoss(DETRLoss):
    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 loss_coeff={
                     'class': 4,
                     'bbox': 5,
                     'giou': 2,
                     'mask': 5,
                     'dice': 5
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 num_sample_points=12544,
                 oversample_ratio=3.0,
                 important_sample_ratio=0.75):
        super(MaskDINOLoss, self).__init__(num_classes, matcher, loss_coeff, aux_loss, use_focal_loss)
        assert oversample_ratio >= 1
        assert important_sample_ratio <= 1 and important_sample_ratio >= 0

        self.num_sample_points = num_sample_points
        self.oversample_ratio = oversample_ratio
        self.important_sample_ratio = important_sample_ratio
        self.num_oversample_points = int(num_sample_points * oversample_ratio)
        self.num_important_points = int(num_sample_points * important_sample_ratio)
        self.num_random_points = num_sample_points - self.num_important_points

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_out_masks=None,
                dn_meta=None,
                **kwargs):
        num_gts = self._get_num_gts(gt_class)
        total_loss = super(MaskDINOLoss, self).forward(
            boxes,
            logits,
            gt_bbox,
            gt_class,
            masks=masks,
            gt_mask=gt_mask,
            num_gts=num_gts)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = DINOLoss.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(MaskDINOLoss, self).forward(
                dn_out_bboxes,
                dn_out_logits,
                gt_bbox,
                gt_class,
                masks=dn_out_masks,
                gt_mask=gt_mask,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': torch.tensor([0.])
                 for k in total_loss.keys()})

        return total_loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts, postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = torch.tensor([0.])
            loss[name_dice] = torch.tensor([0.])
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask, match_indices)
        # sample points
        sample_points = self._get_point_coords_by_uncertainty(src_masks)
        sample_points = 2.0 * sample_points.unsqueeze(1) - 1.0

        src_masks = F.grid_sample(src_masks.unsqueeze(1), sample_points, align_corners=False).squeeze(1).squeeze(1)

        target_masks = F.grid_sample(target_masks.unsqueeze(1), sample_points, align_corners=False).squeeze(1).squeeze(1).detach()

        loss[name_mask] = self.loss_coeff['mask'] * F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='none').mean(1).sum() / num_gts
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
        return loss

    def _get_point_coords_by_uncertainty(self, masks):
        # Sample points based on their uncertainty.
        masks = masks.detach()
        num_masks = masks.shape[0]
        sample_points = torch.rand([num_masks, 1, self.num_oversample_points, 2])

        out_mask = F.grid_sample(masks.unsqueeze(1), 2.0 * sample_points - 1.0, align_corners=False).squeeze(1).squeeze(1)
        out_mask = -torch.abs(out_mask)

        _, topk_ind = torch.topk(out_mask, self.num_important_points, dim=1)
        batch_ind = torch.arange(end=num_masks, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).repeat(1, self.num_important_points)
        topk_ind = torch.stack([batch_ind, topk_ind], dim=-1)

        sample_points = sample_points.squeeze(1).gather(1, topk_ind)
        if self.num_random_points > 0:
            sample_points = torch.cat(
                [sample_points,
                 torch.rand([num_masks, self.num_random_points, 2])],
                dim=1)
        return sample_points

