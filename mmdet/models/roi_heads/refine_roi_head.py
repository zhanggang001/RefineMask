import torch
import torch.nn.functional as F

from mmdet.core import bbox2roi
from mmdet.models.losses.cross_entropy_loss import generate_block_target
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.builder import HEADS


@HEADS.register_module()
class RefineRoIHead(StandardRoIHead):
    def init_weights(self, pretrained):
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None):
        # assign gts and sample proposals
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels, img_metas)

        mask_results = self._mask_forward_train(
            x, sampling_results, bbox_results['bbox_feats'], gt_bboxes, gt_masks, gt_labels, img_metas)

        losses = {}
        losses.update(bbox_results['loss_bbox'])
        losses.update(mask_results['loss_mask'])
        losses.update(mask_results['loss_semantic'])
        return losses

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_bboxes, gt_masks, gt_labels, img_metas):
        pos_bboxes = [res.pos_bboxes for res in sampling_results]
        pos_labels = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        pos_rois = bbox2roi(pos_bboxes)

        stage_mask_targets, semantic_target = \
            self.mask_head.get_targets(pos_bboxes, pos_assigned_gt_inds, gt_masks)

        mask_results = self._mask_forward(x, pos_rois, torch.cat(pos_labels))

        # resize the semantic target
        semantic_target = F.interpolate(
            semantic_target.unsqueeze(1),
            mask_results['semantic_pred'].shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        semantic_target = (semantic_target >= 0.5).float()

        loss_mask, loss_semantic = self.mask_head.loss(
            mask_results['stage_instance_preds'],
            mask_results['semantic_pred'],
            stage_mask_targets,
            semantic_target)

        mask_results.update(loss_mask=loss_mask, loss_semantic=loss_semantic)
        return mask_results

    def _mask_forward(self, x, rois, roi_labels):
        """Mask head forward function used in both training and testing."""

        ins_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
        stage_instance_preds, semantic_pred = self.mask_head(ins_feats, x[0], rois, roi_labels)
        return dict(stage_instance_preds=stage_instance_preds, semantic_pred=semantic_pred)

    def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Simple test for mask head without augmentation."""

        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
            _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            mask_rois = bbox2roi([_bboxes])

            interval = 100  # to avoid memory overflow
            segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
            for i in range(0, det_labels.shape[0], interval):
                mask_results = self._mask_forward(x, mask_rois[i: i + interval], det_labels[i: i + interval])

                # refine instance masks from stage 1
                stage_instance_preds = mask_results['stage_instance_preds'][1:]
                for idx in range(len(stage_instance_preds) - 1):
                    instance_pred = stage_instance_preds[idx].squeeze(1).sigmoid() >= 0.5
                    non_boundary_mask = (generate_block_target(instance_pred, boundary_width=1) != 1).unsqueeze(1)
                    non_boundary_mask = F.interpolate(
                        non_boundary_mask.float(),
                        stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True) >= 0.5
                    pre_pred = F.interpolate(
                        stage_instance_preds[idx],
                        stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True)
                    stage_instance_preds[idx + 1][non_boundary_mask] = pre_pred[non_boundary_mask]
                instance_pred = stage_instance_preds[-1]

                chunk_segm_result = self.mask_head.get_seg_masks(
                    instance_pred, _bboxes[i: i + interval], det_labels[i: i + interval],
                    self.test_cfg, ori_shape, scale_factor, rescale)

                for c, segm in zip(det_labels[i: i + interval], chunk_segm_result):
                    segm_result[c].append(segm)

        return segm_result


@HEADS.register_module()
class SimpleRefineRoIHead(StandardRoIHead):

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks, img_metas):
        """Run forward function and calculate loss for mask head in training."""

        pos_bboxes = [res.pos_bboxes for res in sampling_results]
        pos_labels = [res.pos_gt_labels for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        pos_rois = bbox2roi(pos_bboxes)

        mask_results = self._mask_forward(x, pos_rois, torch.cat(pos_labels))
        stage_mask_targets = self.mask_head.get_targets(pos_bboxes, pos_assigned_gt_inds, gt_masks)
        loss_mask = self.mask_head.loss(mask_results['stage_instance_preds'], stage_mask_targets)

        mask_results.update(loss_mask=loss_mask)
        return mask_results

    def _mask_forward(self, x, rois, roi_labels):
        """Mask head forward function used in both training and testing."""

        ins_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
        stage_instance_preds = self.mask_head(ins_feats, x[0], rois, roi_labels)
        return dict(stage_instance_preds=stage_instance_preds)

    def simple_test_mask(self, x, img_metas, det_bboxes, det_labels, rescale=False):
        """Simple test for mask head without augmentation."""

        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(det_bboxes.device)
            _bboxes = det_bboxes[:, :4] * scale_factor if rescale else det_bboxes
            mask_rois = bbox2roi([_bboxes])

            interval = 100  # to avoid memory overflow
            segm_result = [[] for _ in range(self.mask_head.stage_num_classes[0])]
            for i in range(0, det_labels.shape[0], interval):
                mask_results = self._mask_forward(x, mask_rois[i: i + interval], det_labels[i: i + interval])

                # refine instance masks from stage 1
                stage_instance_preds = mask_results['stage_instance_preds'][1:]
                for idx in range(len(stage_instance_preds) - 1):
                    instance_pred = stage_instance_preds[idx].squeeze(1).sigmoid() >= 0.5
                    non_boundary_mask = (generate_block_target(instance_pred, boundary_width=1) != 1).unsqueeze(1)
                    non_boundary_mask = F.interpolate(
                        non_boundary_mask.float(),
                        stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True) >= 0.5
                    pre_pred = F.interpolate(
                        stage_instance_preds[idx],
                        stage_instance_preds[idx + 1].shape[-2:], mode='bilinear', align_corners=True)
                    stage_instance_preds[idx + 1][non_boundary_mask] = pre_pred[non_boundary_mask]
                instance_pred = stage_instance_preds[-1]

                chunk_segm_result = self.mask_head.get_seg_masks(
                    instance_pred, _bboxes[i: i + interval], det_labels[i: i + interval],
                    self.test_cfg, ori_shape, scale_factor, rescale)

                for c, segm in zip(det_labels[i: i + interval], chunk_segm_result):
                    segm_result[c].append(segm)

        return segm_result
