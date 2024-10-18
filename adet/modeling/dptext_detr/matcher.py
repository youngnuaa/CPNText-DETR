"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from adet.utils.misc import box_cxcywh_to_xyxy, generalized_box_iou, box_cxcylwlhrwrh_to_xyxy
import numpy as np

class BoxHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            giou_weight: float = 1,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        """Creates the matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            giou_weight: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.giou_weight = giou_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        assert class_weight != 0 or coord_weight != 0 or giou_weight != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * \
                ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - \
                neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            """
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox),
                box_cxcywh_to_xyxy(tgt_bbox)
            )
            """
            cost_giou = -generalized_box_iou(
                box_cxcylwlhrwrh_to_xyxy(out_bbox),
                box_cxcylwlhrwrh_to_xyxy(tgt_bbox)
            )

            # Final cost matrix
            C = self.coord_weight * cost_bbox + self.class_weight * \
                cost_class + self.giou_weight * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(
                c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class CtrlPointHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
            self,
            class_weight: float = 1,
            coord_weight: float = 1,
            oks_weight: float = 1,
            giou_weight: float = 1,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0,
            point_box_index=None,
            box_index=None,
    ):
        """Creates the matcher
        Params:
            class_weight: This is the relative weight of the classification error in the matching cost
            coord_weight: This is the relative weight of the L1 error of the keypoint coordinates in the matching cost
        """
        super().__init__()
        self.class_weight = class_weight
        self.coord_weight = coord_weight
        self.oks_weight = oks_weight
        self.giou_weight = giou_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.num_ctrl_points = 16

        if point_box_index is not None:
            point_box_index = np.array(point_box_index)  # n, 4
            point_box_index = point_box_index[:, box_index]  # n, 12
            point_box_index = point_box_index.reshape(-1) # n*12
            self.point_box_index = torch.from_numpy(point_box_index)
        else:
            self.point_box_index = None

        self.sigmas = np.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]) / 10.0

        assert class_weight != 0 or coord_weight != 0, "all costs cant be 0"

    def gen_point_box(self, pre_points, gt_points, index):
        """
        pre_points: bs*inst, 4, 2
        gt_points: bs*inst, 4, 2
        gt_points: n
        """
        # inst_num, _, _ = pre_points.shape
        pre_points = pre_points[:, index, :]
        gt_points = gt_points[:, index, :]
        pre_points = pre_points.reshape(-1, 3, 2)
        gt_points = gt_points.reshape(-1, 3, 2)

        pre_min_x = pre_points[..., 0].min(dim=1)[0].unsqueeze(1)  # inst_num*32
        pre_min_y = pre_points[..., 1].min(dim=1)[0].unsqueeze(1)  # inst_num*32
        pre_max_x = pre_points[..., 0].max(dim=1)[0].unsqueeze(1)  # inst_num*32
        pre_max_y = pre_points[..., 1].max(dim=1)[0].unsqueeze(1)  # inst_num*32
        pre_box = torch.cat([pre_min_x, pre_min_y, pre_max_x, pre_max_y], dim=1)  # inst_num*32, 4

        gt_min_x = gt_points[..., 0].min(dim=1)[0].unsqueeze(1)  # inst_num*32
        gt_min_y = gt_points[..., 1].min(dim=1)[0].unsqueeze(1)  # inst_num*32
        gt_max_x = gt_points[..., 0].max(dim=1)[0].unsqueeze(1)  # inst_num*32
        gt_max_y = gt_points[..., 1].max(dim=1)[0].unsqueeze(1)  # inst_num*32
        gt_box = torch.cat([gt_min_x, gt_min_y, gt_max_x, gt_max_y], dim=1)  # inst_num*32, 4

        return pre_box, gt_box

    def init_prior_points_from_anchor(self, reference_points_anchor):
        # reference_points_anchor: nq, 6 ==> nq, (cx, cy, lw, lh, rw, rh)
        # return size:
        # - reference_points: (bs, nq, n_pts, 2)
        assert reference_points_anchor.shape[-1] == 6

        """
        The best result 89.6
        reference_points_anchor_lh = reference_points_anchor[..., 1] - reference_points_anchor[..., 3]*1.025
        reference_points_anchor_rh = reference_points_anchor[..., 1] + reference_points_anchor[..., 5]*1.025
        reference_points_anchor_lh = torch.clamp(reference_points_anchor_lh, 0, 1)
        reference_points_anchor_rh = torch.clamp(reference_points_anchor_rh, 0, 1)
        reference_points_anchor[..., 3] = reference_points_anchor[..., 1] - reference_points_anchor_lh
        reference_points_anchor[..., 5] = reference_points_anchor_rh - reference_points_anchor[..., 1]
        """

        """
        reference_points_anchor_lh = reference_points_anchor[..., 1] - reference_points_anchor[..., 3]*1.03
        reference_points_anchor_rh = reference_points_anchor[..., 1] + reference_points_anchor[..., 5]*1.03
        reference_points_anchor_lh = torch.clamp(reference_points_anchor_lh, 0, 1)
        reference_points_anchor_rh = torch.clamp(reference_points_anchor_rh, 0, 1)
        reference_points_anchor[..., 3] = reference_points_anchor[..., 1] - reference_points_anchor_lh
        reference_points_anchor[..., 5] = reference_points_anchor_rh - reference_points_anchor[..., 1]
        """



        reference_points = reference_points_anchor[:, None, :].repeat(1, self.num_ctrl_points, 1)
        pts_per_side = self.num_ctrl_points // 4

        reference_points[:, 0, 0].sub_(reference_points[:, 0, 2])
        reference_points[:, 1:pts_per_side, 0] = reference_points[:, 1:pts_per_side, 2] / (pts_per_side - 0.5)
        reference_points[:, :pts_per_side, 0] = torch.cumsum(reference_points[:, :pts_per_side, 0], dim=-1)
        reference_points[:, 3 * pts_per_side:, 0] = reference_points[:, :pts_per_side, 0].flip(dims=[-1])

        reference_points[:, pts_per_side, 0].add_(reference_points[:, pts_per_side, 4] / (pts_per_side - 0.5)/2)
        reference_points[:, pts_per_side + 1:2 * pts_per_side, 0] = reference_points[:,
                                                                       pts_per_side + 1:2 * pts_per_side, 4] / (
                                                                                   pts_per_side - 0.5)
        reference_points[:, pts_per_side:2 * pts_per_side, 0] = torch.cumsum(
            reference_points[:, pts_per_side:2 * pts_per_side, 0], dim=-1)
        reference_points[:, 2 * pts_per_side:3 * pts_per_side, 0] = reference_points[:,
                                                                       pts_per_side:2 * pts_per_side, 0].flip(dims=[-1])

        reference_points[:, :2 * pts_per_side, 1].sub_(reference_points[:, :2 * pts_per_side, 3])
        reference_points[:, 2 * pts_per_side:, 1].add_(reference_points[:, 2 * pts_per_side:, 5])
        reference_points = torch.clamp(reference_points[:, :, :2], 0, 1)
        """

        reference_points = reference_points_anchor[:, None, :].repeat(1, self.num_ctrl_points, 1)
        pts_per_side = self.num_ctrl_points // 4

        reference_points[:, 0, 0].sub_(reference_points[:, 0, 2])
        reference_points[:, 1:pts_per_side, 0] = reference_points[:, 1:pts_per_side, 2] / (pts_per_side - 1)
        reference_points[:, :pts_per_side, 0] = torch.cumsum(reference_points[:, :pts_per_side, 0], dim=-1)
        reference_points[:, 3 * pts_per_side:, 0] = reference_points[:, :pts_per_side, 0].flip(dims=[-1])

        reference_points[:, pts_per_side, 0].add_(reference_points[:, pts_per_side, 4] / (pts_per_side - 0.5))
        reference_points[:, pts_per_side + 1:2 * pts_per_side, 0] = reference_points[:,
                                                                       pts_per_side + 1:2 * pts_per_side, 4] / (
                                                                                   pts_per_side - 0.5)
        reference_points[:, pts_per_side:2 * pts_per_side, 0] = torch.cumsum(
            reference_points[:, pts_per_side:2 * pts_per_side, 0], dim=-1)
        reference_points[:, 2 * pts_per_side:3 * pts_per_side, 0] = reference_points[:,
                                                                       pts_per_side:2 * pts_per_side, 0].flip(dims=[-1])

        reference_points[:, :2 * pts_per_side, 1].sub_(reference_points[:, :2 * pts_per_side, 3])
        reference_points[:, 2 * pts_per_side:, 1].add_(reference_points[:, 2 * pts_per_side:, 5])
        reference_points = torch.clamp(reference_points[:, :, :2], 0, 1)
        """

        return reference_points

    def forward(self, outputs, targets):
        with torch.no_grad():
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()

            # [batch_size, n_queries, n_points, 2] --> [batch_size * num_queries, n_points * 2]
            out_pts = outputs["pred_ctrl_points"].flatten(0, 1)

            # Also concat the target labels and boxes
            tgt_pts = torch.cat([v["ctrl_points"] for v in targets])

            reference_points = self.init_prior_points_from_anchor(tgt_bbox)  # n, k_num, 2
            arget_ctrl_area = (tgt_pts[:, :, 0] - reference_points[:, :, 0]) ** 2 + \
                              (tgt_pts[:, :, 1] - reference_points[:, :, 1]) ** 2  # n, k_num

            sigmas = out_pts.new_tensor(self.sigmas)  #16
            variances = (sigmas * 2) ** 2
            squared_distance = (out_pts[:, None, :, 0] - tgt_pts[None, :, :, 0]) ** 2 + \
                               (out_pts[:, None, :, 1] - tgt_pts[None, :, :, 1]) ** 2
            squared_distance0 = squared_distance / ((arget_ctrl_area+1e-6) * variances[None, :] * 2)
            squared_distance1 = torch.exp(-squared_distance0)
            #squared_distance1 = squared_distance1 * V_gt
            #oks = squared_distance1.sum(dim=-1) / (V_gt.sum(dim=-1) + 1e-6)

            oks = squared_distance1.sum(dim=-1) / out_pts.shape[-2]
            oks = oks.clamp(min=1e-6)
            cost_oks = 1 - oks


            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                             (-(1 - out_prob + 1e-8).log())

            pos_cost_class = self.alpha * \
                             ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            # hack here for label ID 0
            cost_class = (pos_cost_class[..., 0] - neg_cost_class[..., 0]).mean(-1, keepdims=True)

            cost_kpts = torch.cdist(out_pts.flatten(-2), tgt_pts.flatten(-2), p=1)


            if self.point_box_index is not None:
                pre_giou_box, gt_giou_box = self.gen_point_box(out_pts.reshape(out_pts.shape[0], -1, 2),
                                                               tgt_pts.reshape(tgt_pts.shape[0], -1, 2),
                                                               self.point_box_index)

                cost_giou = -generalized_box_iou(pre_giou_box, gt_giou_box)
                #print("cost_giou", cost_giou.shape)
                pre_num, gt_num = cost_kpts.shape
                cost_giou = cost_giou.reshape(pre_num, 20, gt_num, 20)
                cost_giou = cost_giou.mean(-1)
                cost_giou = cost_giou.mean(1)
                if torch.isnan(cost_giou.sum()):
                    C = self.class_weight * cost_class + self.coord_weight * cost_kpts
                else:
                    C = self.class_weight * cost_class + self.coord_weight * cost_kpts + self.giou_weight * cost_giou

            else:
                #print("cost_kpts", cost_kpts.shape)
                C = self.class_weight * cost_class + self.coord_weight * cost_kpts + self.oks_weight*cost_oks

            #C = self.class_weight * cost_class + self.coord_weight * cost_kpts
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["ctrl_points"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            """

            indices = []
            for i, c in enumerate(C.split(sizes, -1)):
                try:
                    indices.append(linear_sum_assignment(c[i]))
                except:
                    #print(c[i])
                    print("cost_kpts", cost_kpts)
                    print("out_pts", out_pts)
                    print("tgt_pts", tgt_pts)
            """

            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    cfg = cfg.MODEL.TRANSFORMER.LOSS
    return BoxHungarianMatcher(class_weight=cfg.BOX_CLASS_WEIGHT,
                               coord_weight=cfg.BOX_COORD_WEIGHT,
                               giou_weight=cfg.BOX_GIOU_WEIGHT,
                               focal_alpha=cfg.FOCAL_ALPHA,
                               focal_gamma=cfg.FOCAL_GAMMA), \
        CtrlPointHungarianMatcher(class_weight=cfg.POINT_CLASS_WEIGHT,
                                 coord_weight=cfg.POINT_COORD_WEIGHT,
                                  oks_weight=cfg.POINT_OKS_WEIGHT,
                                  giou_weight=cfg.POINT_GIOU_WEIGHT,
                                 focal_alpha=cfg.FOCAL_ALPHA,
                                 focal_gamma=cfg.FOCAL_GAMMA,
                                  point_box_index=cfg.POINT_BOX_INDEX,
                                  box_index=cfg.BOX_INDEX
                                  )