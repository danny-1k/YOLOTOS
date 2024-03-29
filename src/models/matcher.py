import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

import sys
sys.path.append("..")
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
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
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(
            0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = - \
            generalized_box_iou(box_cxcywh_to_xyxy(
                out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * \
            cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianMatcher2(nn.Module):
    # Addapted from FB research DETR official implementation

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, predictions, targets):
        """Performs the Matching

        Args:
            predictions : This is a dict that contains at least these entries:
                 "classes": Tensor of dim [batch_size, num_predictions, num_tokens] with the classification logits
                 "bbs": Tensor of dim [batch_size, num_predictions, 4] with the predicted box coordinates

            targets : This is a dict that contains at least these entries:
                 "classes": Tensor of dim [batch_size, num_predictions] with the classification logits
                 "bbs": Tensor of dim [batch_size, num_predictions, 4] with the predicted box coordinates

        Returns:
            A dict with entries:
                "predictions": Tensor of dim [batch_size, num_predictions] with the indices of the selected predictions (in order)
                "targets": Tensor of dim [batch_size, num_predictions] with the indices of the selected targets (in order)
        """


        predicted_bbs = predictions["bbs"]
        predicted_classes = predictions["classes"]

        target_bbs = targets["bbs"]
        target_classes = targets["classes"]

        bs, num_predictions = predicted_bbs.shape[:2]

        predicted_bbs = predicted_bbs.view(bs*num_predictions, -1) # flatten & softmax -> (N * S, 4)
        predicted_classes = predicted_classes.view(bs*num_predictions, -1).softmax(-1) # flatten & softmax -> (N * S, n_tokens)

        target_bbs = target_bbs.view(bs*num_predictions, -1)
        target_classes = target_classes.view(bs*num_predictions)

        cost_class = -predicted_classes[:, target_classes]

        cost_bbs = torch.cdist(predicted_bbs, target_bbs, p=1) # euclid distance

        cost_giou = generalized_box_iou(box_cxcywh_to_xyxy(
                predicted_bbs), box_cxcywh_to_xyxy(target_bbs)) #giou cost
        
        # cost matrix

        C = self.cost_bbox * cost_bbs + self.cost_class * \
            cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_predictions, -1).cpu()

        sizes = tuple(num_predictions for _ in range(bs))

        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(C.split(sizes, -1))]
        

        prediction_indices, target_indices = list(zip(*indices))

        prediction_indices = torch.as_tensor(prediction_indices)
        target_indices = torch.as_tensor(target_indices)

        return {
            "predictions": prediction_indices,
            "targets": target_indices
        }




def build_matcher(args):
    return HungarianMatcher2(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)


if __name__ == "__main__":

    matcher = HungarianMatcher2()

    predictions = {
        "bbs": torch.rand(1, 3, 4).sigmoid(),
        "classes": torch.Tensor(1, 3, 5)
    }

    targets = {
        "bbs": torch.rand(1, 3, 4).sigmoid(),
        "classes": torch.randint(0, 5, (1, 3)).long()
    }

    lengths = torch.Tensor([2, 3, 3])

    print(targets["bbs"])

    indices = matcher(predictions, targets)