import torch
from torch import nn

import sys
sys.path.append("..")
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class OrderInvariantLoss(nn.Module):
    def __init__(self, num_tokens, matcher, bb_weight=1, class_weight=1, giou_weight=1):
        super().__init__()
        self.num_tokens = num_tokens
        self.matcher = matcher
        self.bb_weight = bb_weight
        self.class_weight = class_weight
        self.giou_weight = giou_weight

        self.bb_loss = nn.L1Loss()
        self.class_loss = nn.CrossEntropyLoss()


    def forward(self, predictions, targets):
        """

        Args:
            predictions (_type_): predictions is a dict {
                bbs     : Tensor (N, S, 4),
                classes : Tensor (N, S, num_tokens)
                }

            targets (_type_): targets a dict {
                bbs:     Tensor (N, S, 4),
                classes: Tensor (N, S)
                }

        """
        # predictions a dict {
        #   bbs     : Tensor (N, S, 4),
        #   classes : Tensor (N, S, num_tokens)
        # }
        #
        # targets a dict {
        #   bbs:     Tensor (N, S, 4),
        #   classes: Tensor (N, S)
        # }

        matched_indices = self.matcher(predictions, targets)

        sorted_predictions_indices = matched_indices["predictions"]
        sorted_targets_indices = matched_indices["targets"]

        predicted_boxes = predictions["bbs"]
        
        bs = predicted_boxes.shape[0]
        num_predictions = predicted_boxes.shape[1]


        # select optimal order of boxes

        predicted_boxes = predicted_boxes.view(bs*num_predictions, -1)[sorted_predictions_indices.view(bs*num_predictions)]
        predicted_boxes = predicted_boxes.view(bs, num_predictions, -1)

        # select optimal order of classes 
        predicted_classes = predictions["classes"]
        predicted_classes = predicted_classes.view(bs*num_predictions, -1)[sorted_predictions_indices]
        predicted_classes = predicted_classes.view(bs, num_predictions, -1)

        # select optimal order of targets

        target_boxes = targets["bbs"]
        target_boxes = target_boxes.view(bs*num_predictions, -1)[sorted_targets_indices]
        target_boxes = target_boxes.view(bs, num_predictions, -1)

        target_classes = targets["classes"]
        target_classes = target_classes.view(-1)[sorted_targets_indices]
        target_classes = target_classes.view(bs, num_predictions)



        bb_losss = self.bb_loss(predicted_boxes.view(bs*num_predictions, -1), target_boxes.view(bs*num_predictions, -1))
        class_loss = self.class_loss(predicted_classes.view(bs*num_predictions, -1), target_classes.view(-1))
        giou_loss =  1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(predicted_boxes.view(bs*num_predictions, -1)),
            box_cxcywh_to_xyxy(target_boxes.view(bs*num_predictions, -1)))
        )

        giou_loss = giou_loss.mean()

        loss = self.bb_weight*bb_losss + self.class_weight*class_loss + self.giou_weight*giou_loss

        return loss, {
            "bb_loss": bb_losss.item(),
            "giou_loss": giou_loss.item(),
            "class_loss": class_loss.item(),
        }


if __name__ == "__main__":
    predictions = {
        "bbs": torch.rand(1, 3, 4).sigmoid(),
        "classes": torch.rand(1, 3, 5)
    }

    targets = {
        "bbs": torch.rand(1, 3, 4).sigmoid(),
        "classes": torch.randint(0, 5, (1, 3)).long()
    }

    lengths = torch.Tensor([2, 3, 3])

    from matcher import HungarianMatcher2

    matcher = HungarianMatcher2()

    lossfn = OrderInvariantLoss(num_tokens=5, matcher=matcher)

    loss = lossfn(predictions, targets)

    print(loss)