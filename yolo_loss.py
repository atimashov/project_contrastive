import torch
import torch.nn as nn

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

class YoloLoss(nn.Module):
    def __init__(self, S = 7, B = 2):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction = 'sum')
        self.S = S
        self.B = B
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # TODO: Target has only 1 object per grid cell
        predictions = predictions.reshape(-1, self.S, self.S, self.B * 5)
        iou_b1 = intersection_over_union(predictions[..., 1:5], target[..., 1:5])
        iou_b2 = intersection_over_union(predictions[..., 6:10], target[..., 1:5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)
        ious_maxes, bestbox = torch.max(ious, dim = 0) # bestbox can be either 0 or 1
        exists_box = target[..., 0].unsqueeze(3) # Identity of object existance for grid cells
        # print('bestbox: ', bestbox.shape)
        # ======================== #
        # FOR BOX COORDINATES LOSS #
        # ======================== #
        box_predictions = exists_box * (
            bestbox * predictions[..., 6:10]
            + (1 - bestbox) * predictions[..., 1:5]
        )
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )  # TODO: is there case when w and h are negative?

        box_targets = exists_box * target[..., 1:5]
        box_targets[..., 2:4] = torch.sqrt(box_targets[...,2:4])

        # (N, S, S, 4) -> (N * S * S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim = -2),
            torch.flatten(box_targets, end_dim = -2)
        )

        # =============== #
        # FOR OBJECT LOSS #
        # =============== #
        pred_box = (
                bestbox * predictions[..., 5:6] + (1 - bestbox) * predictions[..., 0:1]
        )

        # N * S * S
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 0:1])
        )

        # ================== #
        # FOR NO OBJECT LOSS #
        # ================== #
        # (N, S, S, 1) -> (N, S * S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 0:1], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., 0:1], start_dim = 1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 5:6], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., 0:1], start_dim = 1)
        )

        # # ================== #
        # # FOR CLASS LOSS #
        # # ================== #
        # # (N, S, S, 20) -> (N * S * S, 20)
        # class_loss = self.mse(
        #     torch.flatten(exists_box * predictions[..., :self.C], end_dim = -2),
        #     torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        # )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            # + class_loss
        )

        return loss / predictions.shape[0] # added mean instead of absolute

def test(S = 7, B = 2):
    target = torch.ones((13, S,  S,  5 * B))
    predictions = torch.randn((13, S * S * (5 * B)))
    print(target.shape, predictions.shape)
    loss_func = YoloLoss(S = S, B = B)
    loss = loss_func(predictions, target)
    print(loss)

if __name__=='__main__':
	test()