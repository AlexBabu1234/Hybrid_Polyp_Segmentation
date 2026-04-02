import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------
# Dice Loss
# --------------------------------
class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):

        preds = torch.sigmoid(preds)

        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (preds * targets).sum()

        dice = (2 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


# --------------------------------
# Boundary Loss (Edge-aware)
# --------------------------------
class BoundaryLoss(nn.Module):

    def __init__(self):
        super(BoundaryLoss, self).__init__()

        # Sobel filters for edge extraction
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        )

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32
        )

        self.sobel_x = sobel_x.view(1, 1, 3, 3)
        self.sobel_y = sobel_y.view(1, 1, 3, 3)

    def forward(self, preds, targets):

        preds = torch.sigmoid(preds)

        sobel_x = self.sobel_x.to(preds.device)
        sobel_y = self.sobel_y.to(preds.device)

        pred_edge_x = F.conv2d(preds, sobel_x, padding=1)
        pred_edge_y = F.conv2d(preds, sobel_y, padding=1)

        target_edge_x = F.conv2d(targets, sobel_x, padding=1)
        target_edge_y = F.conv2d(targets, sobel_y, padding=1)

        eps = 1e-6

        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + eps)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + eps)

        return F.l1_loss(pred_edge, target_edge)


# --------------------------------
# Final Hybrid Loss
# --------------------------------
class HybridLoss(nn.Module):

    def __init__(self):
        super(HybridLoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()

    def forward(self, preds, targets):

        # segmentation losses
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)

        # edge supervision
        boundary_loss = self.boundary(preds, targets)

        # weighted combination
        total_loss = (
            bce_loss +
            dice_loss +
            0.3 * boundary_loss
        )

        return total_loss