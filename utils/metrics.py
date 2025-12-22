"""
Metrics calculation utilities for segmentation
Supports both binary and multi-class segmentation
"""
import torch
import torch.nn.functional as F


def calculate_dice_score(pred, target, num_classes=None, threshold=0.5, smooth=1e-5):
    """
    Calculate Dice score for binary or multi-class segmentation

    Args:
        pred: Predicted logits or probabilities
              - Binary: shape (B, 1, H, W) or (B, 1, D, H, W)
              - Multi-class: shape (B, C, H, W) or (B, C, D, H, W)
        target: Ground truth labels, same shape as pred
        num_classes: Number of classes. If None, inferred from pred.shape[1]
        threshold: Threshold for binary predictions (only used when num_classes=1)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Dice score (scalar) - mean across all classes
    """
    with torch.no_grad():
        if num_classes is None:
            num_classes = pred.shape[1]

        if num_classes == 1:
            # Binary segmentation
            # Apply sigmoid if predictions are logits
            if pred.max() > 1.0 or pred.min() < 0.0:
                pred = torch.sigmoid(pred)

            # Threshold predictions
            pred_binary = (pred > threshold).float()

            # Flatten tensors
            pred_flat = pred_binary.view(-1)
            target_flat = target.view(-1)

            # Calculate intersection and union
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()

            # Calculate Dice score
            dice = (2.0 * intersection + smooth) / (union + smooth)
            return dice.item()

        else:
            # Multi-class segmentation
            # Apply softmax if predictions are logits
            if pred.max() > 1.0 or pred.min() < 0.0:
                pred = torch.softmax(pred, dim=1)

            # Convert to class predictions
            pred_classes = torch.argmax(pred, dim=1, keepdim=True)
            target_classes = target if target.shape[1] == 1 else torch.argmax(target, dim=1, keepdim=True)

            # Calculate Dice for each class and average
            dice_scores = []
            for class_idx in range(num_classes):
                pred_class = (pred_classes == class_idx).float()
                target_class = (target_classes == class_idx).float()

                pred_flat = pred_class.view(-1)
                target_flat = target_class.view(-1)

                intersection = (pred_flat * target_flat).sum()
                union = pred_flat.sum() + target_flat.sum()

                if union > 0:  # Only calculate if class exists
                    dice = (2.0 * intersection + smooth) / (union + smooth)
                    dice_scores.append(dice.item())

            return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0


def calculate_iou(pred, target, num_classes=None, threshold=0.5, smooth=1e-5):
    """
    Calculate IoU (Intersection over Union) for binary or multi-class segmentation

    Args:
        pred: Predicted logits or probabilities
        target: Ground truth labels
        num_classes: Number of classes. If None, inferred from pred.shape[1]
        threshold: Threshold for binary predictions (only used when num_classes=1)
        smooth: Smoothing factor to avoid division by zero

    Returns:
        IoU score (scalar) - mean across all classes
    """
    with torch.no_grad():
        if num_classes is None:
            num_classes = pred.shape[1]

        if num_classes == 1:
            # Binary segmentation
            if pred.max() > 1.0 or pred.min() < 0.0:
                pred = torch.sigmoid(pred)

            pred_binary = (pred > threshold).float()

            pred_flat = pred_binary.view(-1)
            target_flat = target.view(-1)

            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum() - intersection

            iou = (intersection + smooth) / (union + smooth)
            return iou.item()

        else:
            # Multi-class segmentation
            if pred.max() > 1.0 or pred.min() < 0.0:
                pred = torch.softmax(pred, dim=1)

            pred_classes = torch.argmax(pred, dim=1, keepdim=True)
            target_classes = target if target.shape[1] == 1 else torch.argmax(target, dim=1, keepdim=True)

            iou_scores = []
            for class_idx in range(num_classes):
                pred_class = (pred_classes == class_idx).float()
                target_class = (target_classes == class_idx).float()

                pred_flat = pred_class.view(-1)
                target_flat = target_class.view(-1)

                intersection = (pred_flat * target_flat).sum()
                union = pred_flat.sum() + target_flat.sum() - intersection

                if union > 0:
                    iou = (intersection + smooth) / (union + smooth)
                    iou_scores.append(iou.item())

            return sum(iou_scores) / len(iou_scores) if iou_scores else 0.0


def calculate_metrics(pred, target, num_classes=None, threshold=0.5):
    """
    Calculate multiple metrics for binary or multi-class segmentation

    Args:
        pred: Predicted logits or probabilities
        target: Ground truth labels
        num_classes: Number of classes. If None, inferred from pred.shape[1]
        threshold: Threshold for binary predictions

    Returns:
        Dictionary of metrics
    """
    if num_classes is None:
        num_classes = pred.shape[1]

    dice = calculate_dice_score(pred, target, num_classes, threshold)
    iou = calculate_iou(pred, target, num_classes, threshold)

    return {
        'dice': dice,
        'iou': iou
    }
