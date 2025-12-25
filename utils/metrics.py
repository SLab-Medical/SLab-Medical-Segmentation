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


def calculate_precision(pred, target, num_classes=None, threshold=0.5, smooth=1e-5):
    """
    Calculate Precision (Positive Predictive Value) for binary or multi-class segmentation
    Precision = TP / (TP + FP)

    Args:
        pred: Predicted logits or probabilities
        target: Ground truth labels
        num_classes: Number of classes. If None, inferred from pred.shape[1]
        threshold: Threshold for binary predictions
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Precision score (scalar) - mean across all classes
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

            tp = (pred_flat * target_flat).sum()
            fp = (pred_flat * (1 - target_flat)).sum()

            precision = (tp + smooth) / (tp + fp + smooth)
            return precision.item()

        else:
            # Multi-class segmentation
            if pred.max() > 1.0 or pred.min() < 0.0:
                pred = torch.softmax(pred, dim=1)

            pred_classes = torch.argmax(pred, dim=1, keepdim=True)
            target_classes = target if target.shape[1] == 1 else torch.argmax(target, dim=1, keepdim=True)

            precision_scores = []
            for class_idx in range(num_classes):
                pred_class = (pred_classes == class_idx).float()
                target_class = (target_classes == class_idx).float()

                pred_flat = pred_class.view(-1)
                target_flat = target_class.view(-1)

                tp = (pred_flat * target_flat).sum()
                fp = (pred_flat * (1 - target_flat)).sum()

                if (tp + fp) > 0:
                    precision = (tp + smooth) / (tp + fp + smooth)
                    precision_scores.append(precision.item())

            return sum(precision_scores) / len(precision_scores) if precision_scores else 0.0


def calculate_recall(pred, target, num_classes=None, threshold=0.5, smooth=1e-5):
    """
    Calculate Recall (Sensitivity/True Positive Rate) for binary or multi-class segmentation
    Recall = TP / (TP + FN)

    Args:
        pred: Predicted logits or probabilities
        target: Ground truth labels
        num_classes: Number of classes. If None, inferred from pred.shape[1]
        threshold: Threshold for binary predictions
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Recall score (scalar) - mean across all classes
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

            tp = (pred_flat * target_flat).sum()
            fn = ((1 - pred_flat) * target_flat).sum()

            recall = (tp + smooth) / (tp + fn + smooth)
            return recall.item()

        else:
            # Multi-class segmentation
            if pred.max() > 1.0 or pred.min() < 0.0:
                pred = torch.softmax(pred, dim=1)

            pred_classes = torch.argmax(pred, dim=1, keepdim=True)
            target_classes = target if target.shape[1] == 1 else torch.argmax(target, dim=1, keepdim=True)

            recall_scores = []
            for class_idx in range(num_classes):
                pred_class = (pred_classes == class_idx).float()
                target_class = (target_classes == class_idx).float()

                pred_flat = pred_class.view(-1)
                target_flat = target_class.view(-1)

                tp = (pred_flat * target_flat).sum()
                fn = ((1 - pred_flat) * target_flat).sum()

                if (tp + fn) > 0:
                    recall = (tp + smooth) / (tp + fn + smooth)
                    recall_scores.append(recall.item())

            return sum(recall_scores) / len(recall_scores) if recall_scores else 0.0


def calculate_specificity(pred, target, num_classes=None, threshold=0.5, smooth=1e-5):
    """
    Calculate Specificity (True Negative Rate) for binary or multi-class segmentation
    Specificity = TN / (TN + FP)

    Args:
        pred: Predicted logits or probabilities
        target: Ground truth labels
        num_classes: Number of classes. If None, inferred from pred.shape[1]
        threshold: Threshold for binary predictions
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Specificity score (scalar) - mean across all classes
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

            tn = ((1 - pred_flat) * (1 - target_flat)).sum()
            fp = (pred_flat * (1 - target_flat)).sum()

            specificity = (tn + smooth) / (tn + fp + smooth)
            return specificity.item()

        else:
            # Multi-class segmentation
            if pred.max() > 1.0 or pred.min() < 0.0:
                pred = torch.softmax(pred, dim=1)

            pred_classes = torch.argmax(pred, dim=1, keepdim=True)
            target_classes = target if target.shape[1] == 1 else torch.argmax(target, dim=1, keepdim=True)

            specificity_scores = []
            for class_idx in range(num_classes):
                pred_class = (pred_classes == class_idx).float()
                target_class = (target_classes == class_idx).float()

                pred_flat = pred_class.view(-1)
                target_flat = target_class.view(-1)

                tn = ((1 - pred_flat) * (1 - target_flat)).sum()
                fp = (pred_flat * (1 - target_flat)).sum()

                if (tn + fp) > 0:
                    specificity = (tn + smooth) / (tn + fp + smooth)
                    specificity_scores.append(specificity.item())

            return sum(specificity_scores) / len(specificity_scores) if specificity_scores else 0.0


def calculate_f1_score(pred, target, num_classes=None, threshold=0.5, smooth=1e-5):
    """
    Calculate F1 Score (harmonic mean of Precision and Recall)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        pred: Predicted logits or probabilities
        target: Ground truth labels
        num_classes: Number of classes. If None, inferred from pred.shape[1]
        threshold: Threshold for binary predictions
        smooth: Smoothing factor to avoid division by zero

    Returns:
        F1 score (scalar) - mean across all classes
    """
    precision = calculate_precision(pred, target, num_classes, threshold, smooth)
    recall = calculate_recall(pred, target, num_classes, threshold, smooth)

    f1 = (2 * precision * recall) / (precision + recall + smooth)
    return f1


def calculate_accuracy(pred, target, num_classes=None, threshold=0.5):
    """
    Calculate Pixel/Voxel Accuracy for binary or multi-class segmentation
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Args:
        pred: Predicted logits or probabilities
        target: Ground truth labels
        num_classes: Number of classes. If None, inferred from pred.shape[1]
        threshold: Threshold for binary predictions

    Returns:
        Accuracy score (scalar)
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

            correct = (pred_flat == target_flat).sum()
            total = pred_flat.numel()

            accuracy = correct / total
            return accuracy.item()

        else:
            # Multi-class segmentation
            if pred.max() > 1.0 or pred.min() < 0.0:
                pred = torch.softmax(pred, dim=1)

            pred_classes = torch.argmax(pred, dim=1)
            target_classes = target if target.dim() == pred_classes.dim() else torch.argmax(target, dim=1)

            correct = (pred_classes == target_classes).sum()
            total = pred_classes.numel()

            accuracy = correct / total
            return accuracy.item()


def calculate_hausdorff_distance(pred, target, num_classes=None, threshold=0.5, percentile=95):
    """
    Calculate Hausdorff Distance (95th percentile) for binary or multi-class segmentation
    Measures the maximum surface distance between prediction and ground truth

    Args:
        pred: Predicted logits or probabilities
        target: Ground truth labels
        num_classes: Number of classes. If None, inferred from pred.shape[1]
        threshold: Threshold for binary predictions
        percentile: Percentile for robust Hausdorff distance (default 95)

    Returns:
        Hausdorff distance (scalar) - mean across all classes
    """
    try:
        from scipy.ndimage import distance_transform_edt
        import numpy as np
    except ImportError:
        print("Warning: scipy not available, skipping Hausdorff distance calculation")
        return 0.0

    with torch.no_grad():
        if num_classes is None:
            num_classes = pred.shape[1]

        if num_classes == 1:
            # Binary segmentation
            if pred.max() > 1.0 or pred.min() < 0.0:
                pred = torch.sigmoid(pred)

            pred_binary = (pred > threshold).float().cpu().numpy()
            target_binary = target.cpu().numpy()

            # Calculate Hausdorff distance
            hd = _compute_hausdorff_distance(pred_binary, target_binary, percentile)
            return hd

        else:
            # Multi-class segmentation
            if pred.max() > 1.0 or pred.min() < 0.0:
                pred = torch.softmax(pred, dim=1)

            pred_classes = torch.argmax(pred, dim=1, keepdim=True).cpu().numpy()
            target_classes = target if target.shape[1] == 1 else torch.argmax(target, dim=1, keepdim=True)
            target_classes = target_classes.cpu().numpy()

            hd_scores = []
            for class_idx in range(num_classes):
                pred_class = (pred_classes == class_idx).astype(np.float32)
                target_class = (target_classes == class_idx).astype(np.float32)

                # Only calculate if both pred and target have this class
                if pred_class.sum() > 0 and target_class.sum() > 0:
                    hd = _compute_hausdorff_distance(pred_class, target_class, percentile)
                    hd_scores.append(hd)

            return sum(hd_scores) / len(hd_scores) if hd_scores else 0.0


def _compute_hausdorff_distance(pred, target, percentile=95):
    """
    Helper function to compute Hausdorff distance between two binary masks

    Args:
        pred: Binary prediction mask (numpy array)
        target: Binary target mask (numpy array)
        percentile: Percentile for robust Hausdorff distance

    Returns:
        Hausdorff distance (scalar)
    """
    from scipy.ndimage import distance_transform_edt
    import numpy as np

    # Get surface points by finding boundaries
    pred_border = pred - np.roll(pred, 1, axis=0)
    target_border = target - np.roll(target, 1, axis=0)

    # Compute distance transforms
    dt_pred = distance_transform_edt(1 - pred)
    dt_target = distance_transform_edt(1 - target)

    # Distances from target border to pred
    dist_target_to_pred = dt_pred[target_border != 0]
    # Distances from pred border to target
    dist_pred_to_target = dt_target[pred_border != 0]

    if len(dist_target_to_pred) == 0 or len(dist_pred_to_target) == 0:
        return 0.0

    # Compute percentile Hausdorff distance
    hd1 = np.percentile(dist_target_to_pred, percentile) if len(dist_target_to_pred) > 0 else 0
    hd2 = np.percentile(dist_pred_to_target, percentile) if len(dist_pred_to_target) > 0 else 0

    return max(hd1, hd2)


def calculate_average_surface_distance(pred, target, num_classes=None, threshold=0.5):
    """
    Calculate Average Surface Distance (ASD) for binary or multi-class segmentation
    Measures the average distance between surfaces

    Args:
        pred: Predicted logits or probabilities
        target: Ground truth labels
        num_classes: Number of classes. If None, inferred from pred.shape[1]
        threshold: Threshold for binary predictions

    Returns:
        Average surface distance (scalar) - mean across all classes
    """
    try:
        from scipy.ndimage import distance_transform_edt
        import numpy as np
    except ImportError:
        print("Warning: scipy not available, skipping ASD calculation")
        return 0.0

    with torch.no_grad():
        if num_classes is None:
            num_classes = pred.shape[1]

        if num_classes == 1:
            # Binary segmentation
            if pred.max() > 1.0 or pred.min() < 0.0:
                pred = torch.sigmoid(pred)

            pred_binary = (pred > threshold).float().cpu().numpy()
            target_binary = target.cpu().numpy()

            asd = _compute_average_surface_distance(pred_binary, target_binary)
            return asd

        else:
            # Multi-class segmentation
            if pred.max() > 1.0 or pred.min() < 0.0:
                pred = torch.softmax(pred, dim=1)

            pred_classes = torch.argmax(pred, dim=1, keepdim=True).cpu().numpy()
            target_classes = target if target.shape[1] == 1 else torch.argmax(target, dim=1, keepdim=True)
            target_classes = target_classes.cpu().numpy()

            asd_scores = []
            for class_idx in range(num_classes):
                pred_class = (pred_classes == class_idx).astype(np.float32)
                target_class = (target_classes == class_idx).astype(np.float32)

                if pred_class.sum() > 0 and target_class.sum() > 0:
                    asd = _compute_average_surface_distance(pred_class, target_class)
                    asd_scores.append(asd)

            return sum(asd_scores) / len(asd_scores) if asd_scores else 0.0


def _compute_average_surface_distance(pred, target):
    """
    Helper function to compute average surface distance between two binary masks

    Args:
        pred: Binary prediction mask (numpy array)
        target: Binary target mask (numpy array)

    Returns:
        Average surface distance (scalar)
    """
    from scipy.ndimage import distance_transform_edt
    import numpy as np

    # Get surface points
    pred_border = pred - np.roll(pred, 1, axis=0)
    target_border = target - np.roll(target, 1, axis=0)

    # Compute distance transforms
    dt_pred = distance_transform_edt(1 - pred)
    dt_target = distance_transform_edt(1 - target)

    # Distances from surfaces
    dist_target_to_pred = dt_pred[target_border != 0]
    dist_pred_to_target = dt_target[pred_border != 0]

    if len(dist_target_to_pred) == 0 and len(dist_pred_to_target) == 0:
        return 0.0

    # Average of all surface distances
    all_distances = np.concatenate([dist_target_to_pred, dist_pred_to_target])
    return np.mean(all_distances)


def calculate_metrics(pred, target, num_classes=None, threshold=0.5, include_distance_metrics=False):
    """
    Calculate comprehensive metrics for binary or multi-class segmentation

    Args:
        pred: Predicted logits or probabilities
        target: Ground truth labels
        num_classes: Number of classes. If None, inferred from pred.shape[1]
        threshold: Threshold for binary predictions
        include_distance_metrics: Whether to include Hausdorff and ASD (computationally expensive)

    Returns:
        Dictionary of metrics including:
        - dice: Dice coefficient
        - iou: Intersection over Union
        - precision: Precision (PPV)
        - recall: Recall (Sensitivity/TPR)
        - specificity: Specificity (TNR)
        - f1: F1 Score
        - accuracy: Pixel/Voxel accuracy
        - hausdorff_distance: 95th percentile Hausdorff distance (if enabled)
        - avg_surface_distance: Average surface distance (if enabled)
    """
    if num_classes is None:
        num_classes = pred.shape[1]

    # Core metrics (fast)
    dice = calculate_dice_score(pred, target, num_classes, threshold)
    iou = calculate_iou(pred, target, num_classes, threshold)
    precision = calculate_precision(pred, target, num_classes, threshold)
    recall = calculate_recall(pred, target, num_classes, threshold)
    specificity = calculate_specificity(pred, target, num_classes, threshold)
    f1 = calculate_f1_score(pred, target, num_classes, threshold)
    accuracy = calculate_accuracy(pred, target, num_classes, threshold)

    metrics = {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'accuracy': accuracy
    }

    # Distance-based metrics (slow, optional)
    if include_distance_metrics:
        try:
            hausdorff = calculate_hausdorff_distance(pred, target, num_classes, threshold)
            asd = calculate_average_surface_distance(pred, target, num_classes, threshold)
            metrics['hausdorff_distance'] = hausdorff
            metrics['avg_surface_distance'] = asd
        except Exception as e:
            print(f"Warning: Could not calculate distance metrics: {e}")

    return metrics
