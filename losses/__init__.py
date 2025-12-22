import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
from monai.losses import (
    DiceLoss,
    DiceCELoss,
    DiceFocalLoss,
    FocalLoss,
    TverskyLoss,
    GeneralizedDiceLoss,
    MaskedDiceLoss,
    ContrastiveLoss,
    LocalNormalizedCrossCorrelationLoss,
)


class SegmentationLossManager:
    def __init__(self, num_classes=None):
        """
        Args:
            num_classes: Number of output classes. If None, will be inferred from config.
                        - 1 for binary segmentation (uses sigmoid)
                        - >1 for multi-class segmentation (uses softmax)
        """
        self.num_classes = num_classes
        self.loss_registry = {
            'dice': DiceLoss,
            'dice_ce': DiceCELoss,
            'dice_focal': DiceFocalLoss,
            'focal': FocalLoss,
            'tversky': TverskyLoss,
            'generalized_dice': GeneralizedDiceLoss,
            'masked_dice': MaskedDiceLoss,
            'contrastive': ContrastiveLoss,
            'lncc': LocalNormalizedCrossCorrelationLoss,
        }

    def _auto_adjust_params(self, loss_type: str, loss_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically adjust loss parameters based on number of classes.

        For binary segmentation (num_classes=1):
            - Dice-based: Use sigmoid=True, softmax=False, to_onehot_y=False
            - Focal: Use to_onehot_y=False, use_softmax=False

        For multi-class segmentation (num_classes>1):
            - Dice-based: Use sigmoid=False, softmax=True, to_onehot_y=True
            - Focal: Use to_onehot_y=True, use_softmax=True
        """
        params = loss_params.copy()

        # Determine if this is binary or multi-class
        is_binary = self.num_classes == 1

        # Dice-based losses: support sigmoid/softmax parameters
        if loss_type in ['dice', 'dice_ce', 'dice_focal', 'tversky', 'generalized_dice']:
            if is_binary:
                # Binary segmentation
                if 'sigmoid' not in params:
                    params['sigmoid'] = True
                if 'softmax' not in params:
                    params['softmax'] = False
                if 'to_onehot_y' not in params:
                    params['to_onehot_y'] = False
            else:
                # Multi-class segmentation
                if 'sigmoid' not in params:
                    params['sigmoid'] = False
                if 'softmax' not in params:
                    params['softmax'] = True
                if 'to_onehot_y' not in params:
                    params['to_onehot_y'] = True

        # FocalLoss: uses use_softmax parameter instead of sigmoid/softmax
        elif loss_type == 'focal':
            if is_binary:
                # Binary segmentation
                if 'to_onehot_y' not in params:
                    params['to_onehot_y'] = False
                # FocalLoss uses use_softmax parameter
                if 'use_softmax' not in params:
                    params['use_softmax'] = False  # Use sigmoid for binary
            else:
                # Multi-class segmentation
                if 'to_onehot_y' not in params:
                    params['to_onehot_y'] = True
                if 'use_softmax' not in params:
                    params['use_softmax'] = True

        return params

    def create_loss(self, loss_config: Dict[str, Any]) -> nn.Module:

        loss_type = loss_config.get('type', 'dice').lower()
        loss_params = loss_config.get('params', {})

        if loss_type not in self.loss_registry:
            available_losses = ', '.join(self.loss_registry.keys())
            raise ValueError(f"Unsupported loss type: {loss_type}. Available types: {available_losses}")

        # Auto-adjust parameters if num_classes is set
        if self.num_classes is not None:
            loss_params = self._auto_adjust_params(loss_type, loss_params)

        loss_class = self.loss_registry[loss_type]

        try:
            loss_fn = loss_class(**loss_params)
            class_type = "binary" if self.num_classes == 1 else f"multi-class ({self.num_classes} classes)"
            print(f"Created {loss_type} loss for {class_type}, parameters: {loss_params}")
            return loss_fn
        except Exception as e:
            raise ValueError(f"Create {loss_type} loss failed : {str(e)}")
    
    def create_combined_loss(self, loss_configs: list, weights_list: list) -> 'CombinedLoss':
        losses = []
        names = []
        for config in loss_configs:
            losses.append(self.create_loss(config))
            names.append(config.get('type', 'loss'))
        
        return CombinedLoss(losses, weights_list, names)
    


class CombinedLoss(nn.Module):
    def __init__(self, losses: list, weights: list, names: list = None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights
        self.names = names if names is not None else [l.__class__.__name__.lower() for l in losses]
        self.last_components = {}
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        components = {}
        for name, loss_fn, weight in zip(self.names, self.losses, self.weights):
            loss_value = loss_fn(input, target)
            components[name] = loss_value
            total_loss += weight * loss_value
        # cache last unweighted component values for logging
        self.last_components = components
        return total_loss

    @torch.no_grad()
    def components(self) -> dict:
        """Return the last computed unweighted component losses as a dict."""
        return {k: v.detach() if torch.is_tensor(v) else v for k, v in self.last_components.items()}
