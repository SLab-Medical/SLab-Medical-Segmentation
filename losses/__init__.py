import argparse
import yaml
import json
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
    def __init__(self):
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
        
    def create_loss(self, loss_config: Dict[str, Any]) -> nn.Module:

        loss_type = loss_config.get('type', 'dice').lower()
        loss_params = loss_config.get('params', {})
        
        if loss_type not in self.loss_registry:
            available_losses = ', '.join(self.loss_registry.keys())
            raise ValueError(f"Unsupported loss type: {loss_type}. Available types: {available_losses}")
        
        loss_class = self.loss_registry[loss_type]
        
        try:
            loss_fn = loss_class(**loss_params)
            print(f"Create {loss_type} loss, parameters: {loss_params}")
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
