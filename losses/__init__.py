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
        for config in loss_configs:
            losses.append(self.create_loss(config))
        
        return CombinedLoss(losses, weights_list)
    


class CombinedLoss(nn.Module):
    def __init__(self, losses: list, weights: list):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            loss_value = loss_fn(input, target)
            total_loss += weight * loss_value
        return total_loss