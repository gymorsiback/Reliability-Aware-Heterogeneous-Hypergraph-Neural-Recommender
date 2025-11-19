"""
Loss Functions for Model Placement Task

Author: AI Assistant
Date: 2025-10-30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlacementLoss(nn.Module):
    """
    Combined loss for model placement optimization
    Balances placement accuracy, load balance, and reliability
    """
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super(PlacementLoss, self).__init__()
        self.alpha = alpha  # Placement accuracy weight
        self.beta = beta    # Load balance weight
        self.gamma = gamma  # Consistency weight
        
    def forward(self, pred_placement, true_placement):
        """
        Args:
            pred_placement: Predicted placement scores [num_models, num_servers]
            true_placement: Ground truth placements [num_models, num_servers]
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # L1: Binary cross-entropy for placement accuracy
        loss_placement = F.binary_cross_entropy(
            pred_placement, true_placement, reduction='mean'
        )
        
        # L2: Load balance (variance of server loads)
        # Sum of predicted placements per server
        server_loads = pred_placement.sum(dim=0)  # [num_servers]
        # We want even distribution, so minimize variance
        mean_load = server_loads.mean()
        loss_balance = ((server_loads - mean_load) ** 2).mean()
        
        # L3: Consistency (models with similar characteristics should have similar placements)
        # This is optional and can be expensive to compute
        # For now, we'll use a simple penalty for extreme predictions
        loss_consistency = torch.mean((pred_placement - 0.5) ** 2)
        
        # Combine losses
        total_loss = (self.alpha * loss_placement + 
                     self.beta * loss_balance + 
                     self.gamma * loss_consistency)
        
        loss_dict = {
            'total': total_loss.item(),
            'placement': loss_placement.item(),
            'balance': loss_balance.item(),
            'consistency': loss_consistency.item()
        }
        
        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Useful when some model-server pairs are much more common than others
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class WeightedBCELoss(nn.Module):
    """
    Weighted BCE Loss that gives more weight to positive examples
    """
    def __init__(self, pos_weight=10.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred, target):
        loss = F.binary_cross_entropy(pred, target, reduction='none')
        # Apply higher weight to positive examples
        weights = torch.where(target > 0.5, 
                            torch.ones_like(target) * self.pos_weight,
                            torch.ones_like(target))
        weighted_loss = (loss * weights).mean()
        return weighted_loss

