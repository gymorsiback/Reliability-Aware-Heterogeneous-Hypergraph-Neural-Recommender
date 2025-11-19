"""
Evaluation Metrics for Model Placement Recommendation

Comprehensive metrics suite for research publication.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple


class RecommendationMetrics:
    """
    Compute ranking metrics for Top-K recommendation tasks.
    
    Metrics include:
    - Precision@K
    - Recall@K
    - F1@K
    - NDCG@K
    - MRR (Mean Reciprocal Rank)
    - Hit Rate@K
    - MAP (Mean Average Precision)
    """
    
    def __init__(self, k_list: List[int] = [1, 3, 5, 10, 20]):
        """
        Args:
            k_list: List of K values to compute metrics for
        """
        self.k_list = k_list
    
    def compute_all_metrics(
        self,
        pred_scores: torch.Tensor,
        positive_indices: List[List[int]],
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            pred_scores: Predicted scores [num_items, num_candidates]
            positive_indices: List of positive candidate indices for each item
            prefix: Prefix for metric names (e.g., "train_" or "test_")
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        for k in self.k_list:
            # Precision, Recall, F1
            prec, rec, f1 = self._compute_precision_recall_f1(
                pred_scores, positive_indices, k
            )
            metrics[f'{prefix}precision@{k}'] = prec
            metrics[f'{prefix}recall@{k}'] = rec
            metrics[f'{prefix}f1@{k}'] = f1
            
            # NDCG
            ndcg = self._compute_ndcg(pred_scores, positive_indices, k)
            metrics[f'{prefix}ndcg@{k}'] = ndcg
            
            # Hit Rate
            hit_rate = self._compute_hit_rate(pred_scores, positive_indices, k)
            metrics[f'{prefix}hit_rate@{k}'] = hit_rate
        
        # MRR (doesn't depend on K)
        mrr = self._compute_mrr(pred_scores, positive_indices)
        metrics[f'{prefix}mrr'] = mrr
        
        # MAP
        map_score = self._compute_map(pred_scores, positive_indices)
        metrics[f'{prefix}map'] = map_score
        
        return metrics
    
    def _compute_precision_recall_f1(
        self,
        pred_scores: torch.Tensor,
        positive_indices: List[List[int]],
        k: int
    ) -> Tuple[float, float, float]:
        """Compute Precision@K, Recall@K, and F1@K"""
        num_items = len(positive_indices)
        precision_sum = 0.0
        recall_sum = 0.0
        
        for item_idx in range(num_items):
            scores = pred_scores[item_idx]
            pos_items = positive_indices[item_idx]
            
            if len(pos_items) == 0:
                continue
            
            # Get top-K predictions
            top_k_items = torch.topk(scores, k=min(k, len(scores)))[1].cpu().numpy()
            
            # Calculate hits
            hits = len(set(top_k_items) & set(pos_items))
            
            # Precision@K
            precision = hits / k
            precision_sum += precision
            
            # Recall@K
            recall = hits / len(pos_items)
            recall_sum += recall
        
        avg_precision = precision_sum / num_items
        avg_recall = recall_sum / num_items
        
        # F1@K
        if avg_precision + avg_recall > 0:
            f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        else:
            f1 = 0.0
        
        return avg_precision, avg_recall, f1
    
    def _compute_ndcg(
        self,
        pred_scores: torch.Tensor,
        positive_indices: List[List[int]],
        k: int
    ) -> float:
        """Compute NDCG@K (Normalized Discounted Cumulative Gain)"""
        num_items = len(positive_indices)
        ndcg_sum = 0.0
        
        for item_idx in range(num_items):
            scores = pred_scores[item_idx]
            pos_items = positive_indices[item_idx]
            
            if len(pos_items) == 0:
                continue
            
            # Get top-K predictions
            top_k_items = torch.topk(scores, k=min(k, len(scores)))[1].cpu().numpy()
            
            # DCG
            dcg = 0.0
            for i, item in enumerate(top_k_items):
                if item in pos_items:
                    dcg += 1.0 / np.log2(i + 2)
            
            # IDCG
            idcg = 0.0
            for i in range(min(k, len(pos_items))):
                idcg += 1.0 / np.log2(i + 2)
            
            # NDCG
            if idcg > 0:
                ndcg_sum += dcg / idcg
        
        return ndcg_sum / num_items
    
    def _compute_hit_rate(
        self,
        pred_scores: torch.Tensor,
        positive_indices: List[List[int]],
        k: int
    ) -> float:
        """Compute Hit Rate@K (fraction of items with at least one hit)"""
        num_items = len(positive_indices)
        hits = 0
        
        for item_idx in range(num_items):
            scores = pred_scores[item_idx]
            pos_items = positive_indices[item_idx]
            
            if len(pos_items) == 0:
                continue
            
            # Get top-K predictions
            top_k_items = torch.topk(scores, k=min(k, len(scores)))[1].cpu().numpy()
            
            # Check if there's at least one hit
            if len(set(top_k_items) & set(pos_items)) > 0:
                hits += 1
        
        return hits / num_items
    
    def _compute_mrr(
        self,
        pred_scores: torch.Tensor,
        positive_indices: List[List[int]]
    ) -> float:
        """Compute MRR (Mean Reciprocal Rank)"""
        num_items = len(positive_indices)
        rr_sum = 0.0
        
        for item_idx in range(num_items):
            scores = pred_scores[item_idx]
            pos_items = positive_indices[item_idx]
            
            if len(pos_items) == 0:
                continue
            
            # Get all predictions sorted
            sorted_items = torch.argsort(scores, descending=True).cpu().numpy()
            
            # Find rank of first positive item
            for rank, item in enumerate(sorted_items):
                if item in pos_items:
                    rr_sum += 1.0 / (rank + 1)
                    break
        
        return rr_sum / num_items
    
    def _compute_map(
        self,
        pred_scores: torch.Tensor,
        positive_indices: List[List[int]]
    ) -> float:
        """Compute MAP (Mean Average Precision)"""
        num_items = len(positive_indices)
        ap_sum = 0.0
        
        for item_idx in range(num_items):
            scores = pred_scores[item_idx]
            pos_items = set(positive_indices[item_idx])
            
            if len(pos_items) == 0:
                continue
            
            # Get all predictions sorted
            sorted_items = torch.argsort(scores, descending=True).cpu().numpy()
            
            # Calculate Average Precision
            hits = 0
            precision_sum = 0.0
            for rank, item in enumerate(sorted_items):
                if item in pos_items:
                    hits += 1
                    precision_sum += hits / (rank + 1)
            
            if hits > 0:
                ap_sum += precision_sum / len(pos_items)
        
        return ap_sum / num_items


def compute_diversity_metrics(
    pred_placements: torch.Tensor,
    num_servers: int
) -> Dict[str, float]:
    """
    Compute diversity metrics for server load distribution.
    
    Args:
        pred_placements: Binary placement matrix [num_models, num_servers]
        num_servers: Total number of servers
    
    Returns:
        Dictionary of diversity metrics
    """
    metrics = {}
    
    # Server load distribution
    server_loads = pred_placements.sum(dim=0)
    
    # Load statistics
    metrics['load_mean'] = server_loads.mean().item()
    metrics['load_std'] = server_loads.std().item()
    metrics['load_min'] = server_loads.min().item()
    metrics['load_max'] = server_loads.max().item()
    
    # Load balance score (1 = perfectly balanced, 0 = highly imbalanced)
    ideal_load = pred_placements.sum().item() / num_servers
    if ideal_load > 0:
        metrics['load_balance'] = 1.0 / (1.0 + server_loads.std().item() / ideal_load)
    else:
        metrics['load_balance'] = 1.0
    
    # Coverage (fraction of servers used)
    metrics['server_coverage'] = (server_loads > 0).float().mean().item()
    
    # Gini coefficient (inequality measure)
    sorted_loads = torch.sort(server_loads)[0].cpu().numpy()
    n = len(sorted_loads)
    cumsum = np.cumsum(sorted_loads)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_loads)) / (n * cumsum[-1]) - (n + 1) / n
    metrics['gini_coefficient'] = gini
    
    return metrics

