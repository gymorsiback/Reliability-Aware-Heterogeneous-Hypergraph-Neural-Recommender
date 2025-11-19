"""
Experiment Logger for Research Publication

Logs all training and inference results to CSV files for analysis and visualization.
"""

import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import torch


class ExperimentLogger:
    """
    Logger for research experiments.
    
    Saves:
    - Training metrics per epoch (CSV)
    - Configuration (JSON)
    - Model checkpoints
    - Inference results (CSV)
    - System information
    """
    
    def __init__(
        self,
        exp_name: str,
        config: Dict[str, Any],
        base_dir: str = "results"
    ):
        """
        Args:
            exp_name: Experiment name (e.g., "topk_k5_h128")
            config: Configuration dictionary
            base_dir: Base directory for results
        """
        self.exp_name = exp_name
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.exp_dir = Path(base_dir) / f"{exp_name}_{self.timestamp}"
        self.train_dir = self.exp_dir / "train"
        self.inference_dir = self.exp_dir / "inference"
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        
        for dir_path in [self.train_dir, self.inference_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV files
        self.train_csv_path = self.train_dir / "train_metrics.csv"
        self.train_csv_initialized = False
        
        # Save configuration
        self._save_config()
        
        # Track experiment start time
        self.start_time = time.time()
        
        # Store training history
        self.train_history = []
    
    def _save_config(self):
        """Save experiment configuration"""
        config_path = self.exp_dir / "config.json"
        
        config_to_save = self.config.copy()
        config_to_save['exp_name'] = self.exp_name
        config_to_save['timestamp'] = self.timestamp
        config_to_save['device'] = str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU'
        
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        print(f"Configuration saved to: {config_path}")
    
    def log_train_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
        learning_rate: float = None
    ):
        """
        Log training metrics for one epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
            learning_rate: Current learning rate
        """
        # Add epoch and timestamp
        log_entry = {
            'epoch': epoch,
            'timestamp': time.time() - self.start_time,
            **metrics
        }
        
        if learning_rate is not None:
            log_entry['learning_rate'] = learning_rate
        
        # Initialize CSV if first epoch
        if not self.train_csv_initialized:
            with open(self.train_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writeheader()
            self.train_csv_initialized = True
        
        # Append to CSV
        with open(self.train_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writerow(log_entry)
        
        # Store in memory
        self.train_history.append(log_entry)
    
    def log_inference(
        self,
        split: str,
        metrics: Dict[str, float],
        model_info: Dict[str, Any] = None
    ):
        """
        Log inference results.
        
        Args:
            split: Data split name (e.g., 'test', 'val')
            metrics: Dictionary of metrics
            model_info: Additional model information
        """
        csv_path = self.inference_dir / f"inference_{split}.csv"
        
        log_entry = {
            'split': split,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **metrics
        }
        
        if model_info:
            log_entry.update(model_info)
        
        # Write to CSV
        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)
        
        print(f"Inference results saved to: {csv_path}")
    
    def log_detailed_predictions(
        self,
        split: str,
        model_ids: List[int],
        true_servers: List[List[int]],
        pred_servers: List[List[int]],
        pred_scores: List[List[float]]
    ):
        """
        Log detailed per-model predictions.
        
        Args:
            split: Data split name
            model_ids: List of model IDs
            true_servers: List of ground truth server lists
            pred_servers: List of predicted server lists
            pred_scores: List of prediction scores
        """
        csv_path = self.inference_dir / f"predictions_{split}.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'model_id',
                'true_servers',
                'pred_servers',
                'pred_scores',
                'num_correct',
                'precision',
                'recall'
            ])
            
            for i in range(len(model_ids)):
                true_set = set(true_servers[i])
                pred_set = set(pred_servers[i])
                num_correct = len(true_set & pred_set)
                
                precision = num_correct / len(pred_set) if len(pred_set) > 0 else 0
                recall = num_correct / len(true_set) if len(true_set) > 0 else 0
                
                writer.writerow([
                    model_ids[i],
                    ','.join(map(str, true_servers[i])),
                    ','.join(map(str, pred_servers[i])),
                    ','.join(map(str, pred_scores[i])),
                    num_correct,
                    f"{precision:.4f}",
                    f"{recall:.4f}"
                ])
        
        print(f"Detailed predictions saved to: {csv_path}")
    
    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Best model saved to: {best_path}")
    
    def save_final_summary(
        self,
        best_epoch: int,
        best_metrics: Dict[str, float],
        total_time: float
    ):
        """
        Save final experiment summary.
        
        Args:
            best_epoch: Best epoch number
            best_metrics: Best metrics achieved
            total_time: Total training time in seconds
        """
        summary = {
            'exp_name': self.exp_name,
            'timestamp': self.timestamp,
            'config': self.config,
            'best_epoch': best_epoch,
            'best_metrics': best_metrics,
            'total_epochs': len(self.train_history),
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'avg_time_per_epoch': total_time / len(self.train_history) if self.train_history else 0
        }
        
        summary_path = self.exp_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nExperiment summary saved to: {summary_path}")
        print(f"All results saved in: {self.exp_dir}")
    
    def print_summary_table(self, best_metrics: Dict[str, float]):
        """Print formatted summary table"""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Experiment: {self.exp_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"\nConfiguration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print(f"\nBest Metrics:")
        for key, value in best_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("=" * 80)


class BaselineLogger:
    """Logger for baseline methods (Random, Popular, etc.)"""
    
    def __init__(self, method_name: str, base_dir: str = "results/baselines"):
        self.method_name = method_name
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_results(
        self,
        split: str,
        metrics: Dict[str, float],
        k_list: List[int]
    ):
        """Log baseline results"""
        csv_path = self.base_dir / f"{self.method_name}_{split}.csv"
        
        log_entry = {
            'method': self.method_name,
            'split': split,
            'timestamp': self.timestamp,
            **metrics
        }
        
        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)
        
        print(f"Baseline results saved to: {csv_path}")

