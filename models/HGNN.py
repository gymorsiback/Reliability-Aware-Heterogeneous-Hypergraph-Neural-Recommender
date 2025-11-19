import torch
from torch import nn
from models import HGNN_conv
import torch.nn.functional as F


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


class HGNN_ModelPlacement(nn.Module):
    """
    HGNN for model placement prediction task
    Predicts which servers each model should be placed on
    """
    def __init__(self, in_ch, n_hid, num_users, num_models, num_servers, dropout=0.5):
        super(HGNN_ModelPlacement, self).__init__()
        self.num_users = num_users
        self.num_models = num_models
        self.num_servers = num_servers
        self.dropout = dropout
        
        # Hypergraph convolution layers (3 layers - GPU memory constrained)
        # Dense graph operations prevent using 4+ layers on 8GB GPU
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.hgc3 = HGNN_conv(n_hid, n_hid)
        
        # Placement prediction head
        # Use bilinear layer to predict model-server compatibility
        self.placement_predictor = nn.Bilinear(n_hid, n_hid, 1)
        
        # Alternative: MLP predictor
        self.mlp_predictor = nn.Sequential(
            nn.Linear(n_hid * 2, n_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid, n_hid // 2),
            nn.ReLU(),
            nn.Linear(n_hid // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, G):
        """
        Args:
            x: Node features [num_total_nodes, in_ch]
            G: Graph Laplacian [num_total_nodes, num_total_nodes]
            
        Returns:
            placement_scores: [num_models, num_servers]
        """
        # Apply hypergraph convolutions (3 layers with dropout)
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.hgc2(x, G))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc3(x, G)
        
        # Extract embeddings for different node types
        # Node order: [users, models, servers]
        model_start = self.num_users
        model_end = self.num_users + self.num_models
        server_start = model_end
        server_end = server_start + self.num_servers
        
        model_embeddings = x[model_start:model_end]  # [num_models, n_hid]
        server_embeddings = x[server_start:server_end]  # [num_servers, n_hid]
        
        # Predict placement scores for all model-server pairs
        placement_scores = self.predict_placements(model_embeddings, server_embeddings)
        
        return placement_scores
    
    def predict_placements(self, model_embeddings, server_embeddings):
        """
        Predict placement scores for model-server pairs
        
        Args:
            model_embeddings: [num_models, n_hid]
            server_embeddings: [num_servers, n_hid]
            
        Returns:
            scores: [num_models, num_servers]
        """
        num_models = model_embeddings.size(0)
        num_servers = server_embeddings.size(0)
        
        # Method 1: Bilinear
        # Expand dimensions for broadcasting
        model_expanded = model_embeddings.unsqueeze(1).expand(-1, num_servers, -1)
        server_expanded = server_embeddings.unsqueeze(0).expand(num_models, -1, -1)
        
        # Compute scores using bilinear layer
        scores = self.placement_predictor(
            model_expanded.reshape(-1, model_embeddings.size(1)),
            server_expanded.reshape(-1, server_embeddings.size(1))
        ).reshape(num_models, num_servers)
        
        # Remove sigmoid to avoid gradient vanishing
        # BPR loss works better with raw scores
        return scores
    
    def predict_placements_mlp(self, model_embeddings, server_embeddings):
        """
        Alternative: Use MLP for prediction
        """
        num_models = model_embeddings.size(0)
        num_servers = server_embeddings.size(0)
        
        # Concatenate model and server embeddings
        model_expanded = model_embeddings.unsqueeze(1).expand(-1, num_servers, -1)
        server_expanded = server_embeddings.unsqueeze(0).expand(num_models, -1, -1)
        
        combined = torch.cat([model_expanded, server_expanded], dim=2)
        scores = self.mlp_predictor(combined.reshape(-1, model_embeddings.size(1) * 2))
        
        return scores.reshape(num_models, num_servers)
    
    def get_embeddings(self, x, G):
        """
        Get node embeddings after hypergraph convolutions
        """
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.hgc2(x, G))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc3(x, G)
        return x
