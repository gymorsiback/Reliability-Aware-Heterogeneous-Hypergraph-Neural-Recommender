# --------------------------------------------------------
# Utility functions for Hypergraph
#
# Author: Yifan Feng
# Date: November 2018
# --------------------------------------------------------
import numpy as np


def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.asarray(x)
    aa = np.sum(np.multiply(x, x), 1, keepdims=True)
    ab = x @ x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False, use_gpu=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :param use_gpu: whether to use GPU for computation
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight, use_gpu)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight, use_gpu))
        return G


def _generate_G_from_H(H, variable_weight=False, use_gpu=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :param use_gpu: whether to use GPU for computation (requires PyTorch)
    :return: G
    """
    if use_gpu:
        # GPU-accelerated version using PyTorch
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to PyTorch tensor and move to GPU
        if type(H) == np.ndarray:
            H_tensor = torch.from_numpy(H.astype(np.float32)).to(device)
        else:
            H_tensor = torch.FloatTensor(H).to(device)
        
        n_edge = H_tensor.shape[1]
        W = torch.ones(n_edge, device=device)
        
        # Calculate degrees
        DV = torch.sum(H_tensor * W, dim=1)
        DE = torch.sum(H_tensor, dim=0)
        
        # Protection against division by zero
        DV[DV == 0] = 1
        DE[DE == 0] = 1
        
        # Create diagonal matrices efficiently on GPU
        invDE = torch.diag(torch.pow(DE, -1))
        DV2 = torch.diag(torch.pow(DV, -0.5))
        W_diag = torch.diag(W)
        HT = H_tensor.T
        
        if variable_weight:
            DV2_H = DV2 @ H_tensor
            invDE_HT_DV2 = invDE @ HT @ DV2
            # Convert back to numpy
            return DV2_H.cpu().numpy(), W_diag.cpu().numpy(), invDE_HT_DV2.cpu().numpy()
        else:
            # Compute on GPU
            print(f"Computing graph Laplacian on {device}...")
            G = DV2 @ H_tensor @ W_diag @ invDE @ HT @ DV2
            # Convert back to numpy
            return G.cpu().numpy()
    else:
        # CPU version using NumPy
        H = np.array(H, dtype=np.float64)
        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = np.ones(n_edge)
        # the degree of the node
        DV = np.sum(H * W, axis=1)
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)
        
        # Avoid division by zero
        DE[DE == 0] = 1
        DV[DV == 0] = 1

        invDE = np.diag(np.power(DE, -1))
        DV2 = np.diag(np.power(DV, -0.5))
        W = np.diag(W)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 @ H
            invDE_HT_DV2 = invDE @ HT @ DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 @ H @ W @ invDE @ HT @ DV2
            return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H


def construct_H_heterogeneous(num_users, num_models, num_servers,
                               user_model_interactions, model_server_mappings,
                               server_topology, use_knn=True, k_neig=10):
    """
    Construct heterogeneous hypergraph incidence matrix for model placement task
    
    Args:
        num_users: Number of user nodes
        num_models: Number of model nodes  
        num_servers: Number of server nodes
        user_model_interactions: DataFrame with UserID and model steps (step_1, ..., step_5)
        model_server_mappings: DataFrame with ModelID and ServerID
        server_topology: Adjacency matrix of server network [num_servers x num_servers]
        use_knn: Whether to add KNN-based hyperedges within same type nodes
        k_neig: K for KNN hyperedges
        
    Returns:
        H: Hypergraph incidence matrix [num_total_nodes x num_hyperedges]
        edge_types: Type of each hyperedge
        edge_info: Information about each hyperedge
    """
    num_total = num_users + num_models + num_servers
    hyperedges = []
    edge_types = []
    edge_info = []
    
    # Node index ranges:
    # Users: [0, num_users)
    # Models: [num_users, num_users + num_models)
    # Servers: [num_users + num_models, num_total)
    
    # Type 1: User-Model hyperedges
    print("  Constructing user-model hyperedges...")
    
    # Group by user to create hyperedges
    user_model_groups = {}
    for _, row in user_model_interactions.iterrows():
        user_id = int(row['UserID']) - 1  # Convert to 0-indexed
        model_id = int(row['ModelID']) - 1  # Convert to 0-indexed
        
        if user_id < 0 or user_id >= num_users:
            continue
        if model_id < 0 or model_id >= num_models:
            continue
        
        if user_id not in user_model_groups:
            user_model_groups[user_id] = set()
        user_model_groups[user_id].add(model_id)
    
    # Create hyperedges
    for user_id, model_ids in user_model_groups.items():
        if len(model_ids) > 0:
            user_idx = user_id
            model_indices = [num_users + mid for mid in model_ids]
            edge_nodes = [user_idx] + model_indices
            hyperedges.append(edge_nodes)
            edge_types.append('user_model')
            edge_info.append(f"User {user_id + 1} interactions")
    
    print(f"    Created {len([e for e in edge_types if e == 'user_model'])} user-model hyperedges")
    
    # Type 2: Model-Server hyperedges (from existing mappings)
    print("  Constructing model-server hyperedges...")
    model_server_dict = {}
    for _, row in model_server_mappings.iterrows():
        model_id = int(row['ModelID']) - 1
        server_id = int(row['ServerID']) - 1
        
        if 0 <= model_id < num_models and 0 <= server_id < num_servers:
            if model_id not in model_server_dict:
                model_server_dict[model_id] = []
            model_server_dict[model_id].append(server_id)
    
    for model_id, server_list in model_server_dict.items():
        model_idx = num_users + model_id
        server_indices = [num_users + num_models + sid for sid in server_list]
        
        # Create hyperedge: one model + all its servers
        edge_nodes = [model_idx] + server_indices
        hyperedges.append(edge_nodes)
        edge_types.append('model_server')
        edge_info.append(f"Model {model_id + 1} placement")
    
    print(f"    Created {len([e for e in edge_types if e == 'model_server'])} model-server hyperedges")
    
    # Type 3: Server-Server hyperedges (from topology)
    print("  Constructing server-server hyperedges...")
    for i in range(min(num_servers, server_topology.shape[0])):
        neighbors = []
        for j in range(min(num_servers, server_topology.shape[1])):
            if server_topology[i, j] > 0 and i != j:
                neighbors.append(j)
        
        if len(neighbors) > 0:
            server_idx = num_users + num_models + i
            neighbor_indices = [num_users + num_models + n for n in neighbors]
            
            # Create hyperedge: server + its neighbors
            edge_nodes = [server_idx] + neighbor_indices
            hyperedges.append(edge_nodes)
            edge_types.append('server_server')
            edge_info.append(f"Server {i + 1} network")
    
    print(f"    Created {len([e for e in edge_types if e == 'server_server'])} server-server hyperedges")
    
    # Convert hyperedges to incidence matrix
    print("  Building incidence matrix...")
    num_edges = len(hyperedges)
    H = np.zeros((num_total, num_edges), dtype=np.float32)
    
    for edge_idx, nodes in enumerate(hyperedges):
        for node_idx in nodes:
            if 0 <= node_idx < num_total:
                H[node_idx, edge_idx] = 1.0
    
    print(f"  Hypergraph shape: {H.shape}")
    print(f"  Total hyperedges: {num_edges}")
    print(f"  Avg nodes per hyperedge: {np.sum(H) / num_edges:.2f}")
    print(f"  Avg hyperedges per node: {np.sum(H) / num_total:.2f}")
    
    return H, edge_types, edge_info


def construct_H_for_model_placement(dataset, k_neig=10, use_gpu=False):
    """
    High-level function to construct hypergraph for model placement task
    
    Args:
        dataset: ModelPlacementDataset instance
        k_neig: K for KNN hyperedges
        use_gpu: Whether to use GPU for computing the graph Laplacian
        
    Returns:
        H: Hypergraph incidence matrix
        G: Graph Laplacian matrix
        edge_info: Edge information dict
    """
    print("\n" + "=" * 80)
    print("CONSTRUCTING HETEROGENEOUS HYPERGRAPH")
    print("=" * 80)
    
    H, edge_types, edge_info_list = construct_H_heterogeneous(
        num_users=dataset.num_users,
        num_models=dataset.num_models,
        num_servers=dataset.num_servers,
        user_model_interactions=dataset.user_model_df,
        model_server_mappings=dataset.model_server_df,
        server_topology=dataset.topology,
        use_knn=True,
        k_neig=k_neig
    )
    
    # Generate G from H
    print("\nGenerating graph Laplacian from hypergraph...")
    G = generate_G_from_H(H, variable_weight=False, use_gpu=use_gpu)
    
    edge_info = {
        'H': H,
        'G': G,
        'edge_types': edge_types,
        'edge_info_list': edge_info_list,
        'num_edges': len(edge_types),
        'edge_type_counts': {
            'user_model': edge_types.count('user_model'),
            'model_server': edge_types.count('model_server'),
            'server_server': edge_types.count('server_server')
        }
    }
    
    print(f"\nHypergraph construction completed!")
    print(f"  Incidence matrix H: {H.shape}")
    print(f"  Graph Laplacian G: {G.shape}")
    print(f"  Edge type distribution:")
    for edge_type, count in edge_info['edge_type_counts'].items():
        print(f"    {edge_type}: {count}")
    
    return H, G, edge_info
