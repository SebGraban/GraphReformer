import numpy as np
import tensorflow as tf
from collections import defaultdict

def precompute_global_features(dfs_codes):
    """Compute global dataset features: max nodes, vocabularies, and feature length"""
    all_node_labels = set()
    all_edge_labels = set()
    max_discovery = 0
    
    for code in dfs_codes:
        for edge in code:
            t_i, t_j, v_i, e, v_j = edge
            t_i = int(t_i)
            t_j = int(t_j)
            all_node_labels.update([v_i, v_j])
            all_edge_labels.add(e)
            max_discovery = max(max_discovery, t_i, t_j)
    
    max_num_nodes = max_discovery + 1  # Convert max discovery time to node count
    node_vocab = sorted(all_node_labels)
    edge_vocab = sorted(all_edge_labels)
    node_to_index = {label: idx for idx, label in enumerate(node_vocab)}
    edge_to_index = {label: idx for idx, label in enumerate(edge_vocab)}

    edge_to_index['<END>'] = len(edge_to_index)  # Add end token for edges
    node_to_index['<END>'] = len(node_to_index)  # Add end token for nodes
    
    # Calculate feature vector length (GraphGen-style)
    feature_len = (
        2 * (max_num_nodes+1) +       # t_i + t_j one-hot dimensions
        2 * (len(node_vocab) + 1) +     # v_i + v_j one-hot dimensions
        len(edge_vocab) + 1           # edge label one-hot dimension
    )
    
    return {
        'max_num_nodes': max_num_nodes,
        'node_vocab': node_vocab,
        'edge_vocab': edge_vocab,
        'node_to_index': node_to_index,
        'edge_to_index': edge_to_index,
        'feature_len': feature_len
    }

def edge_to_vector(edge, features):
    """Convert a DFS edge tuple to a concatenated one-hot vector"""
    t_i, t_j, v_i, e, v_j = edge
    t_i = int(t_i)
    t_j = int(t_j)
    node_idx = features['node_to_index']
    edge_idx = features['edge_to_index']
    
    # One-hot encoding for each component
    vec_t_i = tf.one_hot(t_i, features['max_num_nodes'] + 1).numpy()
    vec_t_j = tf.one_hot(t_j, features['max_num_nodes'] + 1).numpy()
    vec_v_i = tf.one_hot(node_idx[v_i], len(features['node_vocab']) + 1).numpy()
    vec_v_j = tf.one_hot(node_idx[v_j], len(features['node_vocab']) + 1).numpy()
    vec_e = tf.one_hot(edge_idx[e], len(features['edge_vocab']) + 1).numpy()
    
    return np.concatenate([vec_t_i, vec_t_j, vec_v_i, vec_e, vec_v_j])

def dfs_code_to_sequences(dfs_code, features):
    """Convert DFS code to input/target sequences with zero initial vector"""
    n = len(dfs_code)
    feature_len = features['feature_len']
    
    # Initialize with zeros
    input_seq = np.zeros((n+1, feature_len), dtype=np.float32)
    target_seq = np.zeros((n+1, feature_len), dtype=np.float32)
    
    # Build sequences
    for i in range(n):
        input_seq[i + 1] = edge_to_vector(dfs_code[i], features)
        target_seq[i] = edge_to_vector(dfs_code[i], features)
    
    # Add end token to the final target position
    target_seq[n] = edge_to_vector(
        [features['max_num_nodes'], features['max_num_nodes'], 
        '<END>', '<END>', '<END>'], 
        features
    )
    
    return input_seq, target_seq, n + 1

def create_tf_dataset(dfs_codes, batch_size=32):
    """Create TensorFlow dataset from list of DFS codes"""
    # Precompute global features
    features = precompute_global_features(dfs_codes)
    feature_len = features['feature_len']
    
    # Generate sequences
    inputs, targets, lengths = [], [], []
    for code in dfs_codes:
        i, t, l = dfs_code_to_sequences(code, features)
        inputs.append(i)
        targets.append(t)
        lengths.append(l)
    
    # Find maximum sequence length
    max_seq_len = max(lengths) if lengths else 0
    
    # Pad all sequences to max length
    padded_inputs = [
        np.vstack([seq, np.zeros((max_seq_len - len(seq), feature_len))]) 
                     for seq in inputs
    ]
    padded_targets = [np.vstack([seq, np.zeros((max_seq_len - len(seq), feature_len))]) 
                      for seq in targets]
    
    # Convert to TensorFlow tensors
    input_tensor = tf.convert_to_tensor(padded_inputs, dtype=tf.float32)
    target_tensor = tf.convert_to_tensor(padded_targets, dtype=tf.float32)
    lengths_tensor = tf.convert_to_tensor(lengths, dtype=tf.int32)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor, lengths_tensor))
    dataset = dataset.batch(batch_size)

    features['max_seq_len'] = max_seq_len
    
    return dataset, features

