from graphreformer.datasests.Lung.load_data import produce_graphs_from_raw_format, Vocab, DFSCodeDataset
import networkx as nx
import matplotlib.pyplot as plt
from random import sample
from graphreformer.stats import generate_sample_plots
from graphgen.dfscode.dfs_wrapper import get_min_dfscode
from torch.utils.data import random_split
import torch
from graphreformer.model.graphreformer import DFSGraphReformer, DFSGraphTransformer
from graphreformer.model.train import train_model, generate_dfs_code
import sys
import pandas as pd
import time
from graphreformer.datasests.Lung.load_data import produce_random_walk_sampled_graphs
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append('graphgen')

import metrics.stats as metrics

def process_graph(graph):
    try:
        dfscode = get_min_dfscode(graph)
        dfscode.append(("<END>", "<END>", "<END>", "<END>", "<END>"))
        return dfscode
    except Exception:
        return None

def run_reformer():

    total_df = pd.DataFrame()

    graph_names = ['citeseer_long','citeseer','lung','yeast']

    file_dict = {
        'lung': 'graphreformer/datasests/Lung/lung.txt',
        'yeast': 'graphreformer/datasests/Lung/yeast.txt',
    }

    for graph_name in graph_names:
        print(f"Processing graph: {graph_name}")

        if graph_name == 'citeseer_long':
            graphs = produce_random_walk_sampled_graphs(
                'graphreformer/datasests/citeseer/citeseer.content',
                'graphreformer/datasests/citeseer/citeseer.cites',
                iterations=600,
                num_factor=4,
                min_num_edges=20,
            )
        elif graph_name == 'citeseer':
            graphs = produce_random_walk_sampled_graphs(
                'graphreformer/datasests/citeseer/citeseer.content',
                'graphreformer/datasests/citeseer/citeseer.cites',
                min_num_edges=20,
                iterations=150,
                num_factor=10,
            )
        elif graph_name == 'lung' or graph_name == 'yeast':
            graphs = produce_graphs_from_raw_format(
                file_dict[graph_name]
            )

        print(f"Loaded {len(graphs)}")

        plt.figure(figsize=(10, 10))
        for graph_idx in range(9):
            plt.subplot(3, 3, graph_idx + 1)
            plt.title(f'Graph {graph_idx + 1}')
            nx.draw(graphs[graph_idx], with_labels=False, node_size=50, font_size=8)
        plt.savefig(f'{graph_name}_graphs.png')

        fig = generate_sample_plots(
            graph_samples_1 = sample(graphs, 100),
            graph_samples_2 = sample(graphs, 100),
        )
        fig.savefig(f'{graph_name}_sample_plots.png')

        start = time.time()
        dfscodes = []
        dfs_code_indices = []
        max_length = 0
        # Generate DFSCodes for all the graphs in the dataset
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_graph, graph) for graph in graphs]

            for i, future in enumerate(as_completed(futures), 1):
                print(f"Processing graph {i}/{len(graphs)}")
                dfscode = future.result()
                if dfscode is not None:
                    dfscodes.append(dfscode)
                    dfs_code_indices.append(i-1)
                    if len(dfscode) > max_length:
                        max_length = len(dfscode)

        end = time.time()
        print(f"Time taken to process all graphs: {end - start:.2f} seconds")
        print(f"Max DFSCodes length: {max_length}")

        # Create vocabularies
        node_label_vocab = Vocab()
        edge_label_vocab = Vocab()
        node_id_vocab = Vocab()

        # Add special tokens
        for vocab in [node_label_vocab, edge_label_vocab, node_id_vocab]:
            vocab.add("<PAD>")
            vocab.add("<UNK>")
            vocab.add("<START>")
            vocab.add("<END>")

        
        def pad_to_multiple(length, multiple):
            return (length + multiple - 1) // multiple * multiple   

        max_length = pad_to_multiple(max_length, 8)  # Ensure length is a multiple of 5``

        # Step 1: Build full vocab from all sequences
        full_dataset = DFSCodeDataset(
            dfscodes, 
            node_id_vocab=node_id_vocab,
            node_label_vocab=node_label_vocab,
            edge_label_vocab=edge_label_vocab,
            max_len=max_length,
        )

        # Step 2: Split indices
        val_ratio = 0.2
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size
        train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

        graph_train_indices = [dfs_code_indices[i] for i in train_indices]

        # Step 3: Use Subset to share vocab but split data
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset   = Subset(full_dataset, val_indices)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)


        for batch in train_dataloader:
            print(batch['inputs']['v1'].shape, batch['targets']['v1'].shape)

        for batch in val_dataloader:
            print(batch['inputs']['v1'].shape, batch['targets']['v1'].shape)

        model = DFSGraphTransformer(
            num_node_ids=len(node_id_vocab),
            num_node_labels=len(node_label_vocab),
            num_edge_labels=len(edge_label_vocab),
            d_model=256,
            pad_token_id=node_id_vocab.get_id("<PAD>"),
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        start = time.time()
        train_model(model, train_dataloader, val_dataloader, optimizer, device=device, pad_token_id=node_id_vocab.get_id("<PAD>"), model_name=f"best_model_{graph_name}.pt")
        end = time.time()

        model.load_state_dict(torch.load(f"best_model_{graph_name}.pt", map_location="cuda" or "cpu"))
        model.eval()  # Set to eval mode

        sampled_graphs = []
        for i in range(1000):
            sampled_graph = generate_dfs_code(
                model,
                node_id_vocab,
                node_label_vocab,
                edge_label_vocab,
                max_len=max_length,
                temperature=0.85,
                device=device
            )
            sampled_graphs.append(sampled_graph)

        plt.figure(figsize=(10, 10))
        for graph_idx in range(9):
            plt.subplot(3, 3, graph_idx + 1)
            plt.title(f'Sampled Graph {graph_idx + 1}')
            nx.draw(sampled_graphs[graph_idx], with_labels=False, node_size=50, font_size=8)
        plt.savefig(f'generated_graphs_{graph_name}.png')

        graph_samples_1 = sample(graphs, len(sampled_graphs))

        fig = generate_sample_plots(
            graph_samples_1 = graph_samples_1,
            graph_samples_2 = sampled_graphs,
            label_1='Original Graphs',
            label_2='Predicted Graphs',
        )

        plt.savefig(f'generated_graphs_dist_{graph_name}.png')

        pred_indexes = list(range(len(sampled_graphs)))

        novelty = metrics.novelty_from_list(
            graphs, dfs_code_indices, sampled_graphs, pred_indexes, temp_path='temp', timeout=60
        )
        uniqueness = metrics.uniqueness_from_list(
            sampled_graphs, pred_indexes,  temp_path='temp', timeout=120
        )
        degree = metrics.degree_stats(graph_samples_1, sampled_graphs)
        clustering = metrics.clustering_stats(graph_samples_1, sampled_graphs)
        orbit = metrics.orbit_stats_all(graph_samples_1, sampled_graphs)
        NSPDK = metrics.nspdk_stats(graph_samples_1, sampled_graphs)
        node_label = metrics.node_label_stats(graph_samples_1, sampled_graphs)
        edge_label = metrics.edge_label_stats(graph_samples_1, sampled_graphs)

        this_df = pd.DataFrame(
            {
                'graph_source' : [graph_name],
                'degree' : [degree],
                'clustering' : [clustering],
                'orbit' : [orbit],
                'NSPDK' : [NSPDK],
                'node_label' : [node_label],
                'edge_label' : [edge_label],
                'time' : [end-start],
                'novelty' : [novelty],
                'uniqueness' : [uniqueness],
            }
        )

        if total_df.empty:
            total_df = this_df.copy()
        else:
            total_df = pd.concat(
                [total_df,this_df], axis=0, ignore_index=True
            ).reset_index(drop=True)

        reformer_graph_name = f"reformer_{graph_name}"

        model = DFSGraphReformer(
            num_node_ids=len(node_id_vocab),
            num_node_labels=len(node_label_vocab),
            num_edge_labels=len(edge_label_vocab),
            d_model=256,
            bucket_size=int(max_length / 2),
            pad_token_id=node_id_vocab.get_id("<PAD>"),
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        start = time.time()
        train_model(model, train_dataloader, val_dataloader, optimizer, device=device, pad_token_id=node_id_vocab.get_id("<PAD>"), model_name=f"best_model_{reformer_graph_name}.pt")
        end = time.time()

        model.load_state_dict(torch.load(f"best_model_{reformer_graph_name}.pt", map_location="cuda" or "cpu"))
        model.eval()  # Set to eval mode

        sampled_graphs = []
        for i in range(1000):
            sampled_graph = generate_dfs_code(
                model,
                node_id_vocab,
                node_label_vocab,
                edge_label_vocab,
                max_len=max_length,
                temperature=0.85,
                device=device
            )
            sampled_graphs.append(sampled_graph)

        plt.figure(figsize=(10, 10))
        for graph_idx in range(9):
            plt.subplot(3, 3, graph_idx + 1)
            plt.title(f'Sampled Graph {graph_idx + 1}')
            nx.draw(sampled_graphs[graph_idx], with_labels=False, node_size=50, font_size=8)
        plt.savefig(f'generated_graphs_{reformer_graph_name}.png')

        graph_samples_1 = sample(graphs, len(sampled_graphs))

        fig = generate_sample_plots(
            graph_samples_1 = graph_samples_1,
            graph_samples_2 = sampled_graphs,
            label_1='Original Graphs',
            label_2='Predicted Graphs',
        )

        plt.savefig(f'generated_graphs_dist_{reformer_graph_name}.png')

        pred_indexes = list(range(len(sampled_graphs)))

        novelty = metrics.novelty_from_list(
            graphs, dfs_code_indices, sampled_graphs, pred_indexes, temp_path='temp', timeout=60
        )
        uniqueness = metrics.uniqueness_from_list(
            sampled_graphs, pred_indexes,  temp_path='temp', timeout=120
        )
        degree = metrics.degree_stats(graph_samples_1, sampled_graphs)
        clustering = metrics.clustering_stats(graph_samples_1, sampled_graphs)
        orbit = metrics.orbit_stats_all(graph_samples_1, sampled_graphs)
        NSPDK = metrics.nspdk_stats(graph_samples_1, sampled_graphs)
        node_label = metrics.node_label_stats(graph_samples_1, sampled_graphs)
        edge_label = metrics.edge_label_stats(graph_samples_1, sampled_graphs)

        this_df = pd.DataFrame(
            {
                'graph_source' : [reformer_graph_name],
                'degree' : [degree],
                'clustering' : [clustering],
                'orbit' : [orbit],
                'NSPDK' : [NSPDK],
                'node_label' : [node_label],
                'edge_label' : [edge_label],
                'time' : [end-start],
                'novelty' : [novelty],
                'uniqueness' : [uniqueness],
            }
        )

        if total_df.empty:
            total_df = this_df.copy()
        else:
            total_df = pd.concat(
                [total_df,this_df], axis=0, ignore_index=True
            ).reset_index(drop=True)

    total_df.to_csv('results.csv')

if __name__ == "__main__":
    run_reformer()