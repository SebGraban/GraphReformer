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
from graph_rnn.data import Graph_Adj_Matrix, mapping, calc_max_prev_node
from graph_rnn.model import create_model_rnn, predict_graphs
from graphgen.model import create_model as create_graphgen_model
from graphgen.data import Graph_DFS_code
from graphgen.model import predict_graphgen_graphs  
from GransFormer.data import MyGraph_sequence_sampler_pytorch
from GransFormer.args import BaseArgs
from GransFormer.Transformer.Models import Transformer
from GransFormer.Transformer.Optim import MyScheduledOptim
from GransFormer.predict import generate_graph_exact
import os

sys.path.append('/teamspace/studios/this_studio/GraphReformer/graphgen')

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

    graph_names = ['lung']

    file_dict = {
        'lung': '/teamspace/studios/this_studio/GraphReformer/graphreformer/datasests/Lung/lung.txt',
        'yeast': '/teamspace/studios/this_studio/GraphReformer/graphreformer/datasests/Lung/yeast.txt',
    }

    models = [
        'GraphRNN',
        'graphgen',
        'Gransformer',
        'DFSGraphTransformer',
        'DFSGraphReformer',
    ]

    for graph_name in graph_names:
        print(f"Processing graph: {graph_name}")

        if graph_name == 'citeseer_long':
            graphs = produce_random_walk_sampled_graphs(
                '/teamspace/studios/this_studio/GraphReformer/graphreformer/datasests/citeseer/citeseer.content',
                '/teamspace/studios/this_studio/GraphReformer/graphreformer/datasests/citeseer/citeseer.cites',
                iterations=600,
                num_factor=4,
                min_num_edges=20,
            )
        elif graph_name == 'citeseer':
            graphs = produce_random_walk_sampled_graphs(
                '/teamspace/studios/this_studio/GraphReformer/graphreformer/datasests/citeseer/citeseer.content',
                '/teamspace/studios/this_studio/GraphReformer/graphreformer/datasests/citeseer/citeseer.cites',
                min_num_edges=20,
                iterations=150,
                num_factor=10,
            )
        elif graph_name == 'lung' or graph_name == 'yeast':
            graphs, _ = produce_graphs_from_raw_format(
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
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
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

        # Step 3: Use Subset to share vocab but split data
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset   = Subset(full_dataset, val_indices)

        train_dataloader_dfs = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_dataloader_dfs = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

        feature_map = mapping(graphs)

        max_prev_node = calc_max_prev_node(graphs)

        for model_type in models:
            print(f"Training model: {model_type}")

            if model_type == 'GraphRNN':
                train_dataset_rnn = Graph_Adj_Matrix(
                    graph_list=Subset(graphs, train_indices),
                    feature_map=feature_map,
                    max_prev_node=max_prev_node,
                    random_bfs=True,
                )
                val_dataset_rnn = Graph_Adj_Matrix(
                    graph_list=Subset(graphs, val_indices),
                    feature_map=feature_map,
                    max_prev_node=max_prev_node,
                    random_bfs=True,
                )
                train_dataloader = torch.utils.data.DataLoader(train_dataset_rnn, batch_size=256, shuffle=True)
                val_dataloader = torch.utils.data.DataLoader(val_dataset_rnn, batch_size=256, shuffle=False)
            elif model_type == 'graphgen':
                train_dataloader = torch.utils.data.DataLoader(
                    Graph_DFS_code(
                        dfscodes=[dfscodes[i][:-1] for i in train_indices],
                        feature_map=feature_map,
                    ),
                    batch_size=256,
                    shuffle=True,
                )
                val_dataloader = torch.utils.data.DataLoader(
                    Graph_DFS_code(
                        dfscodes=[dfscodes[i][:-1] for i in val_indices],
                        feature_map=feature_map,
                    ),
                    batch_size=256,
                    shuffle=False,
                )
            elif model_type == 'Gransformer':
                args = BaseArgs(
                    graph_type='protein'
                    ,note='gransformer-6layers-nhead8-gattk4-grposenck4-MADEhl1msk3natuord0dimred1'
                    ,batch_ratio=1.0
                )

                args.max_num_node = feature_map['max_nodes']
                args.min_num_node = min([graphs[i].number_of_nodes() for i in range(len(graphs))])

                dataset_G = MyGraph_sequence_sampler_pytorch(
                    G_list=graphs
                    ,args=args
                    ,max_num_node=feature_map['max_nodes']
                    ,max_prev_node=max_prev_node
                )
                train_dataloader = torch.utils.data.DataLoader(
                    Subset(dataset_G, train_indices),
                    batch_size=256,
                    shuffle=True,
                )
                val_dataloader = torch.utils.data.DataLoader(
                    Subset(dataset_G, val_indices),
                    batch_size=256,
                    shuffle=False,
                )
            elif model_type == 'DFSGraphReformer' or model_type == 'DFSGraphTransformer':
                train_dataloader = train_dataloader_dfs
                val_dataloader = val_dataloader_dfs

            device = "cuda" if torch.cuda.is_available() else "cpu"

            args = {
                'pad_token_id': node_id_vocab.get_id("<PAD>"),
                'feature_map': feature_map,
                'max_prev_node': max_prev_node,
                'max_head_and_tail': None,
            }

            if model_type == 'DFSGraphReformer':
                model = DFSGraphReformer(
                    num_node_ids=len(node_id_vocab),
                    num_node_labels=len(node_label_vocab),
                    num_edge_labels=len(edge_label_vocab),
                    d_model=256,
                    pad_token_id=node_id_vocab.get_id("<PAD>"),
                )
            elif model_type == 'DFSGraphTransformer':
                model = DFSGraphTransformer(
                    num_node_ids=len(node_id_vocab),
                    num_node_labels=len(node_label_vocab),
                    num_edge_labels=len(edge_label_vocab),
                    d_model=256,
                    pad_token_id=node_id_vocab.get_id("<PAD>"),
                )
            elif model_type == 'GraphRNN':
                model = create_model_rnn(
                    feature_map, max_prev_node=max_prev_node, device=device
                )
            elif model_type == 'graphgen':
                model = create_graphgen_model(feature_map, device=device)
            elif model_type == 'Gransformer':
                args = BaseArgs(
                    graph_type='protein'
                    ,note='gransformer-6layers-nhead8-gattk4-grposenck4-MADEhl1msk3natuord0dimred1'
                    ,batch_ratio=1.0
                )
                args.max_num_node = feature_map['max_nodes']
                if args.input_type == 'node_based':
                    args.max_seq_len = dataset_G.max_seq_len
                    args.vocab_size = feature_map['max_nodes'] + 3  # 0 for padding, self.n+1 for add_node, self.n+2 for termination
                elif args.input_type in ['preceding_neighbors_vector', 'max_prev_node_neighbors_vec']:
                    args.max_seq_len = dataset_G.max_seq_len
                    args.vocab_size = None
                else:
                    raise NotImplementedError
                model = Transformer(
                    args.vocab_size,
                    args.vocab_size,
                    src_pad_idx=args.src_pad_idx,
                    trg_pad_idx=args.trg_pad_idx,
                    args=args,
                    trg_emb_prj_weight_sharing=args.proj_share_weight,
                    emb_src_trg_weight_sharing=args.embs_share_weight,
                    d_k=args.d_k,
                    d_v=args.d_v,
                    d_model=args.d_model,
                    d_word_vec=args.d_word_vec,
                    d_inner=args.d_inner_hid,
                    n_layers=args.n_layers,
                    n_ensemble=args.n_ensemble,
                    n_head=args.n_head,
                    dropout=args.dropout,
                    scale_emb_or_prj=args.scale_emb_or_prj
                ).to(device)

            if model_type == 'DFSGraphReformer' or model_type == 'DFSGraphTransformer':
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
            elif model_type == 'GraphRNN' or model_type == 'graphgen':
                # initialize optimizer
                # Scale learning rate inversely with batch size (original lr=0.003 for batch_size=32)
                lr = 0.003 * (32 / 256)
                optimizer = {}
                for name, net in model.items():
                    optimizer['optimizer_' + name] = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                        weight_decay=5e-5)
            elif model_type == 'Gransformer':
                optimizer = MyScheduledOptim(
                    torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                    torch.optim.Adam(list(model.parameters())[model.encoder.num_shared_parameters:], betas=(0.9, 0.98), eps=1e-09),
                    args.milestones, args.lr_list, args.sep_optimizer_start_step
                )

            start = time.time()
            train_model(model_type, model, train_dataloader, val_dataloader, optimizer, device=device, args=args, model_name=f"best_model_{model_type}_{graph_name}.pt")
            end = time.time()

            if model_type == 'DFSGraphReformer' or model_type == 'DFSGraphTransformer':
                model.load_state_dict(torch.load(f"best_model_{model_type}_{graph_name}.pt", map_location="cuda" or "cpu"))
                model.eval()  # Set to eval mode
            elif model_type == 'GraphRNN' or model_type == 'graphgen':
                for name, net in model.items():
                    net.load_state_dict(torch.load(f"best_model_{model_type}_{graph_name}_{name}.pt", map_location="cuda" or "cpu"))
                    net.eval()  # Set to eval mode
            elif model_type == 'Gransformer':
                model.load_state_dict(torch.load(f"best_model_{model_type}_{graph_name}.pt", map_location="cuda" or "cpu"))
                model.eval()  # Set to eval mode

            sampled_graphs = []

            if model_type == 'DFSGraphReformer' or model_type == 'DFSGraphTransformer':
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
            elif model_type == 'GraphRNN':
                sampled_graphs = predict_graphs(
                    model,
                    feature_map,
                    max_prev_node=max_prev_node,
                    device=device,
                    max_head_and_tail=None,
                )
            elif model_type == 'graphgen':
                sampled_graphs = predict_graphgen_graphs(
                    model,
                    feature_map,
                    device=device,
                    loss_type='BCE',
                )
            elif model_type == 'Gransformer':
                sampled_graphs = []
                for i in range(1024 // 256):
                    sampled_batch_graphs = generate_graph_exact(model, args, device=device)
                    sampled_graphs.extend(sampled_batch_graphs)

            plt.figure(figsize=(10, 10))
            for graph_idx in range(9):
                plt.subplot(3, 3, graph_idx + 1)
                plt.title(f'Sampled Graph {graph_idx + 1}')
                nx.draw(sampled_graphs[graph_idx], with_labels=False, node_size=50, font_size=8)
            plt.savefig(f'generated_graphs_{model_type}_{graph_name}.png')

            graph_samples_1 = sample(graphs, len(sampled_graphs))

            fig = generate_sample_plots(
                graph_samples_1 = graph_samples_1,
                graph_samples_2 = sampled_graphs,
                label_1='Original Graphs',
                label_2='Predicted Graphs',
            )

            plt.savefig(f'generated_graphs_dist_{model_type}_{graph_name}.png')

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
                    'model_type' : [model_type],
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

            total_df.to_csv('results_new.csv')

if __name__ == "__main__":
    run_reformer()