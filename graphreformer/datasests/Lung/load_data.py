import networkx as nx
import torch
from torch.utils.data import Dataset
from functools import partial
from multiprocessing import Pool
import math
import random
import networkx as nx
from tqdm import tqdm



def random_walk_with_restart_sampling(
    G, start_node, iterations, fly_back_prob=0.15,
    max_nodes=None, max_edges=None
):
    sampled_graph = nx.Graph()
    sampled_graph.add_node(start_node, label=G.nodes[start_node]['label'])

    curr_node = start_node

    for _ in range(iterations):
        choice = torch.rand(()).item()

        if choice < fly_back_prob:
            curr_node = start_node
        else:
            neigh = list(G.neighbors(curr_node))
            chosen_node_id = torch.randint(
                0, len(neigh), ()).item()
            chosen_node = neigh[chosen_node_id]

            sampled_graph.add_node(
                chosen_node, label=G.nodes[chosen_node]['label'])
            sampled_graph.add_edge(
                curr_node, chosen_node, label=G.edges[curr_node, chosen_node]['label'])

            curr_node = chosen_node

        if max_nodes is not None and sampled_graph.number_of_nodes() >= max_nodes:
            break

        if max_edges is not None and sampled_graph.number_of_edges() >= max_edges:
            break

    # sampled_graph = G.subgraph(sampled_node_set)

    return sampled_graph

def produce_random_walk_sampled_graphs(
    content_path, cities_path, iterations, num_factor,
    num_graphs=None, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):
    print('Producing random_walk graphs - num_factor - {}'.format(num_factor))
    G = nx.Graph()
    d = {}

    # Build graph from .content
    with open(content_path, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split('\t')
            G.add_node(idx, label=parts[-1])
            d[parts[0]] = idx

    # Add edges from .cites
    missing = 0
    with open(cities_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0] in d and parts[1] in d:
                G.add_edge(d[parts[0]], d[parts[1]], label='DEFAULT_LABEL')
            else:
                missing += 1

    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G)

    # Sequential subgraph sampling
    all_subgraphs = []
    for node_index in tqdm(range(G.number_of_nodes())):
        subgraph_list = sample_subgraphs(
            node_index,
            G=G,
            iterations=iterations,
            num_factor=num_factor,
            min_num_nodes=min_num_nodes,
            max_num_nodes=max_num_nodes,
            min_num_edges=min_num_edges,
            max_num_edges=max_num_edges
        )
        all_subgraphs.extend(subgraph_list)


    # Shuffle and truncate if needed
    random.shuffle(all_subgraphs)
    if num_graphs is not None:
        all_subgraphs = all_subgraphs[:num_graphs]

    return all_subgraphs

def sample_subgraphs(
    idx, G, iterations, num_factor, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    subgraphs = []
    deg = G.degree[idx]
    for _ in range(num_factor * int(math.sqrt(deg))):
        G_rw = random_walk_with_restart_sampling(
            G, idx, iterations=iterations, max_nodes=max_num_nodes,
            max_edges=max_num_edges
        )
        G_rw = nx.convert_node_labels_to_integers(G_rw)
        G_rw.remove_edges_from(nx.selfloop_edges(G_rw))

        if not check_graph_size(
            G_rw, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
        ):
            continue

        if nx.is_connected(G_rw):
            subgraphs.append(G_rw)

    return subgraphs

def check_graph_size(
    graph, min_num_nodes=None, max_num_nodes=None,
    min_num_edges=None, max_num_edges=None
):

    if min_num_nodes and graph.number_of_nodes() < min_num_nodes:
        return False
    if max_num_nodes and graph.number_of_nodes() > max_num_nodes:
        return False

    if min_num_edges and graph.number_of_edges() < min_num_edges:
        return False
    if max_num_edges and graph.number_of_edges() > max_num_edges:
        return False

    return True

def produce_graphs_from_raw_format(
    inputfile, num_graphs=None, min_num_nodes=None,
    max_num_nodes=None, min_num_edges=None, max_num_edges=None
):
    """
    :param inputfile: Path to file containing graphs in raw format
    :param output_path: Path to store networkx graphs
    :param num_graphs: Upper bound on number of graphs to be taken
    :param min_num_nodes: Lower bound on number of nodes in graphs if provided
    :param max_num_nodes: Upper bound on number of nodes in graphs if provided
    :param min_num_edges: Lower bound on number of edges in graphs if provided
    :param max_num_edges: Upper bound on number of edges in graphs if provided
    :return: number of graphs produced
    """

    lines = []
    with open(inputfile, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            lines.append(line)

    index = 0
    count = 0
    graphs_ids = set()
    graphs = []
    while index < len(lines):
        if lines[index][0][1:] not in graphs_ids:
            graph_id = lines[index][0][1:]
            G = nx.Graph(id=graph_id)

            index += 1
            vert = int(lines[index][0])
            index += 1
            for i in range(vert):
                G.add_node(i, label=lines[index][0])
                index += 1

            edges = int(lines[index][0])
            index += 1
            for i in range(edges):
                G.add_edge(int(lines[index][0]), int(
                    lines[index][1]), label=lines[index][2])
                index += 1

            index += 1

            if not check_graph_size(
                G, min_num_nodes, max_num_nodes, min_num_edges, max_num_edges
            ):
                continue

            if nx.is_connected(G):
                graphs.append(G)
                graphs_ids.add(graph_id)
                count += 1

                if num_graphs and count >= num_graphs:
                    break

        else:
            vert = int(lines[index + 1][0])
            edges = int(lines[index + 2 + vert][0])
            index += vert + edges + 4

    return graphs, count


class Vocab:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = []
    
    def add(self, token):
        if token not in self.token_to_id:
            self.id_to_token.append(token)
            self.token_to_id[token] = len(self.id_to_token) - 1
        return self.token_to_id[token]

    def get_id(self, token):
        return self.token_to_id.get(token, self.token_to_id["<UNK>"])

    def get_token(self, idx):
        if 0 <= idx < len(self.id_to_token):
            return self.id_to_token[idx]
        return "<UNK>"

    def __len__(self):
        return len(self.id_to_token)
    
class DFSCodeDataset(Dataset):
    def __init__(self, dfs_code_list, node_id_vocab, node_label_vocab, edge_label_vocab, max_len):
        """
        dfs_code_list: list of DFS codes (each is a list of tuples)
        max_len: max sequence length (will pad/truncate to this length)
        """
        self.dfs_code_list = dfs_code_list
        self.max_len = max_len
        self.node_id_vocab = node_id_vocab
        self.node_label_vocab = node_label_vocab
        self.edge_label_vocab = edge_label_vocab
        self.pad_id = node_id_vocab.get_id("<PAD>")
        self.start_id = node_id_vocab.get_id("<START>")
        self.end_id = node_id_vocab.get_id("<END>")

    def __len__(self):
        return len(self.dfs_code_list)

    def encode_sequence(self, dfs_code):
        v1_ids, v2_ids, l1_ids, e_ids, l2_ids = [], [], [], [], []

        for v1, v2, l1, e, l2 in dfs_code:
            v1_ids.append(self.node_id_vocab.add(v1))
            v2_ids.append(self.node_id_vocab.add(v2))
            l1_ids.append(self.node_label_vocab.add(l1))
            e_ids.append(self.edge_label_vocab.add(e))
            l2_ids.append(self.node_label_vocab.add(l2))

        return v1_ids, v2_ids, l1_ids, e_ids, l2_ids

    def shift_and_pad(self, seq, pad_id):
        """
        Creates (input, target) pair with input shifted right, target same
        """
        input_seq = [self.start_id] + seq[:-1]
        input_seq += [pad_id] * (self.max_len - len(input_seq))
        target_seq = seq + [pad_id] * (self.max_len - len(seq))
        return input_seq[:self.max_len], target_seq[:self.max_len]

    def __getitem__(self, idx):
        dfs_code = self.dfs_code_list[idx]
        v1, v2, l1, e, l2 = self.encode_sequence(dfs_code)

        v1_in, v1_out = self.shift_and_pad(v1, self.pad_id)
        v2_in, v2_out = self.shift_and_pad(v2, self.pad_id)
        l1_in, l1_out = self.shift_and_pad(l1, self.pad_id)
        e_in,  e_out  = self.shift_and_pad(e, self.pad_id)
        l2_in, l2_out = self.shift_and_pad(l2, self.pad_id)

        return {
            'inputs': {
                'v1': torch.tensor(v1_in, dtype=torch.long),
                'v2': torch.tensor(v2_in, dtype=torch.long),
                'l1': torch.tensor(l1_in, dtype=torch.long),
                'e':  torch.tensor(e_in,  dtype=torch.long),
                'l2': torch.tensor(l2_in, dtype=torch.long),
            },
            'targets': {
                'v1': torch.tensor(v1_out, dtype=torch.long),
                'v2': torch.tensor(v2_out, dtype=torch.long),
                'l1': torch.tensor(l1_out, dtype=torch.long),
                'e':  torch.tensor(e_out,  dtype=torch.long),
                'l2': torch.tensor(l2_out, dtype=torch.long),
            },
        }