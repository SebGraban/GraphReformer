from graph_rnn.helper import get_attributes_len_for_graph_rnn
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import networkx as nx

EPS = 1e-9

class MLP_Softmax(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            MLP_Plain(input_size, embedding_size, output_size, dropout),
            nn.Softmax(dim=2)
        )

    def forward(self, input):
        return self.mlp(input)


class MLP_Log_Softmax(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Log_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            MLP_Plain(input_size, embedding_size, output_size, dropout),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, input):
        return self.mlp(input)


class MLP_Plain(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Plain, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.Linear(embedding_size, embedding_size),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self.mlp(input)


class RNN(nn.Module):
    """
    Custom GRU layer
    :param input_size: Size of input vector
    :param embedding_size: Embedding layer size (finally this size is input to RNN)
    :param hidden_size: Size of hidden state of vector
    :param num_layers: No. of RNN layers
    :param rnn_type: Currently only GRU and LSTM supported
    :param dropout: Dropout probability for dropout layers between rnn layers
    :param output_size: If provided, a MLP softmax is run on hidden state with output of size 'output_size'
    :param output_embedding_size: If provided, the MLP softmax middle layer is of this size, else 
        middle layer size is same as 'embedding size'
    :param device: torch device to instanstiate the hidden state on right device
    """

    def __init__(
        self, input_size, embedding_size, hidden_size, num_layers, rnn_type='GRU',
        dropout=0, output_size=None, output_embedding_size=None,
        device=torch.device('cpu')
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.output_size = output_size
        self.device = device

        self.input = nn.Linear(input_size, embedding_size)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout
            )
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout
            )

        # self.relu = nn.ReLU()

        self.hidden = None  # Need initialization before forward run

        if self.output_size is not None:
            if output_embedding_size is None:
                self.output = MLP_Softmax(
                    hidden_size, embedding_size, self.output_size)
            else:
                self.output = MLP_Softmax(
                    hidden_size, output_embedding_size, self.output_size)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(
                    param, gain=nn.init.calculate_gain('sigmoid'))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        if self.rnn_type == 'GRU':
            # h0
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        elif self.rnn_type == 'LSTM':
            # (h0, c0)
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))

    def forward(self, input, input_len=None):
        input = self.input(input)
        # input = self.relu(input)

        if input_len is not None:
            input = pack_padded_sequence(
                input, input_len.cpu(), batch_first=True, enforce_sorted=False)

        output, self.hidden = self.rnn(input, self.hidden)

        if input_len is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)

        if self.output_size is not None:
            output = self.output(output)

        return output

def create_model_rnn(feature_map, max_prev_node=None, max_head_and_tail=None, device=None):
    len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(len(
        feature_map['node_forward']), len(feature_map['edge_forward']), max_prev_node, max_head_and_tail)
    feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

    node_level_rnn = RNN(
        input_size=feature_len, embedding_size=64,
        hidden_size=128, num_layers=4,
        device=device).to(device=device)

    embedding_node_to_edge = MLP_Plain(
        input_size=128, embedding_size=64,
        output_size=16).to(device=device)

    edge_level_rnn = RNN(
        input_size=len_edge_vec, embedding_size=8,
        hidden_size=16, num_layers=4,
        output_size=len_edge_vec, output_embedding_size=8,
        device=device).to(device=device)

    output_node = MLP_Softmax(
        input_size=128, embedding_size=64,
        output_size=len_node_vec).to(device=device)

    model = {
        'node_level_rnn': node_level_rnn,
        'embedding_node_to_edge': embedding_node_to_edge,
        'edge_level_rnn': edge_level_rnn,
        'output_node': output_node
    }

    return model

def predict_graphs(model, feature_map, device, max_prev_node, max_head_and_tail):
    """
    Generate graphs (networkx format) given a trained generative graphRNN model
    :param eval_args: ArgsEvaluate object
    """

    for _, net in model.items():
        net.eval()

    max_num_node = feature_map['max_nodes']
    len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len_for_graph_rnn(
        len(feature_map['node_forward']), len(feature_map['edge_forward']),
        max_prev_node, max_head_and_tail)
    feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

    graphs = []

    for _ in range(1024 // 32):
        model['node_level_rnn'].hidden = model['node_level_rnn'].init_hidden(
            batch_size=32)

        # [batch_size] * [num of nodes]
        x_pred_node = np.zeros(
            (32, max_num_node), dtype=np.int32)
        # [batch_size] * [num of nodes] * [num_nodes_to_consider]
        x_pred_edge = np.zeros(
            (32, max_num_node, num_nodes_to_consider), dtype=np.int32)

        node_level_input = torch.zeros(
            32, 1, feature_len, device=device)
        # Initialize to node level start token
        node_level_input[:, 0, len_node_vec - 2] = 1
        for i in range(max_num_node):
            # [batch_size] * [1] * [hidden_size_node_level_rnn]
            node_level_output = model['node_level_rnn'](node_level_input)
            # [batch_size] * [1] * [node_feature_len]
            node_level_pred = model['output_node'](node_level_output)
            # [batch_size] * [node_feature_len] for torch.multinomial
            node_level_pred = node_level_pred.reshape(
                32, len_node_vec)
            # [batch_size]: Sampling index to set 1 in next node_level_input and x_pred_node
            # Add a small probability for each node label to avoid zeros
            node_level_pred[:, :-2] += EPS
            # Start token should not be sampled. So set it's probability to 0
            node_level_pred[:, -2] = 0
            # End token should not be sampled if i less than min_num_node
            if i < 0:
                node_level_pred[:, -1] = 0
            sample_node_level_output = torch.multinomial(
                node_level_pred, 1).reshape(-1)
            node_level_input = torch.zeros(
                32, 1, feature_len, device=device)
            node_level_input[torch.arange(
                32), 0, sample_node_level_output] = 1

            # [batch_size] * [num of nodes]
            x_pred_node[:, i] = sample_node_level_output.cpu().data

            # [batch_size] * [1] * [hidden_size_edge_level_rnn]
            hidden_edge = model['embedding_node_to_edge'](node_level_output)

            hidden_edge_rem_layers = torch.zeros(
                model['edge_level_rnn'].num_layers - 1, 32, hidden_edge.size(2),
                device=device)
            # [num_layers] * [batch_size] * [hidden_len]
            model['edge_level_rnn'].hidden = torch.cat(
                (hidden_edge.permute(1, 0, 2), hidden_edge_rem_layers), dim=0)

            # [batch_size] * [1] * [edge_feature_len]
            edge_level_input = torch.zeros(
                32, 1, len_edge_vec, device=device)
            # Initialize to edge level start token
            edge_level_input[:, 0, len_edge_vec - 2] = 1
            for j in range(min(num_nodes_to_consider, i)):
                # [batch_size] * [1] * [edge_feature_len]
                edge_level_output = model['edge_level_rnn'](edge_level_input)
                # [batch_size] * [edge_feature_len] needed for torch.multinomial
                edge_level_output = edge_level_output.reshape(
                    32, len_edge_vec)

                # [batch_size]: Sampling index to set 1 in next edge_level input and x_pred_edge
                # Add a small probability for no edge to avoid zeros
                edge_level_output[:, -3] += EPS
                # Start token and end should not be sampled. So set it's probability to 0
                edge_level_output[:, -2:] = 0
                sample_edge_level_output = torch.multinomial(
                    edge_level_output, 1).reshape(-1)
                edge_level_input = torch.zeros(
                    32, 1, len_edge_vec, device=device)
                edge_level_input[:, 0, sample_edge_level_output] = 1

                # Setting edge feature for next node_level_input
                node_level_input[:, 0, len_node_vec + j * len_edge_vec: len_node_vec + (j + 1) * len_edge_vec] = \
                    edge_level_input[:, 0, :]

                # [batch_size] * [num of nodes] * [num_nodes_to_consider]
                x_pred_edge[:, i, j] = sample_edge_level_output.cpu().data

        # Save the batch of graphs
        for k in range(32):
            G = nx.Graph()

            for v in range(max_num_node):
                # End node token
                if x_pred_node[k, v] == len_node_vec - 1:
                    break
                elif x_pred_node[k, v] < len(feature_map['node_forward']):
                    G.add_node(
                        v, label=feature_map['node_backward'][x_pred_node[k, v]])
                else:
                    print('Error in sampling node features')
                    exit()

            for u in range(len(G.nodes())):
                for p in range(min(num_nodes_to_consider, u)):
                    if x_pred_edge[k, u, p] < len(feature_map['edge_forward']):
                        if max_prev_node is not None:
                            v = u - p - 1
                        elif max_head_and_tail is not None:
                            if p < max_head_and_tail[1]:
                                v = u - p - 1
                            else:
                                v = p - max_head_and_tail[1]

                        G.add_edge(
                            u, v, label=feature_map['edge_backward'][x_pred_edge[k, u, p]])
                    elif x_pred_edge[k, u, p] == len(feature_map['edge_forward']):
                        # No edge
                        pass
                    else:
                        print('Error in sampling edge features')
                        exit()

            # Take maximum connected component
            if len(G.nodes()):
                max_comp = max(nx.connected_components(G), key=len)
                G = nx.Graph(G.subgraph(max_comp))

            graphs.append(G)

    return graphs
