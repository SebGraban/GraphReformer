import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import networkx as nx
from torch.distributions.categorical import Categorical
from graphgen.dfscode.dfs_wrapper import graph_from_dfscode

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


def create_model(feature_map, device):
    max_nodes = feature_map['max_nodes']
    len_node_vec, len_edge_vec = len(
        feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1

    feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_edge_vec

    MLP_layer = MLP_Softmax

    dfs_code_rnn = RNN(
        input_size=feature_len, embedding_size=92,
        hidden_size=256, num_layers=4,
        rnn_type='LSTM', dropout=0.2,
        device=device).to(device=device)

    output_timestamp1 = MLP_layer(
        input_size=256, embedding_size=512,
        output_size=max_nodes + 1, dropout=0.2).to(device=device)

    output_timestamp2 = MLP_layer(
        input_size=256, embedding_size=512,
        output_size=max_nodes + 1, dropout=0.2).to(device=device)

    output_vertex1 = MLP_layer(
        input_size=256, embedding_size=512,
        output_size=len_node_vec, dropout=0.2).to(device=device)

    output_vertex2 = MLP_layer(
        input_size=256, embedding_size=512,
        output_size=len_node_vec, dropout=0.2).to(device=device)

    model = {
        'dfs_code_rnn': dfs_code_rnn,
        'output_timestamp1': output_timestamp1,
        'output_timestamp2': output_timestamp2,
        'output_vertex1': output_vertex1,
        'output_vertex2': output_vertex2
    }

    output_edge = MLP_layer(
        input_size=256, embedding_size=512,
        output_size=len_edge_vec, dropout=0.2).to(device=device)
    model['output_edge'] = output_edge

    return model

def predict_graphgen_graphs(model, feature_map, device):

    for _, net in model.items():
        net.eval()

    max_nodes = feature_map['max_nodes']
    len_node_vec, len_edge_vec = len(
        feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1
    feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_edge_vec

    graphs = []

    for _ in range(1024 // 32):
        # initialize dfs_code_rnn hidden according to batch size
        model['dfs_code_rnn'].hidden = model['dfs_code_rnn'].init_hidden(
            batch_size=32)

        rnn_input = torch.zeros(
            (32, 1, feature_len), device=device)
        pred = torch.zeros(
            (32, 50, 5), device=device)

        for i in range(50):
            rnn_output = model['dfs_code_rnn'](rnn_input)

            # Evaluating dfscode tuple
            timestamp1 = model['output_timestamp1'](
                rnn_output).reshape(32, -1)
            timestamp2 = model['output_timestamp2'](
                rnn_output).reshape(32, -1)
            vertex1 = model['output_vertex1'](
                rnn_output).reshape(32, -1)
            edge = model['output_edge'](rnn_output).reshape(
                32, -1)
            vertex2 = model['output_vertex2'](
                rnn_output).reshape(32, -1)

            timestamp1 = Categorical(timestamp1).sample()
            timestamp2 = Categorical(timestamp2).sample()
            vertex1 = Categorical(vertex1).sample()
            edge = Categorical(edge).sample()
            vertex2 = Categorical(vertex2).sample()

            rnn_input = torch.zeros(
                (32, 1, feature_len), device=device)

            rnn_input[torch.arange(32), 0, timestamp1] = 1
            rnn_input[torch.arange(32),
                      0, max_nodes + 1 + timestamp2] = 1
            rnn_input[torch.arange(32),
                      0, 2 * max_nodes + 2 + vertex1] = 1
            rnn_input[torch.arange(32), 0,
                      2 * max_nodes + 2 + len_node_vec + edge] = 1
            rnn_input[torch.arange(32), 0, 2 *
                      max_nodes + 2 + len_node_vec + len_edge_vec + vertex2] = 1

            pred[:, i, 0] = timestamp1
            pred[:, i, 1] = timestamp2
            pred[:, i, 2] = vertex1
            pred[:, i, 3] = edge
            pred[:, i, 4] = vertex2

        nb = feature_map['node_backward']
        eb = feature_map['edge_backward']
        for i in range(32):
            dfscode = []
            for j in range(50):
                if pred[i, j, 0] == max_nodes or pred[i, j, 1] == max_nodes \
                        or pred[i, j, 2] == len_node_vec - 1 or pred[i, j, 3] == len_edge_vec - 1 \
                        or pred[i, j, 4] == len_node_vec - 1:
                    break

                dfscode.append(
                    (int(pred[i, j, 0].data), int(pred[i, j, 1].data), nb[int(pred[i, j, 2].data)],
                     eb[int(pred[i, j, 3].data)], nb[int(pred[i, j, 4].data)]))

            graph = graph_from_dfscode(dfscode)

            # Remove self loops
            graph.remove_edges_from(nx.selfloop_edges(graph))

            # Take maximum connected component
            if len(graph.nodes()):
                max_comp = max(nx.connected_components(graph), key=len)
                graph = nx.Graph(graph.subgraph(max_comp))

            graphs.append(graph)

    return graphs
