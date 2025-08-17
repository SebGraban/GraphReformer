import numpy as np
import networkx as nx

def get_enzymes_data():

    G = nx.Graph()
    # load data
    data_adj = np.loadtxt('/root/thesus/graphreformer/datasests/ENZYMES/ENZYMES_A.txt', delimiter=',').astype(int)
    data_node_att = np.loadtxt('/root/thesus/graphreformer/datasests/ENZYMES/ENZYMES_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt('/root/thesus/graphreformer/datasests/ENZYMES/ENZYMES_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt('/root/thesus/graphreformer/datasests/ENZYMES/ENZYMES_graph_indicator.txt', delimiter=',').astype(int)
    data_graph_labels = np.loadtxt('/root/thesus/graphreformer/datasests/ENZYMES/ENZYMES_graph_labels.txt', delimiter=',').astype(int)

    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_att.shape[0]):
        G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    print(G.number_of_nodes())
    print(G.number_of_edges())

    graphs = []
    node_num_list = []

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1

    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        G_sub.graph['id'] = data_graph_labels[i]

        if nx.is_connected(G_sub):
            G_sub = nx.convert_node_labels_to_integers(G_sub)
            G_sub.remove_edges_from(nx.selfloop_edges(G_sub))

            for node in G_sub.nodes():
                node_label = str(G_sub.nodes[node]['label'])
                node_label += '-' + str(G_sub.degree[node])
                G_sub.nodes[node]['label'] = node_label

            nx.set_edge_attributes(G_sub, 'DEFAULT_LABEL', 'label')

            graphs.append(G_sub)
            node_num_list.append(G_sub.number_of_nodes())

    return graphs, node_num_list


