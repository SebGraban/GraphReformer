import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import isomorphism
from joblib import Parallel, delayed

def is_subgraph(g1, g2):
    """
    Check if g1 is a subgraph of g2 (no labels).
    """
    matcher = nx.algorithms.isomorphism.GraphMatcher(g2, g1)
    return matcher.subgraph_is_isomorphic()

def compute_novelty(generated_graphs, training_graphs, n_jobs=10):
    """
    Novelty: % of generated graphs NOT subgraph-isomorphic to any training graph.
    """
    def check_graph(gen_g):
        return not any(is_subgraph(gen_g, train_g) or is_subgraph(train_g, gen_g)
                       for train_g in training_graphs)

    results = Parallel(n_jobs=n_jobs)(
        delayed(check_graph)(gen_g) for gen_g in generated_graphs
    )
    return sum(results) / len(generated_graphs) * 100.0

def compute_uniqueness(generated_graphs, n_jobs=10):
    """
    Uniqueness: % of graphs that are not subgraph-isomorphic to any other generated graph.
    """
    n = len(generated_graphs)

    def is_unique(i):
        gi = generated_graphs[i]
        for j in range(n):
            if i != j:
                gj = generated_graphs[j]
                if is_subgraph(gi, gj) and is_subgraph(gj, gi):
                    return False
        return True

    results = Parallel(n_jobs=n_jobs)(
        delayed(is_unique)(i) for i in range(n)
    )
    return sum(results) / len(generated_graphs) * 100.0


def rbf_kernel(X, Y, sigma=1.0):
    """Compute the RBF (Gaussian) kernel between X and Y."""
    dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-dists / (2 * sigma**2))

def compute_mmd(X, Y, sigma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two samples: X and Y.
    X, Y: numpy arrays of shape (n_samples, n_features)
    """

    if X.shape[1] > Y.shape[1]:
        Y_new = np.zeros((Y.shape[0], X.shape[1]))
        Y_new[:, :Y.shape[1]] = Y
        Y = Y_new
    elif X.shape[1] < Y.shape[1]:
        X_new = np.zeros((X.shape[0], Y.shape[1]))
        X_new[:, :X.shape[1]] = X
        X = X_new

    XX = rbf_kernel(X, X, sigma)
    YY = rbf_kernel(Y, Y, sigma)
    XY = rbf_kernel(X, Y, sigma)

    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd

def degree_distribution(graph):
    """
    Calculate the degree distribution of a graph.
    """
    degrees = [d for n, d in graph.degree()]
    degree_count = {d: degrees.count(d) for d in set(degrees)}
    return degree_count

def clustering_coefficient_distribution(graph):
    """Calculate the clustering coefficient distribution of a graph."""
    return nx.clustering(graph)

def average_distribution(
    graphs,
    measure='degree'
):
    """
    Calculate the average distribution of a measure (degree or clustering coefficient) across multiple graphs.
    """
    total_count = {}
    for g in graphs:
        if measure == 'degree':
            measure_count = degree_distribution(g)
        elif measure == 'clustering':
            measure_count = clustering_coefficient_distribution(g)
        for node_measure, count in measure_count.items():
            if node_measure in total_count:
                total_count[node_measure] += count
            else:
                total_count[node_measure] = count
    return total_count

def sample_distribution(
    graphs,
    measure='degree'
):
    """
    Sample the distribution of a measure (degree or clustering coefficient) across multiple graphs.
    """
    samples = []
    max_len = 0
    for graph in graphs:
        if measure == 'degree':
            measure_count = degree_distribution(graph)
        elif measure == 'clustering':
            measure_count = clustering_coefficient_distribution(graph)
        measure_count = list(measure_count.values())
        if len(measure_count) > max_len:
            max_len = len(measure_count)
        samples.append(measure_count)

    samples_out = np.zeros((len(samples), max_len))
    for i, sample in enumerate(samples):
        samples_out[i, :len(sample)] = sample

    return samples_out

def get_scaler_property(
    graph,
    property_name='degree'
):
    """
    Get a specific scaler property of a graph such as the largest connected component size,
    """
    if property_name == 'largest_component':
        largest_component = max(nx.connected_components(graph), key=len)
        return len(largest_component)
    elif property_name == 'triangle_count':
        return sum(nx.triangles(graph).values()) / 3
    elif property_name == 'characteristic_path_length':
        if nx.is_connected(graph):
            return nx.average_shortest_path_length(graph)
        else:
            return 0
    elif property_name == 'assortativity':
        return nx.degree_assortativity_coefficient(graph)

def get_scaler_properties(
    graphs,
    property_name='degree'
):
    """
    Get a list of specific scaler properties for a list of graphs.
    """
    properties = []
    for graph in graphs:
        prop_value = get_scaler_property(graph, property_name)
        properties.append(prop_value)
    return properties        


def generate_plot(
    data_1,
    data_2,
    title='Distribution Comparison',
    xlabel='Value',
    ylabel='Frequency',
    show_legend=False,
    label_1='Sample 1',
    label_2='Sample 2',
):
    plt.hist(data_1, bins=30, alpha=0.5, label=label_1, color='blue')
    plt.hist(data_2, bins=30, alpha=0.5, label=label_2, color='orange')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if show_legend:
        plt.legend()

def generate_sample_plots(
    graph_samples_1,
    graph_samples_2,
    label_1='Sample 1',
    label_2='Sample 2',
):

    degree_distribution_1 = sample_distribution(graph_samples_1, measure='degree')
    degree_distribution_2 = sample_distribution(graph_samples_2, measure='degree')

    mmd_degree = compute_mmd(degree_distribution_1, degree_distribution_2)
    print(f'MMD for degree distribution: {mmd_degree}')

    clustering_distribution_1 = sample_distribution(graph_samples_1, measure='clustering')
    clustering_distribution_2 = sample_distribution(graph_samples_2, measure='clustering')
    mmd_clustering = compute_mmd(clustering_distribution_1, clustering_distribution_2)
    print(f'MMD for clustering coefficient distribution: {mmd_clustering}')

    largest_component_1 = get_scaler_properties(graph_samples_1, property_name='largest_component')
    largest_component_2 = get_scaler_properties(graph_samples_2, property_name='largest_component')
    jse_largest_component = jensenshannon(largest_component_1, largest_component_2)
    print(f'Jensen-Shannon divergence for largest component size: {jse_largest_component}')

    triangle_count_1 = get_scaler_properties(graph_samples_1, property_name='triangle_count')
    triangle_count_2 = get_scaler_properties(graph_samples_2, property_name='triangle_count')
    jse_triangle_count = jensenshannon(triangle_count_1, triangle_count_2)
    print(f'Jensen-Shannon divergence for triangle count: {jse_triangle_count}')

    characteristic_path_length_1 = get_scaler_properties(graph_samples_1, property_name='characteristic_path_length')
    characteristic_path_length_2 = get_scaler_properties(graph_samples_2, property_name='characteristic_path_length')
    jse_characteristic_path_length = jensenshannon(characteristic_path_length_1, characteristic_path_length_2)
    print(f'Jensen-Shannon divergence for characteristic path length: {jse_characteristic_path_length}')

    assortativity_1 = get_scaler_properties(graph_samples_1, property_name='assortativity')
    assortativity_2 = get_scaler_properties(graph_samples_2, property_name='assortativity')

    fig = plt.figure(figsize=(15, 15))
    plt.subplot(3,2,1)
    generate_plot(largest_component_1, largest_component_2, title='Largest Component Size Distribution', xlabel='Size', ylabel='Frequency',show_legend=True, label_1=label_1, label_2=label_2)
    plt.subplot(3,2,2)
    generate_plot(triangle_count_1, triangle_count_2, title='Triangle Count Distribution', xlabel='Count', ylabel='Frequency', label_1=label_1, label_2=label_2)
    plt.subplot(3,2,3)
    generate_plot(characteristic_path_length_1, characteristic_path_length_2, title='Characteristic Path Length Distribution', xlabel='Length', ylabel='Frequency', label_1=label_1, label_2=label_2)
    plt.subplot(3,2,4)
    generate_plot(assortativity_1, assortativity_2, title='Assortativity Distribution', xlabel='Assortativity', ylabel='Frequency', label_1=label_1, label_2=label_2)

    average_degree_1 = average_distribution(graph_samples_1, measure='degree')
    average_degree_2 = average_distribution(graph_samples_2, measure='degree')
    plt.subplot(3,2,5)
    generate_plot(
        list(average_degree_1.values()),
        list(average_degree_2.values()),
        title='Average Degree Distribution',
        xlabel='Degree',
        ylabel='Frequency',
        label_1=label_1,
        label_2=label_2 
    )
    average_clustering_1 = average_distribution(graph_samples_1, measure='clustering')
    average_clustering_2 = average_distribution(graph_samples_2, measure='clustering')
    plt.subplot(3,2,6)
    generate_plot(
        list(average_clustering_1.values()),
        list(average_clustering_2.values()),
        title='Average Clustering Coefficient Distribution',
        xlabel='Clustering Coefficient',
        ylabel='Frequency',
        label_1=label_1,
        label_2=label_2
    )
    return fig 