# Python standard library imports
from collections import Counter
import itertools
import random

# Typing imports
from typing import Any, Dict, Tuple, List

# Relative imports
import graph_metrics_module

# Third-party library imports
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns


def findPossiblePeers(edgesProbabilities: Dict[Tuple[int, int], float], threshold: float) -> List[Dict[int, float]]:
    """
    This function finds possible peers for each node in a graph based on edge probabilities and a given threshold.
    
    Args:
        edgesProbabilities (Dict[Tuple[int, int], float]): A dictionary where keys are tuples representing edges between two nodes (indices), 
            and values are the probabilities associated with each edge.
        threshold (float): The probability threshold. If the probability of an edge is greater than this threshold, 
            the nodes connected by this edge are considered as possible peers.

    Returns:
        List[Dict[int, float]]: A list of dictionaries. Each dictionary represents possible peers for a node. 
            The key is the peer index, and the value is the probability of the edge between the node and the peer. 
            If there is no edge between the node and the peer in the input edgesProbabilities, the probability is set to 1.
    """
    possiblePeers = [[] for _ in range(graph_metrics_module.num_nodes)]

    # Identify the initial set of nodes that are not possible peers for each node. 
    # Iterate over all the edges and their associated probabilities. 
    # If the probability of an edge is less than the threshold, the nodes connected by this edge are added to each other's list.
    for edge, prob in edgesProbabilities.items():
        if prob < threshold:
            possiblePeers[edge[0]].append(edge[1])
            possiblePeers[edge[1]].append(edge[0])

    # Refine the list for each node. 
    # Remove the node itself and any duplicates from its list. 
    # Create a new list of nodes that excludes the node itself and any nodes already in its list.
    possiblePeers = list(map(lambda item: list(set(range(graph_metrics_module.num_nodes)) - set(item[1] + [item[0]])), enumerate(possiblePeers)))

    # Replace the list for each node with a dictionary. 
    # The dictionary contains the peer index as the key and the probability of the edge between the node and the peer as the value. 
    # If there is no edge between the node and the peer in the input edgesProbabilities, the probability is set to 1.
    for node, peers in enumerate(possiblePeers):
        d = dict()
        for peer in peers:
            try:
                d[peer] = edgesProbabilities[(min(node, peer), max(node, peer))]
            except:
                d[peer] = 1
        possiblePeers[node] = d

    return possiblePeers


def createGraphSnapshot(maxInteractions: int, possiblePeers: List[Dict[int, float]]) -> nx.Graph:
    """
    This function creates a graph snapshot based on the given number of nodes, maximum interactions, and possible peers.
    It then assigns weights to the edges and nodes based on certain conditions.

    Args:
        maxInteractions (int): The maximum number of interactions for each node.
        possiblePeers (List[Dict[int, float]]): A list of dictionaries where each dictionary represents the possible peers for a node and their probabilities.

    Returns:
        nx.Graph: The created graph snapshot.
    """

    # An empty graph is created and nodes are added to it. The nodes represent entities in the network.
    G = nx.Graph()
    G.add_nodes_from(range(graph_metrics_module.num_nodes))
    # The nodes are shuffled to ensure randomness in the process. This randomness can help in simulating real-world scenarios where interactions are often random.
    nodes = list(range(graph_metrics_module.num_nodes))
    random.shuffle(nodes)
    
    # For each node, the function identifies the achievable peers, i.e., those peers that have not reached their maximum interactions and are not already connected to the node.
    # This step simulates the process of finding potential peers for interaction in a network.
    for node in nodes:
        achievablePeers = [u for u in possiblePeers[node] if maxInteractions > G.degree(u) and u not in G.neighbors(node)]

        # The achievable peers are then sorted by their degrees (number of connections), and a certain number of them are selected for interaction.
        # This step simulates the process of selecting peers for interaction based on their degrees.
        degrees = [G.degree(u) for u in achievablePeers]
        achievablePeers = [node for _, node in sorted(zip(degrees, achievablePeers))]

        # The number of interactions is determined by several factors, including the maximum number of interactions allowed, the current degree of the node, and the number of achievable peers.
        # This step simulates the process of determining the number of interactions for a node.
        interactions = min(3, maxInteractions - G.degree(node), len(achievablePeers))

        if interactions == 0:
            continue
        else:
            # For each interaction, the function assigns a weight based on the frequency of each weight in the existing edges of the node.
            # The weights are chosen in such a way that less frequent weights are preferred.
            # This step simulates the process of assigning weights to interactions based on their frequencies.
            peers = achievablePeers[:interactions]
            counter = dict.fromkeys(["A", "B", "C"], 0)
            counter.update(Counter([e[2]["weight"] for e in G.edges(node, data=True)]))
            weights = [key for key, value in sorted(counter.items(), key=lambda item: item[1])][:interactions]

            # The new edges are then added to the graph. Each edge represents an interaction between two nodes.
            for peer, weight in zip(peers, weights):
                G.add_edge(node, peer, weight=weight)

    # After all interactions have been processed, the function initializes the weights for all nodes.
    # This step prepares the nodes for the assignment of weights based on their interactions.
    for node in range(graph_metrics_module.num_nodes):
        G.nodes[node]["A"] = None
        G.nodes[node]["B"] = None
        G.nodes[node]["C"] = None

    # It then assigns weights to the edges and nodes based on certain conditions.
    # This step simulates the process of assigning weights to nodes and edges based on their interactions.
    activitiesIndex = {"A": [1, 2], "B": [3, 4], "C": [5, 6]}
    edgesToRemove = []
    for u, v, d in G.edges(data="weight"):
        node1 = G.nodes[u][d]
        node2 = G.nodes[v][d]

        # If the weights of the nodes at both ends of an edge are not defined, a random weight is assigned to the edge and the nodes.
        # This simulates a scenario where the interaction is new and has not been categorized yet.
        if node1 is None and node2 is None:
            activity = np.random.choice(activitiesIndex[d])
            G.nodes[u][d] = activity
            G.nodes[v][d] = activity
            G[u][v]["weight"] = activity
        # If the weights of the nodes at both ends of an edge are defined and equal, the weight of the edge is set to this common weight.
        # This simulates a scenario where the interaction is consistent between the nodes.
        elif node1 is not None and node2 is not None:
            if node1 == node2:
                G[u][v]["weight"] = node1
            # If the weights of the nodes at both ends of an edge are defined but different, the edge is marked for removal.
            # This simulates a scenario where there is a disagreement or inconsistency in the interaction.
            else:
                edgesToRemove.append((u,v))
        # If the weight of the node at one end of an edge is defined but the weight of the node at the other end is not, the weight of the edge and the undefined node is set to the defined weight.
        # This simulates a scenario where one node influences or determines the interaction.
        else:
            activity = node1 or node2
            G.nodes[u][d] = activity
            G.nodes[v][d] = activity
            G[u][v]["weight"] = activity

    # Finally, the function removes all edges that were marked for removal and returns the resulting graph.
    # This step ensures that the final graph only contains edges that satisfy the specified conditions.
    G.remove_edges_from(edgesToRemove)

    return G


def createGraphSequence(maxTimestamp: int, maxInteractions: int, possiblePeers: List[Dict[int, float]]) -> Dict[str, Any]:
    """
    This function creates a sequence of graph snapshots for each timestamp from 1 to the given maxTimestamp.
    Each snapshot is created with a maximum number of interactions among the possible peers.
    The sequence of these snapshots is then returned.

    Args:
        maxTimestamp (int): The number of timestamps for which to create graph snapshots.
        maxInteractions (int): The maximum number of interactions for each node in each snapshot.
        possiblePeers (List[Dict[int, float]]): A list of dictionaries where each dictionary represents the possible peers for a node and their probabilities.

    Returns:
        dict: A dictionary containing the final list of graph sequences, the average diversity, and the average number of isolated nodes.
    """

    # Initialize an empty list to store the graph sequences. This will hold the network structure at each timestamp.
    graph_sequence = []

    # Initialize an empty list to store the diversity of each node. This will hold the measure of how different each node's connections are.
    node_diversities = []

    # For each timestamp, a graph snapshot is created using the `createGraphSnapshot` function. 
    # The snapshot represents the state of the network at that particular time.
    for t in range(1, maxTimestamp + 1) :
        graph_sequence.append(createGraphSnapshot(maxInteractions=maxInteractions, possiblePeers=possiblePeers))

    # For each node in the network, the function calculates a diversity score. 
    # This score is based on the number of unique neighbors the node has across all graph snapshots, normalized by the total number of nodes minus one.
    for node in range(graph_metrics_module.num_nodes) :
        node_diversity = len(set(sum([list(g.neighbors(node)) for g in graph_sequence], [])))/(graph_metrics_module.num_nodes - 1)
        node_diversities.append(node_diversity)

    # The function defines a helper function to calculate the number of isolated nodes in a graph sequence.
    # An isolated node is a node with no connections.
    # This function iterates over each graph in the sequence, counts the number of isolated nodes, and returns the average.
    def find_number_of_isolated_nodes(graph_sequence) :
        sequence_isolated_nodes = []
        for graph in graph_sequence :
            sequence_isolated_nodes.append(sum(1 for node, degree in graph.degree() if degree == 0)/graph_metrics_module.num_nodes)
        return np.average(sequence_isolated_nodes)

    # The function returns a dictionary containing the final list of graph sequences, the average node diversity, and the average number of isolated nodes.
    return {"graphs": graph_sequence, "avg diversity": np.average(node_diversities), "number of isolated nodes": find_number_of_isolated_nodes(graph_sequence)}


def computeGraphMetrics(edgesProbabilities: Dict[Tuple[int, int], float], threshold: float, maxInteractions: int, maxTimestamp: int) -> Dict[str, float]:
    """
    This function calculates the average diversity of a graph sequence created based on the given parameters.
    
    Args:
        edgesProbabilities (Dict[Tuple[int, int], float]): A dictionary representing the probabilities of edges in the graph.
        threshold (float): The threshold value to find possible peers.
        maxInteractions (int): The maximum number of interactions for creating the graph sequence.
        maxTimestamp (int): The maximum timestamp for creating the graph sequence.

    Returns:
        Dict[str, float]: A dictionary containing the average diversity, the average number of isolated nodes, and the ges metric of the created graph sequence.
    """
    # The function first finds the possible peers for each node in the graph based on the given edges probabilities and threshold.
    possiblePeers = findPossiblePeers(edgesProbabilities=edgesProbabilities, threshold=threshold)
    
    # Then, it creates a sequence of graph snapshots for each timestamp using the `createGraphSequence` function. 
    # The snapshot represents the state of the network at that particular time.
    graph_sequence = createGraphSequence(maxTimestamp=maxTimestamp, maxInteractions=maxInteractions, possiblePeers=possiblePeers)

    # The function defines a helper function to calculate the ges metric of a graph sequence.
    # The ges metric is calculated as the average of the ges values for each pair of consecutive graphs in the sequence.
    # The ges value for a pair of graphs is calculated as twice the number of common edges divided by the total number of edges in both graphs.
    def find_ges_metric(graph_sequence) :
        def calculate_ges(G1, G2) :
            edges_G1 = set(G1.edges())
            edges_G2 = set(G2.edges())
            common_edges = edges_G1 & edges_G2
            return 2*len(common_edges)/(len(edges_G1) + len(edges_G2))
    
        sequence_ges = []
        for pair in list(zip(graph_sequence, graph_sequence[1:])) :
            sequence_ges.append(calculate_ges(pair[0], pair[1]))
        
        return np.average(sequence_ges)
    
    # The function returns a dictionary containing the average diversity, the average number of isolated nodes, and the ges metric of the created graph sequence.
    return {"diversity": graph_sequence["avg diversity"], "number of isolated nodes": graph_sequence["number of isolated nodes"], "ges metric": find_ges_metric(graph_sequence["graphs"])}


def compute_graph_metrics_results(thresholds: List[float], interactions: List[int], timestamps: List[int], edges_probabilities: Dict[str, Any]) -> Dict[str, Dict[Tuple[float, int, int], Any]]:
    """
    Function to compute graph metrics for different models given a range of thresholds, interactions, and timestamps.

    Args:
    thresholds (List[float]): List of threshold values to consider.
    interactions (List[int]): List of interaction values to consider.
    timestamps (List[int]): List of timestamp values to consider.
    edges_probabilities (Dict[str, Any]): Dictionary where keys are model names and values are corresponding edge probabilities.

    Returns:
    results (Dict[str, Dict[Tuple[float, int, int], Any]]): Dictionary where keys are model names and values are dictionaries. 
                                                             Each inner dictionary's keys are tuples of (threshold, interaction, timestamp) 
                                                             and values are the result of computeGraphMetrics for that combination of parameters.
    """
    # Generate all possible combinations of thresholds, interactions, and timestamps
    combinations = list(itertools.product(thresholds, interactions, timestamps))

    # Initialize a dictionary to store results
    results = {}

    # Calculate results for each combination of parameters for each model
    for model, edges in edges_probabilities.items():
        results[model] = {}
        for combination in combinations:
            results[model][combination] = computeGraphMetrics(edges, *combination)

    return results

def plot_graph_metrics(metrics: List[str], thresholds: List[float], interactions: List[int], timestamps: List[int], edges_probabilities: Dict[str, Any]) -> None:
    """
    This function calls the compute_results function to get the results of BERT and Ada models for different combinations of thresholds, interactions, and timestamps.
    It then plots diagrams for each given metric based on the results.

    Args:
        metrics (List[str]): The list of metrics for which to plot diagrams.
        thresholds (List[float]): List of threshold values to consider.
        interactions (List[int]): List of interaction values to consider.
        timestamps (List[int]): List of timestamp values to consider.
        edges_probabilities (Dict[str, Any]): Dictionary where keys are model names and values are corresponding edge probabilities.

    Returns:
        None
    """
    # Call the compute_results function to get the results of BERT and Ada models
    results = compute_graph_metrics_results(thresholds, interactions, timestamps, edges_probabilities)
    for metric in metrics :
        # For each interaction, the function slices the results of BERT and Ada models based on the interaction and timestamp.
        for inter in interactions:
            BERT_results_slices = [{key:value[metric] for key, value in results['BERT'].items() if key[1:] == (inter,t)} for t in timestamps]
            Ada_results_slices = [{key:value[metric] for key, value in results['Ada'].items() if key[1:] == (inter,t)} for t in timestamps]
    
            # The function then creates a subplot for each timestamp.
            fig, ax = plt.subplots(1,len(timestamps), figsize=(24,4))
            title = fig.suptitle(f'Analysis of {metric} Variation with Threshold for maxInteractions = {inter} and different maxTimestamps', y=1.05)
            title.set_weight("bold")
    
            # For each timestamp, the function plots the results of BERT and Ada models.
            for i,j in enumerate(timestamps):
                sns.lineplot(x=thresholds, y=list(BERT_results_slices[i].values()), label="BERT", ax=ax[i])
                sns.lineplot(x=thresholds, y=list(Ada_results_slices[i].values()), label="Ada", ax=ax[i])
                ax[i].set_xlabel("threshold")
                if i == 0:
                    ax[i].set_ylabel(metric)
                ax[i].set_title(f'timestamp = {j}')
                ax[i].legend(loc='best')
                
        # The function adjusts the subplots and displays the plots.
        fig.subplots_adjust(wspace=0.3, hspace=0.5, bottom=0.15, top=0.9)
        plt.show()

