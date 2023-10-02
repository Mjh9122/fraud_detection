import networkx as nx
import numpy as np
import pandas as pd


def degree_centrality(graph, nodes, dataframe):
    """Adds a degree centrality column to the dataframe, calculated along the nodes given, where degree centrality = (in_degree + out_degree)/(num nodes in G)

    Args:
        graph (nx.DiGraph): The graph of the transaction data
        nodes (List()): The list of nodes to add to the column. It is assumed that these nodes have an index attribute, and that index is where the measure will be added
        dataframe (DataFrame): The Dataframe of the transaction data, the column is added to this Dataframe
    """
    degree_centralities = nx.centrality.degree_centrality(graph)
    dataframe['degree_centrality'] = {graph.nodes[t]['index']: degree_centralities[t] for t in nodes}

def page_rank(graph, nodes, dataframe):
    """Adds a pagerank column to the dataframe, calculated along the nodes given

    Args:
        graph (nx.DiGraph): The graph of the transaction data
        nodes (List()): The list of nodes to add to the column. It is assumed that these nodes have an index attribute, and that index is where the  measure will be added
        dataframe (DataFrame): The Dataframe of the transaction data, the column is added to this Dataframe
    """
    page_ranks = nx.pagerank(graph)
    dataframe['page_rank'] = {graph.nodes[t]['index']: page_ranks[t] for t in nodes}

def lpa_community(graph, nodes, dataframe):
    """Adds a community column to the dataframe, calculated along the nodes given

    Args:
        graph (nx.DiGraph): The graph of the transaction data
        nodes (List()): The list of nodes to add to the column. It is assumed that these nodes have an index attribute, and that index is where the  measure will be added
        dataframe (DataFrame): The Dataframe of the transaction data, the column is added to this Dataframe
    """
    lpa_communities = nx.community.label_propagation_communities(graph.to_undirected())
    communities = np.zeros(len(dataframe), dtype=int)
    for index, community in enumerate(lpa_communities):
        # All nodes in the community
        community_nodes_np = np.array(list(community))
        # Mask for transaction nodes
        community_t_mask = np.vectorize(lambda x: 'T' in x)(community_nodes_np)
        # Filter for transaction nodes
        community_t_nodes = community_nodes_np[community_t_mask]
        # Remove the Ts to get index
        community_indices_np = np.vectorize(lambda s: s[1:])(community_t_nodes)
        # Cast to ints
        community_indices_np = community_indices_np.astype(int)
        # Fill mask with t_nodes
        communities[community_indices_np] = index
        # Set the community value of these indicies to the index
    dataframe['lpa_community'] = communities

functions = [degree_centrality, page_rank, lpa_community]
function_names = ['degree_centrality', 'page_rank', 'lpa_community']