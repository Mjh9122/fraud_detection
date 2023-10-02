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
    dataframe.assign(degree_centrality = np.zeros(len(dataframe)))
    degree_centralities = nx.centrality.degree_centrality(graph)
    for n in nodes:
        dataframe.loc[dataframe['customer'] == n, 'degree_centrality'] = degree_centralities[n]

def page_rank(graph, nodes, dataframe):
    """Adds a pagerank column to the dataframe, calculated along the nodes given

    Args:
        graph (nx.DiGraph): The graph of the transaction data
        nodes (List()): The list of nodes to add to the column. It is assumed that these nodes have an index attribute, and that index is where the  measure will be added
        dataframe (DataFrame): The Dataframe of the transaction data, the column is added to this Dataframe
    """
    dataframe.assign(page_rank = np.zeros(len(dataframe)))
    page_ranks = nx.pagerank(graph.to_undirected())
    for n in nodes:
        dataframe.loc[dataframe['customer'] == n, 'page_rank'] = page_ranks[n]

def lpa_community(graph, nodes, dataframe):
    """Adds a community column to the dataframe, calculated along the nodes given

    Args:
        graph (nx.DiGraph): The graph of the transaction data
        nodes (List()): The list of nodes to add to the column.
        dataframe (DataFrame): The Dataframe of the transaction data, the column is added to this Dataframe
    """
    lpa_communities = nx.community.asyn_lpa_communities(graph.to_undirected(), weight='amount', seed=42)
    dataframe.assign(lpa_community = np.zeros(len(dataframe), dtype=int))
    for index, community in enumerate(lpa_communities):
        dataframe.loc[dataframe['customer'].isin(community), 'lpa_community'] = index

functions = [degree_centrality, page_rank, lpa_community]
function_names = ['degree_centrality', 'page_rank', 'lpa_community']