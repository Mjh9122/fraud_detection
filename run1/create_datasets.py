import argparse
import networkx as nx
import numpy as np
import pandas as pd
import sys


def build_graph(dataframe):
    """Takes the transaction dataframe and produces a directed graph structured exactly as the paper illustrates.
    One row in the dataset is turned into two edges, one from the customer to the transaction node, and one from the transaction node to the merchant node.
    Customer nodes have age and gender values,
    .
    Transaction nodes have category and amount values,
    Merchant nodes have zipcode values

    Args:
        data_set (Dataframe): The transaction data, columns are
        [step,customer,age,gender,zipcodeOri,merchant,zipMerchant,category,amount,fraud]

    Returns:
        DiGraph: A directed graph with the above properties
    """
    G = nx.DiGraph()

    for index, row in dataframe.iterrows():
        step, customer, age, _, gender, merchant, zipMerchant, category, amount, fraud = row
        if customer not in G:
            G.add_node(customer, age=age, gender=gender)
        if merchant not in G:
            G.add_node(merchant, zipcode=zipMerchant)
        G.add_node(f'T{index}', index=index, weight=amount, category=category)
        G.add_edge(customer, f'T{index}')
        G.add_edge(f'T{index}', merchant)
    
    return G

def add_graph_features(feature_funcs, graph, nodes, dataframe):
    for func in feature_funcs:
        func(graph, nodes, dataframe)

def save_data(dataframe, graph_features, directory):
    dataframe.to_csv(f'{directory}/transactions_all.csv', index=False )
    non_graph_columns = [col for col in dataframe.columns if col not in graph_features]
    dataframe[non_graph_columns].to_csv(f'{directory}/transactions_none.csv', index=False)
    for feature in graph_features:
        non_graph_columns.append(feature)
        dataframe[non_graph_columns].to_csv(f'{directory}/transactions_{feature.replace(" ", "_")}.csv', index=False)
        non_graph_columns.remove(feature)

def main(feature_functions, feature_names, original_data, directory, verbose = False):
    # Read in raw transaction data
    transactions_df = pd.read_csv(original_data)
    if verbose:
        print("read csv")
    transactions_df.replace("'",'', regex=True, inplace=True) 

    transaction_graph = build_graph(transactions_df)
    if verbose:
        print('built graph')

    # Generate a list of all of the transaction nodes
    transaction_nodes = [n for n in transaction_graph if 'T' in n]

    # Set graph features to generate
    graph_feature_functions = feature_functions
    

    add_graph_features(graph_feature_functions, transaction_graph, transaction_nodes, transactions_df)
    if verbose:
        print('Added graph features')


    graph_features = feature_names
    # Send data to csvs
    save_data(transactions_df, graph_features, directory)
    
    if verbose:
        print('Saved data')
