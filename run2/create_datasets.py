import argparse
import feature_functions
import networkx as nx
import numpy as np
import pandas as pd
import sys


def build_graph(dataframe):
    """Takes the transaction dataframe and produces a directed graph with customers and merchants as nodes and transactions as edges.

    Args:
        data_set (Dataframe): The transaction data, columns are
        [step,customer,age,gender,zipcodeOri,merchant,zipMerchant,category,amount,fraud]

    Returns:
        DiGraph: A directed graph with the above properties
    """
    G = nx.MultiDiGraph()

    for index, row in dataframe.iterrows():
        step, customer, age, _, gender, merchant, _, category, amount, _ = row
        if customer not in G:
            G.add_node(customer, age=age, gender=gender)
        if merchant not in G:
            G.add_node(merchant)
        G.add_edge(customer, merchant, amount=amount, category=category)
    return G

def add_graph_features(feature_funcs, graph, nodes, dataframe):
    for func in feature_funcs:
        func(graph, nodes, dataframe)

def save_data(dataframe, graph_features):
    dataframe.to_csv(f'feature_csvs/transactions_all.csv', index=False )
    non_graph_columns = [col for col in dataframe.columns if col not in graph_features]
    dataframe[non_graph_columns].to_csv('feature_csvs/transactions_none.csv', index=False)
    for feature in graph_features:
        non_graph_columns.append(feature)
        dataframe[non_graph_columns].to_csv(f'feature_csvs/transactions_{feature.replace(" ", "_")}.csv', index=False)
        non_graph_columns.remove(feature)


def main(feature_functions, feature_names, verbose = False):
    # Read in raw transaction data
    transactions_df = pd.read_csv('original_data.csv')
    if verbose:
        print("read csv")
    transactions_df.replace("'",'', regex=True, inplace=True) 

    transaction_graph = build_graph(transactions_df)
    if verbose:
        print('built graph')

    # Generate a list of all of the customer nodes
    customer_nodes = [n for n in transaction_graph if 'C' in n]

    # Set graph features to generate
    graph_feature_functions = feature_functions
    

    add_graph_features(graph_feature_functions, transaction_graph, customer_nodes, transactions_df)
    if verbose:
        print('Added graph features')


    graph_features = feature_names
    # Send data to csvs
    save_data(transactions_df, graph_features)
    
    if verbose:
        print('Saved data')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='create_datasets.py',
                    description='For each feature function passed, creates a csv file with the transactional data and that feature. Also creates csvs for all features and none')
    parser.add_argument('-v', '--verbose', default = False, action='store_true')
    parser.add_argument('feature_functions', type=str, nargs='+', action='append')
    args = parser.parse_args()
    funcs = []
    for func in args.feature_functions[0]:
        funcs.append(getattr(feature_functions, func))
    main(feature_functions=funcs, feature_names = args.feature_functions[0], verbose=args.verbose)