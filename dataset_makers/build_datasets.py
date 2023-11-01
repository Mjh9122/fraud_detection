import argparse
import networkx as nx
import numpy as np
import pandas as pd

from build_Xs import build_Xs
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def build_graph(csv):
    """Takes the original dataset and produces a graph with customers and merchants as nodes and transactions as edges.

    Args:
        data_set (Dataframe): The transaction data, columns are
        [step,customer,age,gender,zipcodeOri,merchant,zipMerchant,category,amount,fraud]

    Returns:
        DiGraph: A directed graph with the above properties
    """
    df = pd.read_csv(csv)
    df.drop(columns=['step', 'age', 'gender', 'zipcodeOri', 'zipMerchant', 'category', 'fraud'], inplace=True)
    df.replace("'",'', regex=True, inplace=True) 
    train, test = train_test_split(df, random_state=42)
    G = nx.MultiGraph()

    for _, row in train.iterrows():
        customer, merchant, amount = row
        if customer not in G:
            G.add_node(customer)
        if merchant not in G:
            G.add_node(merchant)
        G.add_edge(customer, merchant, amount=amount)
    return G

def get_lpa_communities_weighted(G, df, merchant = True, customer = True):
    coms = nx.community.asyn_lpa_communities(G, weight='amount')
    com_dic = {}
    for i, com in enumerate(coms):
        for label in list(com):
            com_dic[label] = i

    if merchant:
        df['lpa_merchant'] = df['merchant'].apply(lambda x: com_dic.get(x))

    if customer:
        df['lpa_customer'] = df['customer'].apply(lambda x: com_dic.get(x))

    return df

def add_standard_feature(G, df, func, name, weighted = False, merchant = True, customer = True):
    if weighted:
        feature_dict = func(G, weight='amount')
    else:
        feature_dict = func(G)

    if merchant:
        if weighted:
            df[f'{name}_w_merchant'] = df['merchant'].apply(lambda x: feature_dict.get(x))
        else:
            df[f'{name}_merchant'] = df['merchant'].apply(lambda x: feature_dict.get(x))

    if customer:
        if weighted:
            df[f'{name}_w_customer'] = df['customer'].apply(lambda x: feature_dict.get(x))
        else:
            df[f'{name}_customer'] = df['customer'].apply(lambda x: feature_dict.get(x))

    return df

def main(feature_set): 
    
    if feature_set == 'none':
        X_train, X_test = build_Xs('dataset_makers/original_data.csv', pd.DataFrame(), [2, 3, 2])
        X_train.to_csv('active_datasets/transactional_features_train.csv', index = False)
        X_test.to_csv('active_datasets/transactional_features_test.csv', index = False)
    else:
        G = build_graph('dataset_makers\original_data.csv')
        df = pd.read_csv('dataset_makers\original_data.csv')
        df.drop(columns=['step', 'age', 'gender', 'zipcodeOri', 'zipMerchant', 'category', 'fraud'], inplace=True)
        df.replace("'",'', regex=True, inplace=True) 
        if feature_set == 'original':
            df = add_standard_feature(G, df, nx.degree_centrality, 'degree_centrality')
            df = get_lpa_communities_weighted(G, df)
            df = add_standard_feature(G, df, nx.pagerank, 'page_rank')
            df.drop(columns=['customer', 'merchant', 'amount'], inplace=True)
            X_train, X_test = build_Xs('dataset_makers/original_data.csv', graph_features=df, cuts = [2, 3, 2])
            X_train.to_csv('active_datasets/original_paper_features_train.csv', index = False)
            X_test.to_csv('active_datasets/original_paper_features_test.csv', index = False)
        else:
            df = add_standard_feature(G, df, nx.degree_centrality, 'degree_centrality', False)
            print("1/12")
            df = get_lpa_communities_weighted(G, df)
            print("2/12")
            df = add_standard_feature(G, df, nx.pagerank, 'page_rank', False)
            print("3/12")
            df = add_standard_feature(G, df, nx.pagerank, 'page_rank', True)
            print("4/12")
            df = add_standard_feature(G, df, nx.closeness_centrality, 'closeness_centrality', False)
            print("5/12")
            #df = add_standard_feature(G, df, nx.betweenness_centrality, 'betweenness_centrality', False)
            print("6/12")
            #df = add_standard_feature(G, df, nx.betweenness_centrality, 'betweenness_centrality', True)
            print("7/12")
            df = add_standard_feature(G, df, nx.load_centrality, 'load_centrality', False)
            print("8/12")
            df = add_standard_feature(G, df, nx.load_centrality, 'load_centrality', True)
            print('9/12')
            df = add_standard_feature(G, df, nx.second_order_centrality, 'second_order_centrality', False)
            print('10/12')
            df = add_standard_feature(G, df, nx.centrality.laplacian_centrality, 'laplacian_centrality', False)
            print('11/12')
            df = add_standard_feature(G, df, nx.centrality.laplacian_centrality, 'laplacian_centrality', True)
            print('12/12')
            df.drop(columns =['customer', 'merchant', 'amount'], inplace=True)
            X_train, X_test = build_Xs('dataset_makers/original_data.csv', graph_features=df, cuts = [2, 3, 2])
            X_train.to_csv('active_datasets/all_features_train.csv', index = False)
            X_test.to_csv('active_datasets/all_features_test.csv', index = False)


    #nx.write_graphml(G, './dataset_makers/graph.graphml')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('features', choices=['none', 'all', 'original'])
    args = parser.parse_args()
    main(args.features)