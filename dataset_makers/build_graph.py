import networkx as nx
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm

def build_graph(dataframe):
    """Takes the transaction dataframe and produces a directed graph with customers and merchants as nodes and transactions as edges.

    Args:
        data_set (Dataframe): The transaction data, columns are
        [step,customer,age,gender,zipcodeOri,merchant,zipMerchant,category,amount,fraud]

    Returns:
        DiGraph: A directed graph with the above properties
    """
    G = nx.MultiDiGraph()

    for _, row in dataframe.iterrows():
        customer, merchant, amount = row
        if customer not in G:
            G.add_node(customer)
        if merchant not in G:
            G.add_node(merchant)
        G.add_edge(customer, merchant, amount=amount)
    return G

df = pd.read_csv('./dataset_makers/original_data.csv')
df.drop(columns=['step', 'age', 'gender', 'zipcodeOri', 'zipMerchant', 'category', 'fraud'], inplace=True)
df.replace("'",'', regex=True, inplace=True) 
train, test = train_test_split(df, random_state=42)
#print(train.head())

G = build_graph(train)
nx.write_graphml(G, './dataset_makers/graph.graphml')