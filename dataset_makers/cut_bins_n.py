import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  MinMaxScaler


def main(cut1, cut2, cut3):
    transaction_df = pd.read_csv('./dataset_makers/original_data.csv')
    transaction_df.replace("'",'', regex=True, inplace=True) 
    #Drop unused columns
    transaction_df.drop(columns = ['step', 'age', 'gender', 'zipcodeOri', 'zipMerchant'], inplace=True)
    #Split data to train, test
    train, test = train_test_split(transaction_df, random_state=42)
    
    merchant_fraud_rate = pd.cut(train.groupby('merchant').mean('fraud')['fraud'], bins = cut1, labels=range(cut1))
    train['merchant fraud rate'] = train['merchant'].apply(lambda x: merchant_fraud_rate.get(x))
    test['merchant fraud rate'] = test['merchant'].apply(lambda x: merchant_fraud_rate.get(x))
    
    customer_previous_fraud = pd.cut(train.groupby('customer').mean('fraud')['fraud'], bins = cut2, labels=range(cut2))
    train['previous fraud'] = train['customer'].apply(lambda x: customer_previous_fraud.get(x))
    test['previous fraud'] = test['customer'].apply(lambda x: customer_previous_fraud.get(x))
    
    category_fraud_rate = pd.cut(train.groupby('category').mean('fraud')['fraud'], bins = cut3, labels = range(cut3))
    train['category fraud rate'] = train['category'].apply(lambda x: category_fraud_rate.get(x))
    test['category fraud rate'] = test['category'].apply(lambda x: category_fraud_rate.get(x))
    
    train.drop(columns=['customer', 'merchant', 'category'], inplace=True)
    test.drop(columns=['customer', 'merchant', 'category'], inplace=True)
    
    X_train = train.drop(columns = ['fraud'])
    X_test = test.drop(columns = ['fraud'])
    
    MMscaler = MinMaxScaler()
    X_train_transformed = MMscaler.fit_transform(X_train)
    X_test_transformed = MMscaler.transform(X_test)
    
    X_train_df = pd.DataFrame(X_train_transformed)
    X_train_df.to_csv(f'./active_datasets/no_graph_train.csv', index = None, header=None)
    X_test_df = pd.DataFrame(X_test_transformed)
    X_test_df.to_csv(f'./active_datasets/no_graph_test.csv', index = None, header=None)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('cut1')
    parser.add_argument('cut2')
    parser.add_argument('cut3')

    args = parser.parse_args()
    main(int(args.cut1), int(args.cut2), int(args.cut3))