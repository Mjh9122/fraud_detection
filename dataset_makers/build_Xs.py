import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  MinMaxScaler


def build_Xs(original, graph_features, cuts):
    df = pd.read_csv(original)
    df.replace("'",'', regex=True, inplace=True) 
    df = pd.concat([df,graph_features], axis=1)
    #Drop unused columns
    df.drop(columns = ['step', 'age', 'gender', 'zipcodeOri', 'zipMerchant'], inplace=True)
    #Split data to train, test
    train, test = train_test_split(df, random_state=42)
    
    merchant_fraud_rate = pd.cut(train.groupby('merchant').mean('fraud')['fraud'], cuts[0], labels=False)
    train['merchant_fraud_rate'] = train['merchant'].apply(lambda x: merchant_fraud_rate.get(x))
    test['merchant_fraud_rate'] = test['merchant'].apply(lambda x: merchant_fraud_rate.get(x))
    
    customer_previous_fraud = pd.cut(train.groupby('customer').mean('fraud')['fraud'], cuts[1], labels=False)
    train['customer_fraud_rate'] = train['customer'].apply(lambda x: customer_previous_fraud.get(x))
    test['customer_fraud_rate'] = test['customer'].apply(lambda x: customer_previous_fraud.get(x))
    
    category_fraud_rate = pd.cut(train.groupby('category').mean('fraud')['fraud'], cuts[2], labels=False)
    train['category_fraud_rate'] = train['category'].apply(lambda x: category_fraud_rate.get(x))
    test['category_fraud_rate'] = test['category'].apply(lambda x: category_fraud_rate.get(x))
    
    train.drop(columns=['customer', 'merchant', 'category'], inplace=True)
    test.drop(columns=['customer', 'merchant', 'category'], inplace=True)
    
    X_train = train.drop(columns = ['fraud'])
    X_test = test.drop(columns = ['fraud'])
    
    MMscaler = MinMaxScaler()
    X_train_transformed = MMscaler.fit_transform(X_train)
    X_test_transformed = MMscaler.transform(X_test)
    
    X_train_df = pd.DataFrame(X_train_transformed, columns=X_train.columns)
    
    X_test_df = pd.DataFrame(X_test_transformed, columns=X_test.columns)
    
    return X_train_df, X_test_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('cuts')
    args = parser.parse_args()
    build_Xs(int(args.cuts))