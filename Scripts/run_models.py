import pandas as pd
import numpy as np

from os import listdir
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, precision_score, roc_auc_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

def score_model(model, X_test, y_test, map = None):
    model_preds = model.predict(X_test)
    scores = {}
    if map:
        model_preds = [map[i] for i in model_preds]
    evaluation_funcs = {'accuracy':accuracy_score, 'precision':precision_score, 'recall':recall_score, 'f1':f1_score, 'MCC':matthews_corrcoef}
    for name, score in evaluation_funcs.items():
        scores[name] = score(y_pred = model_preds, y_true = y_test)
    scores['ROC-AUC'] = roc_auc_score(y_score = model_preds, y_true = y_test)
    scores['AUPR'] = average_precision_score(y_score = model_preds, y_true = y_test)
    return scores


def main(source, target):
    csvs = listdir(source)

    for csv in tqdm(csvs):
        # Read in data
        transaction_df = pd.read_csv(f'{source}/{csv}')
        #Split data to x, y, train, test
        y = transaction_df['fraud']
        X = transaction_df.drop(columns=['fraud'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        #Fit transformer to data
        transformer = ColumnTransformer([('One Hot Encoder', OneHotEncoder(drop='first'), ['category', 'gender']),
                                 ('Age Pipe', Pipeline([('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)), ('scale', MinMaxScaler())]), ['age']),
                                 ('MinMaxScaler', MinMaxScaler(), ['amount', 'step']),
                                 ('drop', 'drop', ['zipcodeOri', 'zipMerchant', 'customer', 'merchant'])], remainder = MinMaxScaler(), sparse_threshold=0)
        transformer.fit(X_train)

        #Build and score models

        #Decision Tree
        decision_tree = Pipeline(steps=[('transformer', transformer), ('model', DecisionTreeClassifier(random_state=42))])
        decision_tree.fit(X_train, y_train)
        desc_tree_scores = score_model(decision_tree, X_test, y_test)

        # Random Forest
        random_forest = Pipeline(steps=[('transformer', transformer), ('model', RandomForestClassifier(random_state=42))])
        random_forest.fit(X_train, y_train)
        random_forest_scores = score_model(random_forest, X_test, y_test)

        # KNN
        KNN = Pipeline(steps=[('transformer', transformer), ('model', KNeighborsClassifier())])
        KNN.fit(X_train, y_train)
        KNN_scores = score_model(KNN, X_test, y_test)

        #Multilayer perceptron
        MLP = Pipeline(steps=[('transformer', transformer), ('model', MLPClassifier(hidden_layer_sizes=(15, 15, 15), random_state=42))])
        MLP.fit(X_train, y_train)
        MLP_scores = score_model(MLP, X_test, y_test)

        #Support Vector machine
        SVM = Pipeline(steps=[('transformer', transformer), ('model', SVC(class_weight='balanced', random_state=42))])
        SVM.fit(X_train, y_train)
        SVM_scores = score_model(SVM, X_test, y_test)

        #Isolation forest
        ISO = Pipeline(steps=[('transformer', transformer), ('model', IsolationForest(contamination=sum(y_train)/len(y_train), random_state=42))])
        ISO.fit(X_train, y_train)
        ISO_scores = score_model(ISO, X_test, y_test, map={1:0, -1:1})

        #Local Outlier Factor
        LOF = Pipeline(steps=[('transformer', transformer), ('model', LocalOutlierFactor(n_neighbors=10, novelty=True, contamination=sum(y_train)/len(y_train)))])
        LOF.fit(X_train, y_train)
        LOF_scores = score_model(LOF, X_test, y_test, map={1:0, -1:1})

        scores = {
            'Decision Tree': desc_tree_scores,
            'Random Forest': random_forest_scores,
            'K-NN': KNN_scores,
            'MLP': MLP_scores,
            'SVM': SVM_scores,
            'LOF': LOF_scores,
            'IF': ISO_scores
            }
        scores_df = pd.DataFrame(scores).T * 100
        scores_df.to_csv(f'{target}/{csv.replace("transactions", "scores")}')
