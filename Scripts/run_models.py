import argparse
import pandas as pd
import numpy as np

from os import listdir
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, precision_score, roc_auc_score, recall_score, average_precision_score
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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


def main(source, target, verbose):
    # Read in data from source directory
    csvs = listdir(source)
    assert 'y_train.csv' in csvs
    y_train = pd.read_csv(f'{source}/y_train.csv', header=None).to_numpy().ravel()
    csvs.remove('y_train.csv')
    assert 'y_test.csv' in csvs
    y_test = pd.read_csv(f'{source}/y_test.csv', header=None).to_numpy().ravel()
    csvs.remove('y_test.csv')
    
    X_trains = sorted([f for f in csvs if 'train' in f])
    X_tests = sorted([f for f in csvs if 'test' in f])
    xs = list(zip(X_trains, X_tests))

    if verbose:
        print('\n-----------------------\ntrain/test files found:\n-----------------------')
        for train, test in xs:
            print(train[:-4], test[:-4])

    for X_train_csv, X_test_csv in xs:
        X_train = pd.read_csv(f'{source}/{X_train_csv}', header=None).to_numpy()
        X_test = pd.read_csv(f'{source}/{X_test_csv}', header=None).to_numpy()

        
        if verbose:
            print(f'\n-----------------------\nRunning 7 Models on {X_train_csv[:-10]}\n-----------------------')


        # Decision Tree
        decision_tree =  DecisionTreeClassifier(random_state=42)
        decision_tree.fit(X_train, y_train)
        desc_tree_scores = score_model(decision_tree, X_test, y_test)

        if verbose:
            print('Decision Tree')

        # Random Forest
        random_forest = RandomForestClassifier(random_state=42, n_jobs=-1)
        random_forest.fit(X_train, y_train)
        random_forest_scores = score_model(random_forest, X_test, y_test)
        
        if verbose:
            print('Random Forest')

        # KNN
        KNN = KNeighborsClassifier(n_jobs=-1)
        KNN.fit(X_train, y_train)
        KNN_scores = score_model(KNN, X_test, y_test)

        if verbose:
            print('K-NN')

        # Multilayer perceptron
        MLP =  MLPClassifier(hidden_layer_sizes=(15, 15, 15), random_state=42)
        MLP.fit(X_train, y_train)
        MLP_scores = score_model(MLP, X_test, y_test)

        if verbose:
            print('MLP')

        # Support Vector machine
        SVM = SVC(random_state=42)
        SVM.fit(X_train, y_train)
        SVM_scores = score_model(SVM, X_test, y_test)

        if verbose:
            print('SVM')

        # Isolation forest
        ISO = IsolationForest(contamination=sum(y_train)/len(y_train), random_state=42, n_jobs=-1)
        ISO.fit(X_train, y_train)
        ISO_scores = score_model(ISO, X_test, y_test, map={1:0, -1:1})

        if verbose:
            print('Isolation Forest')

        # Local Outlier Factor
        LOF = LocalOutlierFactor(n_neighbors=100, novelty=True, contamination=sum(y_train)/len(y_train))
        LOF.fit(X_train, y_train)
        LOF_scores = score_model(LOF, X_test, y_test, map={1:0, -1:1})

        if verbose:
            print('Local Outlier Factor')

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
        scores_df.to_csv(f'{target}/{X_train_csv.replace("train", "scores")}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()
    main(args.source, args.target, args.verbose)