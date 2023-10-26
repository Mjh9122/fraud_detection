import argparse
import pandas as pd
import numpy as np

from os import listdir
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, precision_score, roc_auc_score, recall_score, average_precision_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def score_model(y_pred, y_true, map = None):
    scores = {}
    if map:
        y_pred = [map[i] for i in y_pred]
    evaluation_funcs = {'accuracy':accuracy_score, 'precision':precision_score, 'recall':recall_score, 'f1':f1_score, 'MCC':matthews_corrcoef}
    for name, score in evaluation_funcs.items():
        scores[name] = score(y_pred = y_pred, y_true = y_true)
    scores['ROC-AUC'] = roc_auc_score(y_score = y_pred, y_true = y_true)
    scores['AUPR'] = average_precision_score(y_score = y_pred, y_true = y_true)
    return scores


def DT(X_train, y_train, X_test, y_test, grid_search, verbose):
    
    if grid_search:
        params = {
            'criterion': ['gini', 'entropy'],
            'max_depth': range(1, 15),
            'min_samples_split': range(2, 10),
            'min_samples_leaf':range(1, 5)
        }
        dt_grid = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter = 100, scoring = 'f1', n_jobs=-1, random_state=42)
        dt_grid.fit(X_train, y_train)
        if verbose:
            print(dt_grid.best_params_)
        y_pred = dt_grid.predict(X_test)
    else:
        dt =  DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

    if verbose:
        print('Decision Tree')

    return score_model(y_pred, y_test)
        

def LOF(X_train, y_train, X_test, y_test, grid_search, verbose):
    if grid_search:
        max_avg, max_scores, max_n = 0, {}, 0
        for n in range(1, 501):
            lof = LocalOutlierFactor(n_neighbors=n, contamination=sum(y_train)/len(y_train))
            y_pred = lof.fit_predict(X_test)
            LOF_scores = score_model(y_pred, y_test, map={1:0, -1:1}) 
            if np.mean(list(LOF_scores.values())) > max_avg:
                max_avg = np.mean(list(LOF_scores.values()))
                max_scores = LOF_scores
                max_n = n
                if verbose:
                 print(f'New best params for LOF, num neighbors = {n}, avg score = {max_avg}')
            LOF_scores = max_scores
    else:
        lof = LocalOutlierFactor(n_neighbors=300, contamination=sum(y_train)/len(y_train))
        y_preds = lof.fit_predict(X_test)
        LOF_scores = score_model(y_pred = y_preds, y_true = y_test,  map={1:0, -1:1})
        
    if verbose:
        print('Local Outlier Factor Complete')

        return LOF_scores


def RF(X_train, y_train, X_test, y_test, grid_search, verbose):
    
    if grid_search:
        params = {
            'criterion': ['gini', 'entropy'],
            'max_depth': range(1, 10),
            'min_samples_split': range(2, 10),
            'min_samples_leaf':range(1, 5)
        }
        rf_grid = RandomizedSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), params, n_iter=100, scoring='f1', n_jobs=-1, random_state=42, verbose=int(verbose))
        rf_grid.fit(X_train, y_train)
        if verbose:
            print(rf_grid.best_params_)

        y_pred = rf_grid.predict(X_test)
    else:
        random_forest = RandomForestClassifier(random_state=42, n_jobs=-1)
        random_forest.fit(X_train, y_train)
        y_pred = random_forest.predict(X_test)

    if verbose:
        print('Random Forest')
    return  score_model(y_pred, y_test)


def KNN(X_train, y_train, X_test, y_test, grid_search, verbose):
    if grid_search:
        params = {
            'n_neighbors': range(1, 100),
            'weights': ['uniform', 'distance'],
            'p':[1, 2]
        }
        knn_grid = RandomizedSearchCV(KNeighborsClassifier(n_jobs=-1), params, scoring = 'f1', n_iter = 100, n_jobs = -1, random_state=42, verbose=int(verbose))
        knn_grid.fit(X_train, y_train)
        if verbose:
            print(knn_grid.best_params_)
        y_pred = knn_grid.predict(X_test)
    else:
        K_neighbors = KNeighborsClassifier(n_jobs=-1)
        K_neighbors.fit(X_train, y_train)
        y_pred = K_neighbors.predict(X_test)
    
    
    if verbose:
            print('KNN')
    return score_model(y_pred, y_test)

        
def MLP(X_train, y_train, X_test, y_test, grid_search, verbose):
    if grid_search:
        params = {
             'hidden_layer_sizes': [(100, ), (10, 10, 10), (20, 20, 20), (50, 50, 50)],
             'learning_rate': ['constant', 'invscaling', 'adaptive']
        }
        MLP_grid = RandomizedSearchCV(MLP(random_state = 42), params, scoring = 'f1', n_iter = 100, n_jobs = -1, random_state=42, verbose=int(verbose))
        MLP_grid.fit(X_train, y_train)
        if verbose:
            print(MLP_grid.best_params_)
        y_pred = MLP_grid.predict(X_test)
    else:
        percep =  MLPClassifier(hidden_layer_sizes=(15, 15, 15), random_state=42)
        percep.fit(X_train, y_train)
        y_pred = percep.predict(X_test)
    if verbose:
            print('MLP')
    return score_model(y_pred, y_test)

def SVM(X_train, y_train, X_test, y_test, grid_search, verbose):
    if grid_search:
        params = {
            'C':[2 ** k for k in range(-5, 5)],
            'gamma':['scale', 'auto'] + [2 ** k for k in range(-5, 5)],
            'class_weight': ['balanced', None]
        }
        svc_grid = RandomizedSearchCV(SVC(random_state=42), params, scoring = 'f1', n_iter = 100, n_jobs = -1, random_state=42, verbose=int(verbose))
        svc_grid.fit(X_test, y_train)
        if verbose:
            print(svc_grid.best_params_)
        y_pred = svc_grid.predict(X_test)
    else:
        support_vec = SVC(random_state=42)
        support_vec.fit(X_train, y_train)
        y_pred = support_vec.predict(X_test)
    if verbose:
        print('SVM')
    return score_model(y_pred, y_test)
    

def ISO(X_train, y_train, X_test, y_test, grid_search, verbose):
    if grid_search:
        params = {
            'n_estimators' : [10, 50, 100],
            'contamination' : ['auto', sum(y_train)/len(y_train)] + [2 ** k for k in range(-10, 0)]
        }
        iso_grid = RandomizedSearchCV(IsolationForest(random_state=42, n_jobs=-1), params, scoring = 'f1', n_iter = 100, n_jobs = -1, random_state=42, verbose=int(verbose))
        iso_grid.fit(X_train, y_train)
        if verbose: print(iso_grid.best_params_)
        y_pred = iso_grid.predict(X_test)
    else:
        iso_forest = IsolationForest(contamination=sum(y_train)/len(y_train), random_state=42, n_jobs=-1)
        y_pred = iso_forest.fit_predict(X_test)
    if verbose:
        print('Isolation Forest')
    return score_model(y_pred, y_test, map={1:0, -1:1})

        


def main(source, target, verbose, grid_search):
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
        X_train = pd.read_csv(f'{source}/{X_train_csv}').to_numpy()
        X_test = pd.read_csv(f'{source}/{X_test_csv}').to_numpy()

        
        if verbose:
            print(f'\n-----------------------\nRunning 7 Models on {X_train_csv[:-10]}\n-----------------------')


        dec_tree_scores = DT(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, grid_search=grid_search, verbose=verbose)

        random_forest_scores = RF(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, grid_search=grid_search, verbose=verbose)

        KNN_scores = KNN(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, grid_search=grid_search, verbose=verbose)
        
        MLP_scores = MLP(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, grid_search=grid_search, verbose=verbose)
        
        SVM_scores = SVM(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, grid_search=grid_search, verbose=verbose)

        ISO_scores = ISO(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, grid_search=grid_search, verbose=verbose)

        LOF_scores = LOF(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, grid_search=grid_search, verbose=verbose)
        

        scores = {
            'Decision Tree': dec_tree_scores,
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
    parser.add_argument('-g', '--grid_search', action = 'store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()
    main(args.source, args.target, args.verbose, args.grid_search)