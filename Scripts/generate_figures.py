import argparse
import pandas as pd
import seaborn as sns
import numpy as np


from matplotlib import pyplot as plt
from os import listdir

def avg_diff_table(scores_dict, comparison, target):
    mean_changes = dict()
    for score in [s for s in scores_dict.keys() if s != comparison]:
        mean_changes[score] = (scores_dict[score] - scores_dict[comparison]).mean(axis=1)

    score_changes = pd.concat(list(mean_changes.values()), axis=1)
    score_changes.columns = list(mean_changes.keys())
    plt.axis('off')
    plt.table(cellText = score_changes.values, rowLabels = score_changes.index, colLabels = score_changes.columns, loc='center')
    plt.title('Score increase average in percent')
    plt.savefig(f'{target}/mean_score_change.png')
    return score_changes

def plot_scores(first, second, target, change):
    df = pd.concat([first, second], axis = 0)
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    sns.barplot(data=df, x=df.index, y='Decision Tree', hue='hue', ax=axs[0, 0])
    sns.barplot(data=df, x=df.index, y='Random Forest', hue='hue', ax=axs[0, 1])
    sns.barplot(data=df, x=df.index, y='K-NN', hue='hue', ax=axs[0, 2])
    sns.barplot(data=df, x=df.index, y='MLP', hue='hue', ax=axs[0, 3])
    sns.barplot(data=df, x=df.index, y='SVM', hue='hue', ax=axs[1, 0])
    sns.barplot(data=df, x=df.index, y='LOF', hue='hue', ax=axs[1, 1])
    sns.barplot(data=df, x=df.index, y='IF', hue='hue', ax=axs[1, 2])
    axs[1, 3].axis('off')
    axs[1, 3].table(cellText = np.array([change.values]).T, rowLabels = change.index, colLabels = ['Percent Change'], loc='center', cellLoc='center', colWidths=[.5])
    plt.title('Average percent score increase')
    plt.suptitle(f'{second["hue"][0]}')
    plt.savefig(f'{target}/{second["hue"][0]}.png')


def comparison_plots(scores_dict, comparison, target, diff_table):
    for feature, scores in scores_dict.items():
        scores['hue'] = feature
    for feature, scores in ((f, s) for (f, s) in scores_dict.items() if f != comparison):
        plot_scores(scores_dict[comparison], scores_dict[feature], target, diff_table[feature])

def main(source, target, comparison = 'none'):
    scores = dict()
    for score_file in listdir(source):
        scores[score_file[7:-4]] = pd.read_csv(f'{source}/{score_file}', index_col=0, header=0).T
    diff_table = avg_diff_table(scores, comparison, target)
    comparison_plots(scores, comparison, target, diff_table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generate_figures.py',
        description='turns a directory of score files into usable figures'
    )
    parser.add_argument('source_directory_path')
    parser.add_argument('target_directory_path')
    parser.add_argument('-c', '--comparison_feature_set', default='none')
    args = parser.parse_args()
    main(args.source_directory_path, args.target_directory_path, args.comparison_feature_set)
