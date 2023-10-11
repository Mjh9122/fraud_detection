import argparse
import pandas as pd
import seaborn as sns
import numpy as np


from matplotlib import pyplot as plt
from os import listdir

def plot_scores(scores, target, title):
    df = pd.concat(scores, axis = 0)
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    sns.barplot(data=df, x=df.index, y='Decision Tree', hue='hue', ax=axs[0, 0])
    sns.barplot(data=df, x=df.index, y='Random Forest', hue='hue', ax=axs[0, 1])
    sns.barplot(data=df, x=df.index, y='K-NN', hue='hue', ax=axs[0, 2])
    sns.barplot(data=df, x=df.index, y='MLP', hue='hue', ax=axs[0, 3])
    sns.barplot(data=df, x=df.index, y='SVM', hue='hue', ax=axs[1, 0])
    sns.barplot(data=df, x=df.index, y='LOF', hue='hue', ax=axs[1, 1])
    sns.barplot(data=df, x=df.index, y='IF', hue='hue', ax=axs[1, 2])
    for row in range(2):
        for col in range(4):
            axs[row, col].tick_params(axis = 'x', rotation = 30)
    plt.suptitle(title)
    plt.savefig(f'{target}/{title.replace(" ", "_")}.png')


def main(source, target, title):
    scores = []
    for score_file in listdir(source):
        score_df = pd.read_csv(f'{source}/{score_file}', index_col=0, header=0).T
        score_df['hue'] = score_file[:-11]
        scores.append(score_df)
    plot_scores(scores=scores, target = target, title = title)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generate_figures.py',
        description='turns a directory of score files into usable figures'
    )
    parser.add_argument('source_directory_path')
    parser.add_argument('target_directory_path')
    parser.add_argument('title')
    args = parser.parse_args()
    main(args.source_directory_path, args.target_directory_path, args.title)
