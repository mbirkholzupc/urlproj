"""
.. module:: pamap2_preprocess.py

*************

:Description: Script to preprocess the banknote authentication dataset
              the preprocessing is:
               - Drop class column

:Authors: birkholz

:License: BSD 3 clause

:Version:

:Created on:

"""
import sys

sys.path.insert(1, '../kemlglearn/')

import argparse
import csv
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


from kemlglearn.cluster.DBSCAN import DBSCAN

__author__ = 'birkholz'

# Helper function to plot DBSCAN results based on DBSCAN example in scikit-learn user guide
# If data has more than two dimensions, the first two will be used
def plot_dbscan_results(x, labels, core_sample_indices):
    core_samples_mask=np.zeros_like(labels, dtype=bool)
    core_samples_mask[core_sample_indices] = True
    n_clusters_=len(set(labels))-(1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1] # Black for noise

        class_member_mask = (labels == k)

        xy = x[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=14)

        xy = x[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=6)

    # Show clusters (and noise)
    plt.title(f'Estimated number of clusters: {n_clusters_}')
    plt.show()

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", required=True, help="path to input data file data_banknote_authentication.txt. Can be downloaded from: https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt")
    args=vars(ap.parse_args())

    # Read in data file
    df = pd.read_csv(args['filename'], header=None)
    print(df)

    # Drop date and time columns
    df=df.drop(4,axis=1)
    print(df)

    # Normalize each feature to range [0, 1e5]
    #nparr=df.to_numpy()
    #scaler=MinMaxScaler((0,1e5))
    #scaler.fit(nparr)
    #nparr=scaler.transform(nparr)
    #df=pd.DataFrame(nparr)

    #df.to_csv('banknotepreprocessed.csv', header=False, index=False, float_format='%1.4f')
    df.to_csv('banknotepreprocessed.csv', header=False, index=False)

    exit(0)
