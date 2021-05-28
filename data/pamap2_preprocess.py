"""
.. module:: pamap2_preprocess.py

*************

:Description: Script to preprocess the PAMAP2 dataset. In particular,
              the preprocessing is:
               - Deal with missing values due to sensors running at different sample rates
               - Deal with missing values due to dropped packets from wireless sensors
               - Reduce dimensionality using PCA (like DBSCAN Revisited, Revisited paper)

:Authors: birkholz

:License: BSD 3 clause

:Version:

:Created on:

"""
import argparse
import csv
import math

import pandas as pd
import numpy as np
from scipy.spatial import distance, distance_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from DBSCAN import DBSCAN

# Random-number generator
the_rng = np.random.default_rng()

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
    ap.add_argument("-f", "--filename", required=True, help="path to input data csv file")
    #ap.add_argument("-g", "--generate", help="type of data to generate on the fly")
    #ap.add_argument("-c", "--count", help="limit to count samples(file input only)")
    ap.add_argument("-s", "--seed", help="random seed (only applicable with count)")
    args=vars(ap.parse_args())

    # Set random seed, if specified. Otherwise, will default to rng with unspecified seed.
    if args['seed']:
        the_rng=np.random.default_rng(int(args['seed']))

    # Read in data file
    df = pd.read_csv(args['filename'], header=None, delimiter=' ')

    # Replace NaN measurements due to slow sensor update rate with last good value
    tmplist=[]
    last_val = float('nan')
    ctr=0
    for val in df[2]:
        if not math.isnan(val):
            last_val = val
        tmplist.append(last_val)
    df=df.drop(2,axis=1)
    df.insert(2,'2',tmplist)

    # Drop timestamp and activity identifier
    df=df.drop(0,axis=1)
    df=df.drop(1,axis=1)

    # Drop any rows still containing NaN measurements (due to dropped wireless packets, and possibly first
    # couple of rows before the first heartrate update)
    before_rows=df.shape[0]
    df=df.apply(pd.to_numeric, errors='coerce')
    df=df.dropna()
    df=df.reset_index(drop=True)
    after_rows=df.shape[0]
    percent_dropped=100*(before_rows-after_rows)/before_rows
    print(f'Dropped {percent_dropped:1.2f}% of data')
    print(df)

    df.to_csv('tmpfile.csv', header=False, index=False)

    # Normalize each feature to range [0, 1e5]
    nparr=df.to_numpy()
    scaler=MinMaxScaler((0,1e5))
    scaler.fit(nparr)
    nparr=scaler.transform(nparr)

    # Do PCA to reduce dimensions
    pca = PCA(n_components=4)
    pca.fit(nparr)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    nparrpca=pca.transform(nparr)

    dfpca=pd.DataFrame(nparrpca)
    dfpca.to_csv('tmpfilepca.csv', header=False, index=False)

    exit(0)
