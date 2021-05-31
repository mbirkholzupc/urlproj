"""
.. module:: DBSCAN_param_helper

DBSCAN
*************

:Description: Utility that simply runs DBSCAN when provided with dataset, MinPts (k) and Epsilon


:Authors: birkholz

:License: BSD 3 clause

:Version:

:Created on:

"""
import sys

sys.path.insert(1, './kemlglearn/')


import argparse

import pandas as pd
import numpy as np
from scipy.spatial import distance, distance_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons


from kemlglearn.datasets.gen_dbscan_dataset import gen_dbscan_dataset1, gen_dbscan_dataset2, gen_dbscan_dataset3, \
                               gen_dbscan_blobs, gen_dbscan_moons
from kemlglearn.cluster.DBSCAN import DBSCAN

# Random-number generator
the_rng = np.random.default_rng()

__author__ = 'birkholz'

# Helper function to plot DBSCAN results based on DBSCAN example in scikit-learn user guide
# If data has more than two dimensions, the first two will be used
def plot_dbscan_results(x, labels, core_sample_indices,small_points=False):
    core_samples_mask=np.zeros_like(labels, dtype=bool)
    core_samples_mask[core_sample_indices] = True
    n_clusters_=len(set(labels))-(1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            #col = [0, 0, 0, 1] # Black for noise
            col = [0, 0, 0, 0.1] # Black for noise, but set alpha to 0.1 so it doesn't block clusters

        class_member_mask = (labels == k)

        if small_points:
            xy = x[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor=tuple(col), markersize=2)

        else:
            xy = x[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=12)

            xy = x[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

    # Show clusters (and noise)
    plt.title(f'Estimated number of clusters: {n_clusters_}')
    plt.show()

datagen_patterns = [ 'dbscan1', 'dbscan2', 'dbscan3', 'blobs', 'moons' ]

def gen_data(pattern):
    if pattern == 'dbscan1':
        data = gen_dbscan_dataset1()
    elif pattern == 'dbscan2':
        data = gen_dbscan_dataset2()
    elif pattern == 'dbscan3':
        data = gen_dbscan_dataset3()
    elif pattern == 'blobs':
        data = gen_dbscan_blobs(2000, 4, std=0.50, random_state=None)
    elif pattern == 'moons':
        data = gen_dbscan_moons(4000)
    return data

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", help="path to input data csv file")
    ap.add_argument("-g", "--generate", help="type of data to generate on the fly")
    ap.add_argument("-c", "--count", help="limit to count samples(file input only)")
    ap.add_argument("-k", "--kdist", required=True, help="k-dist (MinPts)")
    ap.add_argument("-e", "--eps", required=True, help="epsilon")
    ap.add_argument("-s", "--seed", help="random seed (only applicable with count)")
    ap.add_argument("-m", "--smallpoints", action='store_true', help="plot small points, no difference between core/border")
    args=vars(ap.parse_args())

    # Set random seed, if specified. Otherwise, will default to rng with unspecified seed.
    if args['seed']:
        the_rng=np.random.default_rng(int(args['seed']))

    if args['generate']:
        if args['generate'] in datagen_patterns:
            X=gen_data(args['generate'])
        else:
            raise Exception('Try again. Could not understand requested pattern.')
    elif args['filename']:
        df=pd.read_csv(args['filename'],header=None)
        X=df.to_numpy()

        if args['count']:
            X=the_rng.choice(X,size=int(args['count']),replace=False)
    else:
        raise Exception('Need to specify filename or type of data to generate')

    # Set MinPts (k) and Epsilon
    k=int(args['kdist'])
    eps=float(args['eps'])

    # Run DBSCAN and show results
    mydbscan = DBSCAN(eps=eps,min_samples=k).fit(X)
    plot_dbscan_results(X, mydbscan.labels_, mydbscan.core_sample_indices_,small_points=args['smallpoints'])
