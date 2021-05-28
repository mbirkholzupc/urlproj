"""
.. module:: ca_preprocess.py

*************

:Description: Script to preprocess the SEQUOIA 2000 California (regional) point dataset


:Authors: birkholz

:License: BSD 3 clause

:Version:

:Created on:

"""
import sys

sys.path.insert(1, '../kemlglearn/')

import argparse
import csv

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

# Output file path (62556 points)
PPFILE='ca_preprocessed.csv'

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
    ap.add_argument("-f", "--filename", required=True, help="path to input data file ca. Can be downloaded from: http://s2k-ftp.cs.berkeley.edu/sequoia/benchmark/point/ or https://github.com/mbirkholzupc/urldata/blob/main/ca")
    #ap.add_argument("-g", "--generate", help="type of data to generate on the fly")
    #ap.add_argument("-c", "--count", help="limit to count samples(file input only)")
    ap.add_argument("-s", "--seed", help="random seed (only applicable with count)")
    args=vars(ap.parse_args())

    # Set random seed, if specified. Otherwise, will default to rng with unspecified seed.
    if args['seed']:
        the_rng=np.random.default_rng(int(args['seed']))

    # Read in data file
    with open(args['filename'],newline='') as cafile:
        careader=csv.reader(cafile, delimiter=':')
        tmpval=0
        xvals=[]
        yvals=[]
        names=[]
        for row in careader:
            xvals.append(row[0])
            yvals.append(row[1])
            names.append(row[2])
            if len(row) != 3:
                print(row)
                break

        coords=np.array([xvals,yvals,names]).transpose()

        DROP_DUPS=True
        if DROP_DUPS:
            keep=[]
            n_prev=''
            for i,n in enumerate(names):
                if n!=n_prev:
                    keep.append(i)
                n_prev=n
            coords=coords[keep]

        # Another option: try averaging out duplicates
        # TODO

        # Save file
        with open(PPFILE, 'w', newline='') as ppfile:
            ppwriter=csv.writer(ppfile)
            for r in coords:
                row=r[0:2]
                ppwriter.writerow(row)

        print(coords[:20,:])
        print(coords[-20:,:])


    exit(0)
