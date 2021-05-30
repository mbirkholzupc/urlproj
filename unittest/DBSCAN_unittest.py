"""
.. module:: DBSCAN_unittest

DBSCAN
*************

:Description: Unit tests for DBSCAN algorithm


:Authors: birkholz

:License: BSD 3 clause (due to using API of DBSCAN from scikit-learn)

:Version:

:Created on:

"""
import sys

sys.path.insert(1, '../')
sys.path.insert(1, '../kemlglearn/')

import numpy as np

import unittest
import time
import logging

from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
import pandas as pd

# Our implementation
from kemlglearn.cluster.DBSCAN import DBSCAN, NOISE, NO_CLUSTER

# Reference implementation for comparison
from sklearn.cluster import DBSCAN as sklDBSCAN

from kemlglearn.datasets.gen_dbscan_dataset import gen_dbscan_dataset1, gen_dbscan_dataset2, gen_dbscan_dataset3, \
                                        gen_dbscan_blobs, gen_dbscan_moons, gen_gridbscan_synth10k


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

def dbscan_equivalent_results(labels1, corepts1, labels2, corepts2):
    # This is not the absolute most efficient implementation, but it
    # doesn't have to be since it's just a unit test.

    # Confirm results are identical. They should have all the same core points, but
    # it is possible for a border point to be grouped with a different cluster if it's
    # close enough to two different clusters.

    # Noise must be identical
    noise1=set(np.where(labels1==NOISE)[0])
    noise2=set(np.where(labels2==NOISE)[0])

    if noise1 != noise2:
        return False

    # Core samples must be identical (although they might belong to different clusters)
    if set(corepts1) != set(corepts2):
        return False

    # Since the clusters may have different cluster IDs, we'll have to kind of guess which
    # clusters match.
    uniquelabels1 = np.unique(labels1)
    uniquelabels2 = np.unique(labels2)

    clusterdifferences = []

    for lbl1 in uniquelabels1:
        if lbl1 == -1:
            # Already checked noise, so continue with next cluster
            continue
        # Try to find most similar cluster
        set1=set(np.where(labels1==lbl1)[0])
        # Format of tuple: how many different samples there are, the list of different samples
        bestcluster = (None, None)
        for lbl2 in uniquelabels2:
            if lbl2 == -1:
                continue
            set2 = set(np.where(labels2==lbl2)[0])
            cluster_difference = set1 ^ set2
            mag_cluster_difference = len(cluster_difference)
            if bestcluster == (None,None):
                bestcluster = (mag_cluster_difference, cluster_difference)
            elif mag_cluster_difference < bestcluster[0]:
                bestcluster = (mag_cluster_difference, cluster_difference)

            # If they're identical, we can skip to the next iteration directly
            if mag_cluster_difference == 0:
                break

        if bestcluster[0] != None and bestcluster[0] > 0:
            clusterdifferences.append(bestcluster)

    if len(clusterdifferences) != 0:
        for difference in clusterdifferences:
            for pt in difference[1]:
                if pt in set(corepts1):
                    # If any core points are different, we're in trouble. Border
                    # points can be in different clusters though.
                    return False

    return True

def gen_shuffle_unshuffle_idx(data):
    shuffle_idx=np.random.permutation(len(data))
    revidx=[(x,y) for x,y in enumerate(shuffle_idx)]
    revidx=sorted(revidx, key=lambda x: x[1])
    unshuffle_idx=np.array([x[0] for x in revidx])
    return shuffle_idx, unshuffle_idx

class TestDBSCAN(unittest.TestCase):
    def test_db1(self):
        # 200 iterations generally seems to be enough to get an ambiguous border point
        for rs in range(200):
            X=gen_dbscan_dataset1(random_state=rs)

            # Shuffle the data to force (possibly) some border points to be different
            shuffle, unshuffle = gen_shuffle_unshuffle_idx(X)
            shuffleX = X[shuffle]
            mydbscan=DBSCAN(eps=43,min_samples=4).fit(shuffleX)
            # Unshuffle the results so they can be compared with original indices
            unshuffled_labels=mydbscan.labels_[unshuffle]
            unshuffled_corepts=np.array([shuffle[x] for x in mydbscan.core_sample_indices_])

            #plot_dbscan_results(shuffleX, mydbscan.labels_, mydbscan.core_sample_indices_)

            refdbscan=sklDBSCAN(eps=43,min_samples=4).fit(X)
            #plot_dbscan_results(X,refdbscan.labels_, refdbscan.core_sample_indices_)

            self.assertEqual(True,dbscan_equivalent_results(
                             unshuffled_labels, unshuffled_corepts,
                             refdbscan.labels_, refdbscan.core_sample_indices_))

    def test_db2(self):
        # 100 iterations generally seems to be enough to get an ambiguous border point
        for rs in range(100):
            X=gen_dbscan_dataset2(random_state=rs)

            # Shuffle the data to force (possibly) some border points to be different
            shuffle, unshuffle = gen_shuffle_unshuffle_idx(X)
            shuffleX = X[shuffle]
            mydbscan=DBSCAN(eps=36,min_samples=4).fit(shuffleX)
            # Unshuffle the results so they can be compared with original indices
            unshuffled_labels=mydbscan.labels_[unshuffle]
            unshuffled_corepts=np.array([shuffle[x] for x in mydbscan.core_sample_indices_])

            #plot_dbscan_results(shuffleX, mydbscan.labels_, mydbscan.core_sample_indices_)

            refdbscan=sklDBSCAN(eps=36,min_samples=4).fit(X)
            #plot_dbscan_results(X,refdbscan.labels_, refdbscan.core_sample_indices_)

            self.assertEqual(True,dbscan_equivalent_results(
                             unshuffled_labels, unshuffled_corepts,
                             refdbscan.labels_, refdbscan.core_sample_indices_))

    def test_db3(self):
        # 500 iterations generally seems to be enough to get an ambiguous border point
        for rs in range(500):
            X=gen_dbscan_dataset3(random_state=rs)

            # Shuffle the data to force (possibly) some border points to be different
            shuffle, unshuffle = gen_shuffle_unshuffle_idx(X)
            shuffleX = X[shuffle]
            mydbscan=DBSCAN(eps=40,min_samples=4).fit(shuffleX)
            # Unshuffle the results so they can be compared with original indices
            unshuffled_labels=mydbscan.labels_[unshuffle]
            unshuffled_corepts=np.array([shuffle[x] for x in mydbscan.core_sample_indices_])

            #plot_dbscan_results(shuffleX, mydbscan.labels_, mydbscan.core_sample_indices_)

            refdbscan=sklDBSCAN(eps=40,min_samples=4).fit(X)
            #plot_dbscan_results(X,refdbscan.labels_, refdbscan.core_sample_indices_)

            self.assertEqual(True,dbscan_equivalent_results(
                             unshuffled_labels, unshuffled_corepts,
                             refdbscan.labels_, refdbscan.core_sample_indices_))

    def test_4d(self):
        # Dataset from https://archive.ics.uci.edu/ml/datasets/banknote+authentication
        df=pd.read_csv('../data/banknotepreprocessed.csv', header=None)
        X=df.to_numpy()

        # Shuffle the data to force (possibly) some border points to be different
        shuffle, unshuffle = gen_shuffle_unshuffle_idx(X)
        shuffleX = X[shuffle]
        # Used DBSCAN_param_helper.py to pick out good parameters
        mydbscan=DBSCAN(eps=1.66,min_samples=8).fit(shuffleX)
        # Unshuffle the results so they can be compared with original indices
        unshuffled_labels=mydbscan.labels_[unshuffle]
        unshuffled_corepts=np.array([shuffle[x] for x in mydbscan.core_sample_indices_])

        refdbscan=sklDBSCAN(eps=1.66,min_samples=8).fit(X)

        self.assertEqual(True,dbscan_equivalent_results(
                         unshuffled_labels, unshuffled_corepts,
                         refdbscan.labels_, refdbscan.core_sample_indices_))

    def test_timing(self):
        our_timing = 0
        their_timing = 0
        iterations=10

        # For some reason, the first iteration is always much longer
        # than the rest (caching, other???). Let's discard that result
        first_iteration = True

        for rs in range(iterations):
            X=gen_dbscan_dataset1(random_state=rs)
            #X=gen_gridbscan_synth10k(random_state=rs)

            shuffle, unshuffle = gen_shuffle_unshuffle_idx(X)
            shuffleX = X[shuffle]
            start_time=time.time()
            mydbscan=DBSCAN(eps=43,min_samples=4).fit(shuffleX)
            end_time=time.time()
            if not first_iteration:
                our_timing += (end_time-start_time)

            start_time=time.time()
            refdbscan=sklDBSCAN(eps=43,min_samples=4).fit(shuffleX)
            end_time=time.time()
            if not first_iteration:
                their_timing += (end_time-start_time)
            first_iteration = False

        print('Timing results:')
        print(f'Ours:   {our_timing/(iterations-1)}')
        print(f'Theirs: {their_timing/(iterations-1)}')

# These aren't really "tests," but rather demonstrations
class TestDBSCANInteractive(unittest.TestCase):
    def test_one(self):
        n_samples = 4000
        n_blobs = 4
        X=gen_dbscan_blobs(n_samples, n_blobs, std=0.50, random_state=None)
        mydbscan=DBSCAN(eps=0.3,min_samples=4).fit(X)
        plot_dbscan_results(X, mydbscan.labels_, mydbscan.core_sample_indices_)

        X = gen_dbscan_moons(n_samples)
        mydbscan=DBSCAN(eps=0.085,min_samples=4).fit(X)
        plot_dbscan_results(X,mydbscan.labels_, mydbscan.core_sample_indices_)
        self.assertEqual(1,1)

    def test_two(self):
        n_samples = 400
        n_blobs = 4
        X, y_true = make_blobs(n_samples=n_samples,
                               centers=n_blobs,
                               cluster_std=0.60,
                               random_state=0)
        X = X[:, ::-1]

        # Show raw points
        plt.figure(1)
        plt.scatter(X[:,0], X[:,1])
        plt.show()

        mydbscan=DBSCAN(eps=0.5,min_samples=4).fit(X)

        # The following code is based on DBSCAN example in scikit-learn user guide
        plot_dbscan_results(X,mydbscan.labels_, mydbscan.core_sample_indices_)

        self.assertEqual(1,1)

    def test_db1(self):
        X=gen_dbscan_dataset1()
        mydbscan=DBSCAN(eps=43,min_samples=4).fit(X)
        plot_dbscan_results(X, mydbscan.labels_, mydbscan.core_sample_indices_)
        self.assertEqual(1,1)

    def test_db2(self):
        X=gen_dbscan_dataset2()
        mydbscan=DBSCAN(eps=36,min_samples=4).fit(X)
        plot_dbscan_results(X, mydbscan.labels_, mydbscan.core_sample_indices_)
        self.assertEqual(1,1)

    def test_db3(self):
        X=gen_dbscan_dataset3()
        mydbscan=DBSCAN(eps=40,min_samples=4).fit(X)
        plot_dbscan_results(X, mydbscan.labels_, mydbscan.core_sample_indices_)
        self.assertEqual(1,1)

if __name__ == '__main__':
    # Set up logging subsystem. Level should be one of: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    logging.basicConfig()
    # DEBUG is a good level for unit tests, but you can change to INFO if you want to shut it up
    # dbscan_logger=logging.get_logger('dbscan')
    # dbscan_logger.setLevel(logging.DEBUG)
    unittest.main()
