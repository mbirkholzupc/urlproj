"""
.. module:: GriDBSCAN_unittest

*************

:Description: GriDBSCAN unit test

:Authors: birkholz

:License: BSD 3 clause (due to using API of DBSCAN from scikit-learn)

:Version:

:Created on:

"""
import sys

sys.path.insert(1, '../')
sys.path.insert(1, '../kemlglearn/')

import numpy as np

import time
import unittest
import logging

from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
import pandas as pd

# Our implementation
from kemlglearn.cluster.DBSCAN import DBSCAN, NOISE, NO_CLUSTER

# Reference implementation for comparison
from sklearn.cluster import DBSCAN as sklDBSCAN

from kemlglearn.cluster.GriDBSCAN import GriDBSCAN

from kemlglearn.datasets.gen_dbscan_dataset import gen_dbscan_dataset1, gen_dbscan_dataset2, gen_dbscan_dataset3, \
                                        gen_dbscan_blobs, gen_dbscan_moons, gen_gridbscan_synth10k

from RunningStat import RunningStat

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
        print('noise mismatch!')
        return False

    # Core samples must be identical (although they might belong to different clusters)
    if set(corepts1) != set(corepts2):
        print('core sample mismatch!')
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

        xy = x[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=6)

        '''
        # Alternate plotting option to differentiate core/border points
        xy = x[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=14)

        xy = x[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
            markeredgecolor='k', markersize=6)
        '''


    # Show clusters (and noise)
    plt.title(f'Estimated number of clusters: {n_clusters_}')
    plt.show()

class TestGriDBSCAN(unittest.TestCase):
    def test_one(self):
        print('\nRunning test_one:')
        n_samples = 4000
        n_blobs = 4
        X, y_true = make_blobs(n_samples=n_samples,
                               centers=n_blobs,
                               cluster_std=0.60,
                               random_state=0)
        X = X[:, ::-1]

        plt.figure(1)
        plt.scatter(X[:,0], X[:,1])
        plt.show()

        X, y_true = make_moons(n_samples=n_samples, noise=0.1)
        plt.figure(1)
        plt.scatter(X[:,0], X[:,1], s=100)
        plt.show()
        print('Done with test_one.')
        self.assertEqual(1,1)

    def test_two(self):
        print('\nRunning test_two:')
        n_samples = 4000
        n_blobs = 4
        X, y_true = make_blobs(n_samples=n_samples,
                               centers=n_blobs,
                               cluster_std=0.60,
                               random_state=0)
        X = X[:, ::-1]

        mygridbscan=GriDBSCAN(eps=0.2,min_samples=10).fit(X)

        print('Done with test_two.')
        self.assertEqual(1,1)

    def test_minmax(self):
        print('\nRunning test_minmax:')
        n_samples = 10000
        n_blobs = 4
        X, y_true = make_blobs(n_samples=n_samples,
                               centers=n_blobs,
                               cluster_std=0.60,
                               random_state=0)
        X = X[:, ::-1]
        print(X)

        mygridbscan=GriDBSCAN(eps=0.2,min_samples=10).fit(X)
        mygridbscan=GriDBSCAN(eps=0.2,min_samples=10,grid=(2,2)).fit(X)
        mygridbscan=GriDBSCAN(eps=0.2,min_samples=10,grid=(6,2)).fit(X)

    def test_innerouter(self):
        print('\nRunning test_innerouter:')
        X = np.array( [ [  8, 8],
                        [2.1, 4],
                        [ -4, 3],
                        [  0, 0],
                        [  3,-2],
                        [  4, 1],
                        [  0, 4.1] ] )
        print(X)

        mygridbscan=GriDBSCAN(eps=0.2,min_samples=10,grid=(4,5)).fit(X)

    def test_blobs(self):
        print('\nRunning test_blobs:')
        n_samples = 1000
        n_blobs = 4
        X, y_true = make_blobs(n_samples=n_samples,
                               centers=n_blobs,
                               cluster_std=0.60,
                               random_state=2)
        X = X[:, ::-1]

        # Show points
        plt.figure(1)
        plt.scatter(X[:,0], X[:,1])
        plt.show()

        mygridbscan=GriDBSCAN(eps=0.5,min_samples=4,grid=(6,2)).fit(X)

        print('GriDBSCAN')
        print(mygridbscan.labels_)
        print(mygridbscan.core_sample_indices_)

        print('DBSCAN')
        refdbscan=sklDBSCAN(eps=0.5,min_samples=4).fit(X)
        print(refdbscan.labels_)
        print(refdbscan.core_sample_indices_)

class TestGriDBSCANAuto(unittest.TestCase):
    def test_auto1(self):
        our_timing = RunningStat()
        their_timing = RunningStat()

        # 100 iterations to average time
        for rs in range(1):
            n_samples = 20000
            n_blobs = 4
            X, y_true = make_blobs(n_samples=n_samples,
                                   centers=n_blobs,
                                   cluster_std=0.60,
                                   random_state=rs)
            X = X[:, ::-1]

            '''
            # Shuffle the data to force (possibly) some border points to be different
            shuffle, unshuffle = gen_shuffle_unshuffle_idx(X)
            shuffleX = X[shuffle]
            mydbscan=DBSCAN(eps=43,min_samples=4).fit(shuffleX)
            # Unshuffle the results so they can be compared with original indices
            unshuffled_labels=mydbscan.labels_[unshuffle]
            unshuffled_corepts=np.array([shuffle[x] for x in mydbscan.core_sample_indices_])
            mygridbscan=GriDBSCAN(eps=0.5,min_samples=4,grid=(3,4)).fit(shuffleX)
            '''
            print('ours...')
            start_time=time.time()
            mygridbscan=GriDBSCAN(eps=0.5,min_samples=4,grid=(5,5)).fit(X)
            end_time=time.time()
            our_timing.push(end_time-start_time)

            print('theirs...')
            start_time=time.time()
            refdbscan=sklDBSCAN(eps=0.5,min_samples=4,algorithm='brute',n_jobs=1).fit(X)
            #refdbscan=DBSCAN(eps=0.5,min_samples=4).fit(X)
            end_time=time.time()
            their_timing.push(end_time-start_time)
            print(mygridbscan.core_sample_indices_)
            print(refdbscan.core_sample_indices_)
            print(mygridbscan.labels_)
            print(refdbscan.labels_)

            uniquelabels1 = np.unique(mygridbscan.labels_)
            uniquelabels2 = np.unique(refdbscan.labels_)
            print(uniquelabels1)
            print(uniquelabels2)

            #plot_dbscan_results(X, mygridbscan.labels_, mygridbscan.core_sample_indices_)

            #plot_dbscan_results(X, refdbscan.labels_, refdbscan.core_sample_indices_)

            print(f'checking results for seed: {rs}')
            self.assertEqual(True,dbscan_equivalent_results(
                             #unshuffled_labels, unshuffled_corepts,
                             mygridbscan.labels_, mygridbscan.core_sample_indices_,
                             refdbscan.labels_, refdbscan.core_sample_indices_))
        print('Timing results:')
        print(f'Ours:   {our_timing.mean()} ({our_timing.standard_deviation()})')
        print(f'Theirs: {their_timing.mean()} ({their_timing.standard_deviation()})')

    def test_auto2(self):
        our_timing = RunningStat()
        their_timing = RunningStat()

        # 100 iterations to average time
        for rs in range(1):
            X = gen_gridbscan_synth10k(random_state=rs)

            '''
            # Shuffle the data to force (possibly) some border points to be different
            shuffle, unshuffle = gen_shuffle_unshuffle_idx(X)
            shuffleX = X[shuffle]
            mydbscan=DBSCAN(eps=43,min_samples=4).fit(shuffleX)
            # Unshuffle the results so they can be compared with original indices
            unshuffled_labels=mydbscan.labels_[unshuffle]
            unshuffled_corepts=np.array([shuffle[x] for x in mydbscan.core_sample_indices_])
            mygridbscan=GriDBSCAN(eps=0.5,min_samples=4,grid=(3,4)).fit(shuffleX)
            '''
            print('ours...')
            start_time=time.time()
            mygridbscan=GriDBSCAN(eps=0.5,min_samples=4,grid=(5,5)).fit(X)
            end_time=time.time()
            our_timing.push(end_time-start_time)

            print('theirs...')
            start_time=time.time()
            #refdbscan=sklDBSCAN(eps=0.5,min_samples=4,algorithm='brute',n_jobs=1).fit(X)
            #refdbscan=sklDBSCAN(eps=0.5,min_samples=4,n_jobs=1).fit(X)
            refdbscan=DBSCAN(eps=0.5,min_samples=4).fit(X)
            end_time=time.time()
            their_timing.push(end_time-start_time)
            print(mygridbscan.core_sample_indices_)
            print(refdbscan.core_sample_indices_)
            print(mygridbscan.labels_)
            print(refdbscan.labels_)

            uniquelabels1 = np.unique(mygridbscan.labels_)
            uniquelabels2 = np.unique(refdbscan.labels_)
            print(uniquelabels1)
            print(uniquelabels2)

            #plot_dbscan_results(X, mygridbscan.labels_, mygridbscan.core_sample_indices_)

            #plot_dbscan_results(X, refdbscan.labels_, refdbscan.core_sample_indices_)

            print(f'checking results for seed: {rs}')
            self.assertEqual(True,dbscan_equivalent_results(
                             #unshuffled_labels, unshuffled_corepts,
                             mygridbscan.labels_, mygridbscan.core_sample_indices_,
                             refdbscan.labels_, refdbscan.core_sample_indices_))
        print('Timing results:')
        print(f'Ours:   {our_timing.mean()} ({our_timing.standard_deviation()})')
        print(f'Theirs: {their_timing.mean()} ({their_timing.standard_deviation()})')


if __name__ == '__main__':
    # Set up logging subsystem. Level should be one of: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    logging.basicConfig()
    # DEBUG is a good level for unit tests, but you can change to INFO if you want to shut it up
    # dbscan_logger=logging.get_logger('dbscan')
    # dbscan_logger.setLevel(logging.DEBUG)
    unittest.main()
