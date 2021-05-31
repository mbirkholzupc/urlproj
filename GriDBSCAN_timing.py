"""
.. module:: GriDBSCAN_timing

*************

:Description: Module to run some timing tests on GriDBSCAN

:Authors: birkholz

:License: BSD 3 clause

:Version:

:Created on:

"""
import sys

sys.path.insert(1, './kemlglearn/')

import numpy as np

import time
import argparse

import matplotlib.pyplot as plt
import pandas as pd

from profilehooks import profile

# Our implementation
from kemlglearn.cluster.GriDBSCAN import GriDBSCAN
from kemlglearn.cluster.DBSCAN import DBSCAN, NOISE, NO_CLUSTER

# Reference implementation for comparison
from sklearn.cluster import DBSCAN as sklDBSCAN

from kemlglearn.datasets.gen_dbscan_dataset import gen_dbscan_dataset1, gen_dbscan_dataset2, gen_dbscan_dataset3, \
                                                   gen_dbscan_blobs, gen_dbscan_moons, gen_gridbscan_synth10k

from RunningStat import RunningStat

# Random-number generator
the_rng = np.random.default_rng()

def gen_shuffle_unshuffle_idx(data):
    shuffle_idx=np.random.permutation(len(data))
    revidx=[(x,y) for x,y in enumerate(shuffle_idx)]
    revidx=sorted(revidx, key=lambda x: x[1])
    unshuffle_idx=np.array([x[0] for x in revidx])
    return shuffle_idx, unshuffle_idx

def max_grid_size(X,eps):
    mins=[]
    maxs=[]
    for i in range(len(X[0])):
        mins.append(min(X[:,i]))
        maxs.append(max(X[:,i]))
    mins=np.array(mins)
    maxs=np.array(maxs)

    dimrange=[themax-themin for themax, themin in zip(maxs,mins)]
    dimrange=np.array(dimrange)

    return tuple(np.trunc((dimrange/(2*eps))).astype(int))
    

def test1(dataset,eps,minpts,use_grid=True):
    """
    This test is for testing small synthetic datasets
    """
    X=dataset()
    shuffle, _ = gen_shuffle_unshuffle_idx(X)
    X=X[shuffle]

    if use_grid:
        max_grid=max_grid_size(X,eps)
        max_grid_square=min(max_grid)
        max_grid_square=min(max_grid_square,30)
        for i in range(1,max_grid_square+1):
            grid=tuple([i]*len(max_grid))
            start_time=time.time()
            #gridbscan=GriDBSCAN(eps=eps,min_samples=minpts,grid=grid,algorithm='brute',n_jobs=1).fit(X)
            gridbscan=GriDBSCAN(eps=eps,min_samples=minpts,grid=grid).fit(X)
            end_time=time.time()
            print(f'{i}: {(end_time-start_time)}')
            print(f'     {gridbscan.dbscan_total_time_}')
    else:
        start_time=time.time()
        #dbscan=DBSCAN(eps=eps,min_samples=minpts,algorithm='brute',n_jobs=1).fit(X)
        dbscan=DBSCAN(eps=eps,min_samples=minpts).fit(X)
        end_time=time.time()
        print(f'DBSCAN: {(end_time-start_time)}')

    return

def test2(dataset,eps,minpts,use_grid=True):
    """
    This test is for reading datasets from a file
    """
    X=dataset
    shuffle, _ = gen_shuffle_unshuffle_idx(X)
    X=X[shuffle]

    if use_grid:
        max_grid=max_grid_size(X,eps)
        max_grid_square=min(max_grid)
        max_grid_square=min(max_grid_square,30)
        for i in range(1,max_grid_square+1):
            grid=tuple([i]*len(max_grid))
            start_time=time.time()
            #gridbscan=GriDBSCAN(eps=eps,min_samples=minpts,grid=grid,algorithm='brute').fit(X)
            gridbscan=GriDBSCAN(eps=eps,min_samples=minpts,grid=grid).fit(X)
            end_time=time.time()
            print(f'{i}: {(end_time-start_time)}')
            print(f'     {gridbscan.dbscan_total_time_}')
    else:
        start_time=time.time()
        #dbscan=DBSCAN(eps=eps,min_samples=minpts,algorithm='brute',n_jobs=1).fit(X)
        dbscan=DBSCAN(eps=eps,min_samples=minpts).fit(X)
        end_time=time.time()
        print(f'DBSCAN: {(end_time-start_time)}')

    return

def test_gridbscan_sequoia(dataset):
    """
    This routine performs a series of tests on the CA SEQUOIA 2000 dataset using
    DBSCAN and GriDBSCAN
    """
    Xfull=dataset

    mins=[]
    maxs=[]
    for i in range(len(Xfull[0])):
        mins.append(min(Xfull[:,i]))
        maxs.append(max(Xfull[:,i]))
    mins=np.array(mins)
    maxs=np.array(maxs)

    dimrange=[themax-themin for themax, themin in zip(maxs,mins)]
    dimrange=np.array(dimrange)

    print(dimrange)

    #X=the_rng.choice(X,size=int(args['count']),replace=False)

    """
    shuffle, _ = gen_shuffle_unshuffle_idx(X)
    X=X[shuffle]

    if use_grid:
        max_grid=max_grid_size(X,eps)
        max_grid_square=min(max_grid)
        max_grid_square=min(max_grid_square,30)
        for i in range(1,max_grid_square+1):
            grid=tuple([i]*len(max_grid))
            start_time=time.time()
            #gridbscan=GriDBSCAN(eps=eps,min_samples=minpts,grid=grid,algorithm='brute').fit(X)
            gridbscan=GriDBSCAN(eps=eps,min_samples=minpts,grid=grid).fit(X)
            end_time=time.time()
            print(f'{i}: {(end_time-start_time)}')
            print(f'     {gridbscan.dbscan_total_time_}')
    else:
        start_time=time.time()
        #dbscan=DBSCAN(eps=eps,min_samples=minpts,algorithm='brute',n_jobs=1).fit(X)
        dbscan=DBSCAN(eps=eps,min_samples=minpts).fit(X)
        end_time=time.time()
        print(f'DBSCAN: {(end_time-start_time)}')
    """

    return


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument("-t", "--test", required=True, help="name of test to run")
    ap.add_argument("-f", "--filename", help="path to data file")
    ap.add_argument("-c", "--count", help="count of samples to read from data file")
    ap.add_argument("-s", "--seed", help="random seed (only applicable with count)")
    ap.add_argument("-k", "--kdist", help="k-dist (MinPts)")
    ap.add_argument("-e", "--eps", help="epsilon")
    args=vars(ap.parse_args())

    # Set random seed, if specified. Otherwise, will default to rng with unspecified seed.
    if args['seed']:
        the_rng=np.random.default_rng(int(args['seed']))

    # If filename specified, read in file even if the test doesn't use it.
    if args['filename']:
        df=pd.read_csv(args['filename'],header=None)
        X=df.to_numpy()

        if args['count']:
            X=the_rng.choice(X,size=int(args['count']),replace=False)

        # Set MinPts (k) and Epsilon
        k=int(args['kdist'])
        eps=float(args['eps'])

    testid=args['test']
    if testid=='1':
        test1(gen_dbscan_dataset1,eps=43,minpts=4,use_grid=True)
        test1(gen_dbscan_dataset1,eps=43,minpts=4,use_grid=False)
    elif testid=='2':
        test1(gen_dbscan_dataset2,eps=36,minpts=4,use_grid=True)
        test1(gen_dbscan_dataset2,eps=36,minpts=4,use_grid=False)
    elif testid=='3':
        test1(gen_dbscan_dataset3,eps=40,minpts=4,use_grid=True)
        test1(gen_dbscan_dataset3,eps=40,minpts=4,use_grid=False)
    elif testid=='4':
        test1(gen_gridbscan_synth10k,eps=10,minpts=4,use_grid=True)
        test1(gen_gridbscan_synth10k,eps=10,minpts=4,use_grid=False)
    elif testid=='5':
        test2(X,eps=eps,minpts=k,use_grid=True)
        test2(X,eps=eps,minpts=k,use_grid=False)
    elif testid=='sequoia':
        test_gridbscan_sequoia(X)
    else:
        raise Exception('Unrecognized test: ' + testid)

    exit(0)
