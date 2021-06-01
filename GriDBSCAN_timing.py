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


    num_samples = [9000, 10400, 12500]
    eps=5000         # Approximate epsilon calculated from Figure 11 in GriDBSCAN paper
    minpts=4         # Standard 2-D value
    max_grid_idx=27  # Calculate up to 26x26
    timing_actual_results = {}
    timing_theoretical_results = {}

    for n in num_samples:
        # Create smaller dataset
        X=the_rng.choice(Xfull,size=n,replace=False)
        timing_theoretical_results[n] = []
        timing_actual_results[n] = []

        # Calculate baseline DBSCAN result
        start_time=time.time()
        #dbscan=DBSCAN(eps=eps,min_samples=minpts,algorithm='brute',n_jobs=1).fit(X)
        dbscan=DBSCAN(eps=eps,min_samples=minpts).fit(X)
        end_time=time.time()
        dbscan_elapsed=end_time-start_time
        print(f'baseline: {dbscan_elapsed}')

        # Calculate results for various grid sizes
        for i in range(1,max_grid_idx):
            grid=(i,i,)
            start_time=time.time()
            #gridbscan=GriDBSCAN(eps=eps,min_samples=minpts,grid=grid,algorithm='brute',n_jobs=1).fit(X)
            gridbscan=GriDBSCAN(eps=eps,min_samples=minpts,grid=grid).fit(X)
            end_time=time.time()
            gridbscan_elapsed=end_time-start_time
            theoretical_improvement=dbscan_elapsed/gridbscan.dbscan_total_time_
            actual_improvement=dbscan_elapsed/gridbscan_elapsed
            print(f'{i:02d}: {gridbscan_elapsed:1.4f} {gridbscan.dbscan_total_time_:1.4f}')
            print(f'\t\t theoretical improvement: {theoretical_improvement:02.4f}')
            print(f'\t\t      actual improvement: {actual_improvement:02.4f}')
            timing_theoretical_results[n].append(theoretical_improvement)
            timing_actual_results[n].append(actual_improvement)

    x_coord=[idx**2 for idx in range(1,max_grid_idx)]
    lines=['sb-','sm-','sy-']
    legendlines=[]
    legendlabels=[]
    for i,n in enumerate(num_samples):
        tmp, = plt.plot(x_coord,timing_theoretical_results[n],lines[i])
        legendlines.append(tmp)
        legendlabels.append('dataset ' + str(n))
    plt.legend(legendlines,legendlabels)
    plt.show()

    legendlines=[]
    legendlabels=[]
    for i,n in enumerate(num_samples):
        tmp, = plt.plot(x_coord,timing_actual_results[n],lines[i])
        legendlines.append(tmp)
        legendlabels.append('dataset ' + str(n))
    plt.legend(legendlines,legendlabels)
    plt.show()

    return

@profile
def test_dbscan_profile(dataset,k,eps):
    """
    This routine calls DBSCAN and is profiled
    """
    # Calculate baseline DBSCAN result
    X=dataset
    dbscan=DBSCAN(eps=eps,min_samples=k).fit(X)

    # Nothing else to do: the profile decorator will take care of the rest
    return

@profile
def test_gridbscan_profile(dataset,k,eps,grid=(5,5)):
    """
    This routine calls GriDBSCAN and is profiled
    """
    # Calculate baseline GriDBSCAN result
    X=dataset
    dbscan=GriDBSCAN(eps=eps,min_samples=k,grid=grid).fit(X)

    # Nothing else to do: the profile decorator will take care of the rest
    return

def test_pamap2_varyn(dataset):
    """
    This routine performs a series of tests on the PAMAP2 dataset varying number of samples
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

    #minpts = [10, 50, 100]
    #eps=[500, 1000, 2000]
    num_samples = [1000, 10000, 100000, 1000000]
    minpts = 10
    eps=500
    grid=(4,4,4,4)
    timing_actual_results = []
    timing_theoretical_results = []

    for n in num_samples:
        # Create smaller dataset
        X=the_rng.choice(Xfull,size=n,replace=False)

        # Calculate baseline DBSCAN result
        start_time=time.time()
        dbscan=DBSCAN(eps=eps,min_samples=minpts).fit(X)
        end_time=time.time()
        dbscan_elapsed=end_time-start_time
        print(f'baseline: {dbscan_elapsed}')

        # Calculate results for GriDBSCAN
        start_time=time.time()
        gridbscan=GriDBSCAN(eps=eps,min_samples=minpts,grid=grid).fit(X)
        end_time=time.time()
        gridbscan_elapsed=end_time-start_time
        theoretical_improvement=dbscan_elapsed/gridbscan.dbscan_total_time_
        actual_improvement=dbscan_elapsed/gridbscan_elapsed
        print(f'{i:02d}: {gridbscan_elapsed:1.4f} {gridbscan.dbscan_total_time_:1.4f}')
        print(f'\t\t theoretical improvement: {theoretical_improvement:02.4f}')
        print(f'\t\t      actual improvement: {actual_improvement:02.4f}')
        timing_theoretical_results.append(theoretical_improvement)
        timing_actual_results.append(actual_improvement)

    x_coord=[n for n in num_samples]
    style=['bo']
    tmp, = plt.plot(x_coord,timing_theoretical_results, 'bo-')
    stylelines=[tmp]
    plt.title('Theoretical Improvement')
    #plt.legend(stylelines,['Theoretical Improvement'])
    plt.show()

    x_coord=[n for n in num_samples]
    tmp, = plt.plot(x_coord,timing_actual_results, 'bo-')
    stylelines=[tmp]
    plt.title('Actual Improvement')
    #plt.legend(stylelines,['Actual Improvement'])
    plt.show()

    return

def test_pamap2_varyk(dataset):
    """
    This routine performs a series of tests on the PAMAP2 dataset varying min points
    """
    #eps=[500, 1000, 2000]
    num_samples = 100000
    minpts = [10, 50, 100]
    eps=500
    grid=(4,4,4,4)

    timing_actual_results = []
    timing_theoretical_results = []

    Xfull=dataset
    # Create smaller dataset
    X=the_rng.choice(Xfull,size=num_samples,replace=False)

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

    for k in minpts:
        # Calculate baseline DBSCAN result
        start_time=time.time()
        dbscan=DBSCAN(eps=eps,min_samples=k).fit(X)
        end_time=time.time()
        dbscan_elapsed=end_time-start_time
        print(f'baseline: {dbscan_elapsed}')

        # Calculate results for GriDBSCAN
        start_time=time.time()
        gridbscan=GriDBSCAN(eps=eps,min_samples=k,grid=grid).fit(X)
        end_time=time.time()
        gridbscan_elapsed=end_time-start_time
        theoretical_improvement=dbscan_elapsed/gridbscan.dbscan_total_time_
        actual_improvement=dbscan_elapsed/gridbscan_elapsed
        print(f'{i:02d}: {gridbscan_elapsed:1.4f} {gridbscan.dbscan_total_time_:1.4f}')
        print(f'\t\t theoretical improvement: {theoretical_improvement:02.4f}')
        print(f'\t\t      actual improvement: {actual_improvement:02.4f}')
        timing_theoretical_results.append(theoretical_improvement)
        timing_actual_results.append(actual_improvement)

    x_coord=[k for k in minpts]
    style=['bo']
    tmp, = plt.plot(x_coord,timing_theoretical_results, 'bo-')
    stylelines=[tmp]
    plt.title('Theoretical Improvement')
    plt.show()

    x_coord=[k for k in minpts]
    tmp, = plt.plot(x_coord,timing_actual_results, 'bo-')
    stylelines=[tmp]
    plt.title('Actual Improvement')
    plt.show()

    return

def test_pamap2_varyeps(dataset):
    """
    This routine performs a series of tests on the PAMAP2 dataset varying epsilon
    """
    eps=[500, 1000, 2000]
    num_samples = 100000
    minpts = 50
    grid=(4,4,4,4)

    timing_actual_results = []
    timing_theoretical_results = []

    Xfull=dataset
    # Create smaller dataset
    X=the_rng.choice(Xfull,size=num_samples,replace=False)

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

    for e in eps:
        # Calculate baseline DBSCAN result
        start_time=time.time()
        dbscan=DBSCAN(eps=e,min_samples=minpts).fit(X)
        end_time=time.time()
        dbscan_elapsed=end_time-start_time
        print(f'baseline: {dbscan_elapsed}')

        # Calculate results for GriDBSCAN
        start_time=time.time()
        gridbscan=GriDBSCAN(eps=e,min_samples=minpts,grid=grid).fit(X)
        end_time=time.time()
        gridbscan_elapsed=end_time-start_time
        theoretical_improvement=dbscan_elapsed/gridbscan.dbscan_total_time_
        actual_improvement=dbscan_elapsed/gridbscan_elapsed
        print(f'{i:02d}: {gridbscan_elapsed:1.4f} {gridbscan.dbscan_total_time_:1.4f}')
        print(f'\t\t theoretical improvement: {theoretical_improvement:02.4f}')
        print(f'\t\t      actual improvement: {actual_improvement:02.4f}')
        timing_theoretical_results.append(theoretical_improvement)
        timing_actual_results.append(actual_improvement)

    x_coord=[e for e in eps]
    style=['bo']
    tmp, = plt.plot(x_coord,timing_theoretical_results, 'bo-')
    stylelines=[tmp]
    plt.title('Theoretical Improvement')
    plt.show()

    x_coord=[e for e in eps]
    tmp, = plt.plot(x_coord,timing_actual_results, 'bo-')
    stylelines=[tmp]
    plt.title('Actual Improvement')
    plt.show()

    return

def test_household_varyn(dataset):
    """
    This routine performs a series of tests on the Household dataset varying number of samples
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

    #minpts = [10, 50, 100]
    #eps=[500, 1000, 2000]
    num_samples = [1000, 10000, 100000, 1000000]
    minpts = 10
    eps=500
    grid=(3,3,3,3,3,3,3)
    timing_actual_results = []
    timing_theoretical_results = []

    for n in num_samples:
        # Create smaller dataset
        X=the_rng.choice(Xfull,size=n,replace=False)

        # Calculate baseline DBSCAN result
        start_time=time.time()
        dbscan=DBSCAN(eps=eps,min_samples=minpts).fit(X)
        end_time=time.time()
        dbscan_elapsed=end_time-start_time
        print(f'baseline: {dbscan_elapsed}')

        # Calculate results for GriDBSCAN
        start_time=time.time()
        gridbscan=GriDBSCAN(eps=eps,min_samples=minpts,grid=grid).fit(X)
        end_time=time.time()
        gridbscan_elapsed=end_time-start_time
        theoretical_improvement=dbscan_elapsed/gridbscan.dbscan_total_time_
        actual_improvement=dbscan_elapsed/gridbscan_elapsed
        print(f'{i:02d}: {gridbscan_elapsed:1.4f} {gridbscan.dbscan_total_time_:1.4f}')
        print(f'\t\t theoretical improvement: {theoretical_improvement:02.4f}')
        print(f'\t\t      actual improvement: {actual_improvement:02.4f}')
        timing_theoretical_results.append(theoretical_improvement)
        timing_actual_results.append(actual_improvement)

    x_coord=[n for n in num_samples]
    style=['bo']
    tmp, = plt.plot(x_coord,timing_theoretical_results, 'bo-')
    stylelines=[tmp]
    plt.title('Theoretical Improvement')
    #plt.legend(stylelines,['Theoretical Improvement'])
    plt.show()

    x_coord=[n for n in num_samples]
    tmp, = plt.plot(x_coord,timing_actual_results, 'bo-')
    stylelines=[tmp]
    plt.title('Actual Improvement')
    #plt.legend(stylelines,['Actual Improvement'])
    plt.show()

    return

def test_household_varyk(dataset):
    """
    This routine performs a series of tests on the Household dataset varying min points
    """
    #eps=[500, 1000, 2000]
    num_samples = 100000
    minpts = [10, 50, 100]
    eps=500
    grid=(3,3,3,3,3,3,3)

    timing_actual_results = []
    timing_theoretical_results = []

    Xfull=dataset
    # Create smaller dataset
    X=the_rng.choice(Xfull,size=num_samples,replace=False)

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

    for k in minpts:
        # Calculate baseline DBSCAN result
        start_time=time.time()
        dbscan=DBSCAN(eps=eps,min_samples=k).fit(X)
        end_time=time.time()
        dbscan_elapsed=end_time-start_time
        print(f'baseline: {dbscan_elapsed}')

        # Calculate results for GriDBSCAN
        start_time=time.time()
        gridbscan=GriDBSCAN(eps=eps,min_samples=k,grid=grid).fit(X)
        end_time=time.time()
        gridbscan_elapsed=end_time-start_time
        theoretical_improvement=dbscan_elapsed/gridbscan.dbscan_total_time_
        actual_improvement=dbscan_elapsed/gridbscan_elapsed
        print(f'{i:02d}: {gridbscan_elapsed:1.4f} {gridbscan.dbscan_total_time_:1.4f}')
        print(f'\t\t theoretical improvement: {theoretical_improvement:02.4f}')
        print(f'\t\t      actual improvement: {actual_improvement:02.4f}')
        timing_theoretical_results.append(theoretical_improvement)
        timing_actual_results.append(actual_improvement)

    x_coord=[k for k in minpts]
    style=['bo']
    tmp, = plt.plot(x_coord,timing_theoretical_results, 'bo-')
    stylelines=[tmp]
    plt.title('Theoretical Improvement')
    plt.show()

    x_coord=[k for k in minpts]
    tmp, = plt.plot(x_coord,timing_actual_results, 'bo-')
    stylelines=[tmp]
    plt.title('Actual Improvement')
    plt.show()

    return

def test_household_varyeps(dataset):
    """
    This routine performs a series of tests on the Household dataset varying epsilon
    """
    eps=[500, 1000, 2000]
    num_samples = 100000
    minpts = 50
    grid=(3,3,3,3,3,3,3)

    timing_actual_results = []
    timing_theoretical_results = []

    Xfull=dataset
    # Create smaller dataset
    X=the_rng.choice(Xfull,size=num_samples,replace=False)

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

    for e in eps:
        # Calculate baseline DBSCAN result
        start_time=time.time()
        dbscan=DBSCAN(eps=e,min_samples=minpts).fit(X)
        end_time=time.time()
        dbscan_elapsed=end_time-start_time
        print(f'baseline: {dbscan_elapsed}')

        # Calculate results for GriDBSCAN
        start_time=time.time()
        gridbscan=GriDBSCAN(eps=e,min_samples=minpts,grid=grid).fit(X)
        end_time=time.time()
        gridbscan_elapsed=end_time-start_time
        theoretical_improvement=dbscan_elapsed/gridbscan.dbscan_total_time_
        actual_improvement=dbscan_elapsed/gridbscan_elapsed
        print(f'{i:02d}: {gridbscan_elapsed:1.4f} {gridbscan.dbscan_total_time_:1.4f}')
        print(f'\t\t theoretical improvement: {theoretical_improvement:02.4f}')
        print(f'\t\t      actual improvement: {actual_improvement:02.4f}')
        timing_theoretical_results.append(theoretical_improvement)
        timing_actual_results.append(actual_improvement)

    x_coord=[e for e in eps]
    style=['bo']
    tmp, = plt.plot(x_coord,timing_theoretical_results, 'bo-')
    stylelines=[tmp]
    plt.title('Theoretical Improvement')
    plt.show()

    x_coord=[e for e in eps]
    tmp, = plt.plot(x_coord,timing_actual_results, 'bo-')
    stylelines=[tmp]
    plt.title('Actual Improvement')
    plt.show()

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

    testid=args['test']

    if testid in ['5', 'profile','profilepamap2dbscan','profilepamap2gridbscan']:
        # Set MinPts (k) and Epsilon
        k=int(args['kdist'])
        eps=float(args['eps'])

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
    elif testid=='profile':
        test_dbscan_profile(X,k,eps)
        test_gridbscan_profile(X,k,eps)
    elif testid=='profilepamap2dbscan':
        test_dbscan_profile(X,k,eps,)
    elif testid=='profilepamap2gridbscan':
        test_gridbscan_profile(X,k,eps,grid=(4,4,4,4))
    elif testid=='pamap2varyn':
        test_pamap2_varyn(X)
    elif testid=='pamap2varyk':
        test_pamap2_varyk(X)
    elif testid=='pamap2varyeps':
        test_pamap2_varyeps(X)
    elif testid=='householdvaryn':
        test_household_varyn(X)
    elif testid=='householdvaryk':
        test_household_varyk(X)
    elif testid=='householdvaryeps':
        test_household_varyeps(X)
    else:
        raise Exception('Unrecognized test: ' + testid)

    exit(0)
