# Coded for Python 3. Run this file to reproduce simulations.

import numpy as np, pandas as pd, multiprocessing as mp, traceback, sys
from scipy import spatial
from scipy.stats import norm
from scipy.sparse import identity, lil_matrix
from scipy.sparse.linalg import inv
from functions import *

processes = 16        # number of cores for parallel computing
B = 5000              # number of simulations
Ns = [250,500,1000]   # sample sizes
m_nCoef = 2/3         # number of clusters m_n will be equal to n^m_nCoef
weight_matrix = 'RGG' # choice of spatial weight matrix of cliff-ord model. options: RGG, invdist
kappa = 4             # parameter of spatial weight matrix of moving average model
p = 0.5               # cluster assignment probability
superpop = False      # set to True to evaluate CI coverage for E[\theta_n] rather than \theta_n
beta_cliff_ord = np.array([-1,0.8,1,1]) # parameters of cliff-ord model

##### Task per node #####

def one_sim(seed, n, R_n, m_n, K_n):
    """
    Task to be parallelized: one simulation draw. 
    """
    np.random.seed(seed=seed)

    #### generate data ####
    positions = R_n * np.random.uniform(-1,1,[n,2])
    D, clusters = cluster_rand(positions, m_n, p, seed)
    W = spatial_weights_RGG(positions) if weight_matrix=='RGG' else spatial_weights_ID(positions,kappa)
    errors = np.random.normal(size=n)
    leontieff = inv(identity(n,format='csc') - beta_cliff_ord[1]*W) if weight_matrix=='RGG' else W
    W_norm = W if weight_matrix=='RGG' else np.zeros(n)
    Y = cliff_ord(D, W_norm, leontieff, errors, beta_cliff_ord)
    estimand = (cliff_ord(np.ones(n), W_norm, leontieff, errors, beta_cliff_ord) - cliff_ord(np.zeros(n), W_norm, leontieff, errors, beta_cliff_ord)).mean()

    #### HT estimator ####
    kdtree = spatial.cKDTree(positions) 
    Kneighbors = kdtree.query_ball_point(positions, K_n) # K_n-neighbors of each unit
    Kneighbor_treatments = [np.array([D[i] for i in vec]) for vec in Kneighbors]
    ind1 = np.array([(vec==1).prod() for vec in Kneighbor_treatments])
    ind0 = np.array([(vec==0).prod() for vec in Kneighbor_treatments])
    cluster_nbhrs = [np.unique(clusters[vec]) for vec in Kneighbors] # for each unit i, records the list of clusters that intersect i's K_n-neighborhood. see nested list comprehension: https://blog.finxter.com/list-comprehension/
    intersects = np.array([vec.size for vec in cluster_nbhrs]) # number of clusters that intersect each unit's K_n-neighborhood
    Z = make_Zs(Y, ind1, ind0, p**intersects, (1-p)**intersects)
    estimator = Z.mean()

    #### standard error ####
    cluster_members = [np.flatnonzero(clusters==cluster) for cluster in range(m_n)] # for each cluster, records the vector of units that are members
    unit_cluster_nbhrs = [[unit for cluster in clusters for unit in cluster_members[cluster]] for clusters in cluster_nbhrs] # for each unit i, records the list of units in the clusters that intersect i's K_n-neighborhood
    A0 = lil_matrix((n,n),dtype='int')
    for i in range(n): # convert unit_cluster_nbhrs into a sparse binary matrix (graph)
        for j in unit_cluster_nbhrs[i]:
            A0[i,j]=1
    A0 = A0.tocsr()
    SE = standard_error(Z, A0, seed)
    naive_SE = np.sqrt(Z.var() / n)

    return [estimator, SE, naive_SE, estimand]

##### Containers #####

estimators = np.zeros((B,len(Ns))) 
estimands = np.zeros((B,len(Ns)))
SEs = np.zeros((B,len(Ns)))
naive_SEs = np.zeros((B,len(Ns)))

##### Main #####

for index,n in enumerate(Ns):
    R_n = np.sqrt(n)
    m_n = int(np.round(n**(m_nCoef)))
    K_n = R_n/np.sqrt(m_n)
    
    def one_sim_wrapper(b):
        try:
            return one_sim(b, n, R_n, m_n, K_n)
        except:
            print('%s: %s' % (b, traceback.format_exc()))
            sys.stdout.flush()
    pool = mp.Pool(processes=processes, maxtasksperchild=1)
    parallel_output = pool.imap(one_sim_wrapper, range(B), chunksize=25) 
    pool.close()
    pool.join()
    results = np.array([r for r in parallel_output])

    estimators[:,index] = results[:,0]
    SEs[:,index] = results[:,1]
    naive_SEs[:,index] = results[:,2]
    estimands[:,index] = results[:,3]

estimate = estimators.mean(axis=0)
SE = SEs.mean(axis=0)
naive_SE = naive_SEs.mean(axis=0)
bias_vec = estimators-estimands.mean(axis=0) if superpop else estimators-estimands
bias = np.abs(bias_vec.mean(axis=0))
variance = estimators.var(axis=0)
coverage = (np.abs(bias_vec) / SEs < norm.ppf(1-0.05/2)).mean(axis=0)
naive_coverage = (np.abs(bias_vec) / naive_SEs < norm.ppf(1-0.05/2)).mean(axis=0)
oracle_coverage = (np.abs(bias_vec) / np.sqrt(variance) < norm.ppf(1-0.05/2)).mean(axis=0)
table = pd.DataFrame(np.vstack([ Ns, coverage, naive_coverage, oracle_coverage, bias, variance, SE, naive_SE, estimate ]))
table.index = ['$n$', 'Coverage', 'Naive Cvg', 'Oracle Cvg', 'Bias', 'Variance', 'SE', 'Naive SE', 'Estimate']
print(table.to_latex(float_format = lambda x: '%.3f' % x, header=True, escape=False))

nstring = ''
for n in Ns:
    nstring += '_' + str(n)
name = weight_matrix if weight_matrix=='RGG' else weight_matrix + str(kappa)
print('Weight matrix: ' + name + '\n')
table.to_csv('results_' + name + nstring + '.csv', header=False, index=False)

