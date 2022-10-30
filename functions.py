import numpy as np, networkx as nx
from scipy import spatial
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import SpectralClustering

def cluster_rand(positions, num_clusters, p, seed):
    """Generates spatial clusters and treatments via cluster randomization.

    Parameters
    ----------
    positions : numpy array
        n x d array of d-dimensional positions, one for each of the n units.
    num_clusters : int
        number of clusters.
    p : float
        probability of assignment to treatment.
    seed : int
        set seed for k-means clustering initialization.

    Returns
    -------
    D : numpy array
        n x 1 array of indicators, one for each of the n units.
    clusters : numpy array
        n x 1 array of cluster assignments, one for each of the n units. Clusters are labeled 0 to num_clusters-1.
    """
    clustering = SpectralClustering(n_clusters=num_clusters, random_state=seed).fit(positions)
    clusters = clustering.labels_ 
    cluster_rand = np.random.binomial(1, p, num_clusters)
    D = np.array([cluster_rand[clusters[i]] for i in range(positions.shape[0])])
    return D, clusters

def gen_RGG(positions, r=1):
    """Generates an RGG.

    Parameters
    ----------
    positions : numpy array
        n x d array of d-dimensional positions, one for each of the n nodes.
    r : float
        RGG parameter.

    Returns
    -------
    RGG as NetworkX graph
    """
    kdtree = spatial.cKDTree(positions)
    pairs = kdtree.query_pairs(r) # default is Euclidean norm
    RGG = nx.empty_graph(n=positions.shape[0], create_using=nx.Graph())
    RGG.add_edges_from(list(pairs))
    return RGG

def spatial_weights_RGG(positions):
    """Generates spatial weight matrix for Cliff-Ord model based on RGG. 

    Parameters
    ----------
    positions : numpy array
        n x d array of d-dimensional positions, one for each of the n nodes.

    Returns
    -------
    W_norm : scipy sparse matrix in csc format
        Row-normalized adjacency matrix.
    """
    n = positions.shape[0]
    W = gen_RGG(positions)
    W_mat = nx.to_scipy_sparse_matrix(W, nodelist=range(n), format='csc')
    deg_seq_sim = np.squeeze(W_mat.dot(np.ones(n)[:,None]))
    r,c = W_mat.nonzero()
    rD_sp = csc_matrix(((1.0/np.maximum(deg_seq_sim,1))[r], (r,c)), shape=(W_mat.shape))
    return W_mat.multiply(rD_sp) 

def spatial_weights_ID(positions,kappa):
    """Generates spatial weight matrix for moving average model using inverse distances. 

    Parameters
    ----------
    positions : numpy array
        n x d array of d-dimensional positions, one for each of the n nodes.
    kappa : integer
        Rate of decay.

    Returns
    -------
    W_norm : scipy sparse matrix in csc format
        Weight matrix, not row-normalized.
    """
    dists = spatial.distance.squareform(spatial.distance.pdist(positions))
    dists[dists==0] = 1
    return np.minimum(1/np.power(dists,kappa),0.5) # make weights at most 0.5 to avoid singularity at zero

def cliff_ord(D, W_norm, leontieff, errors, theta):
    """Generates outcomes from Cliff-Ord model.

    Parameters
    ----------
    D : numpy array
        n-dimensional vector of treatment indicators.
    W_norm : scipy sparse matrix (csr format)
        Row-normalized spatial weight matrix.
    leontieff : scipy sparse matrix
        Leontieff matrix.
    errors : numpy array
        n-dimensional array of error terms.
    theta : numpy array
        Vector of structural parameters: intercept, endogenous effect, exogenous effect, treatment effect.

    Returns
    -------
    n-dimensional array of outcomes.
    """
    return leontieff.dot(theta[0] + theta[2]*W_norm.dot(D) + theta[3]*D + errors)

def make_Zs(Y,ind1,ind0,pscores1,pscores0):
    """Generates vector of Z_i's, the average of which is the HT estimator.

    Parameters
    ----------
    Y : numpy float array
        n-dimensional outcome vector.
    ind1 : numpy boolean array
        n-dimensional vector of indicators for first exposure mapping.
    ind0 : numpy boolean array
        n-dimensional vector of indicators for second exposure mapping.
    pscores1 : numpy float array
        n-dimensional vector of probabilities of first exposure mapping for each unit.
    pscores0 : numpy float array
        n-dimensional vector of probabilities of second exposure mapping for each unit.

    Returns
    -------
    n-dimensional numpy float array.
    """
    weight1 = ind1.copy().astype('float')
    weight0 = ind0.copy().astype('float')
    weight1[weight1 == 1] = ind1[weight1 == 1] / pscores1[weight1 == 1]
    weight0[weight0 == 1] = ind0[weight0 == 1] / pscores0[weight0 == 1]
    Z = Y * (weight1 - weight0)
    return Z

def standard_error(Z, A, seed):
    """Computes standard error for HT estimator.

    Parameters
    ----------
    Z : (n x k)-dimensional numpy float array
    A : (n x n) scipy sparse matrix
    seed : int

    Returns
    -------
    SE : float
        Standard error of HT estimator.
    """
    weights = (A + A.dot(A) > 0).astype('int') # final dependency graph
    Zcen = (Z - Z.mean()) / Z.shape[0] # demeaned and scaled data
    var = Zcen.dot(weights.dot(Zcen))

    if var <= 0:
        print(f'Non-positive variance estimate in seed {seed}. Using correction.')
        dist_matrix = dijkstra(csgraph=A, directed=False, unweighted=True)
        b = round(Z.shape[0]**(1/6))
        b_neighbors = dist_matrix <= b
        row_sums = np.squeeze(b_neighbors.dot(np.ones(Z.shape[0])[:,None]))
        b_norm = b_neighbors / np.sqrt(row_sums)[:,None]
        weights = b_norm.dot(b_norm.T)
        var = Zcen.dot(weights.dot(Zcen))

    return np.sqrt(var)


