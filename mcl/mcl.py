import numpy
import math

_DEFAULT_THRESH=1E-12


def _delta(A,B):
    """Computes the difference between two matrices (L_2,2 norm)"""
    return math.sqrt(sum((A-B).A1**2))

def _create_clusters(M,n_nodes,thresh):
    """Interprets the idempotent matrix at the end of the MCL process
    to produce the actual node clusters it represents."""
    # Get attractors
    attractors = [x for x in range(n_nodes) if M[x,x]>=thresh]
    # Combine attractors into base of clusters. If there is a non-zero entry 
    # for an arc from one attractor to another, they're in the same cluster.
    clusters = []
    for a in attractors:
        for c in clusters:
            if max(M[a,e] for e in c) >= thresh:
                c.append(a)
                break
        else:
            clusters.append([a])
    # Now do the rest of them. Each remaining node mathematically guaranteed
    # to go to at least one attractor. If there's a non-zero entry for an arc
    # from a node to an attractor, then that node is in that attractor's cluster.
    for x in (n for n in range(n_nodes) if n not in attractors):
        for c in clusters:
            if max(M[x,e] for e in c) >= thresh:
                c.append(x)
    return clusters

def _expand(M,e):
    """Expand the matrix by raising it to the e^th power"""
    return M**e
    
def _inflate(M,r):
    """Inflate the matrix by raising each entry to the r^th power"""
    return numpy.power(M,r)    

def _make_stochastic(M):
    """Transforms a matrix into a row-stochastic matrix by dividing
    each row by its sum."""
    return M/M.sum(axis=1)

def _MCL_step(M,e,r):
    """Performs one iteration of the MCL algorithm.
    M is assumed to be a square, row-stochastic matrix with float64 
    data type.
    """
    return _make_stochastic(_inflate(_expand(M,e),r))
    
def MCL(M,E,R,loop_weight=1,labels=None,thresh=_DEFAULT_THRESH):
    """Does the entire MCL process and returns clusters as a list of lists.
    Parameters:
    M - square (weighted) adjacency matrix for graph (type numpy.matrix)
    E - iterable of parameters, e, to use in successive "expand" steps 
    R - iterable of parameters, r, to use in successive "inflate" steps 
    Optional:
    loop_weight - Weight given to the loop edges that are added to each node 
        before MCL begins
    labels - optional list of node labels. Column/row i of 'M' is labeled by
        entry i of 'labels'. Affects how clusters are returned.
    thresh - threshold for changes in successive steps. When the change is
        below 'thresh', then the process stops.
    Returns: a list of lists representing clusters in the input graph. If 
        'labels' is None, the elements will be indices of the matrix. If 
        'labels' is supplied, the elements will be the appropriate labels.
    """
    # Check to see that inputs are valid
    if M.shape[0] != M.shape[1]:
        raise Exception('Matrix must be a square')
    if (labels is not None) and (len(labels) != M.shape[0]):
        raise Exception('Must be exactly one label per matrix column/row')
    n_nodes = M.shape[0]
    # Set up the matrix
    T = _make_stochastic(M+loop_weight*numpy.identity(n_nodes))
    # Loop through the algorithm with the supplied parameters until equilibrium
    for e,r in zip(E,R):
        T2 = _MCL_step(T,e,r)
        if _delta(T,T2) < thresh:
            break
        T = T2
    else:
        raise Exception('Not enough iterations performed to reach equilibrium')
    # Interpret the results to form the clusters
    clusters = _create_clusters(T,n_nodes,thresh)
    if labels is not None:
        return [[labels[x] for x in c] for c in clusters]
    else:
        return clusters

def param_iter(start=[],tail=None):
    """A generator that first goes through each item in 'start', then
    repeats 'tail' indefinitely."""
    for x in start:
        yield x
    while tail is not None:
        yield tail
