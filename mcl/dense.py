"""mcl/dense.py
Code for performing the Markov Cluster Algorithm using numpy.matrix as
the underlying matrix. (AKA "dense" matrices)"""

import numpy
from datetime import datetime

_DEFAULT_ROW_THRESH=1E-14
_DEFAULT_CHECK_ITERATIONS=50

def _delta(A,B):
    """Computes the difference between two matrices"""
    #return numpy.sqrt(sum((A-B).A1**2))
    return abs(A-B).sum()

def _maxdiff(A,B):
    """Computes the maximum difference of corresponding elements in A and B."""
    return abs(A-B).max()
    
def _create_clusters(M,logger):
    """Interprets the idempotent matrix at the end of the MCL process
    to produce the actual node clusters it represents."""
    n_nodes = M.shape[0]
    zero_thresh = 1/(n_nodes+0.5)  #Stable values are all zero or of the form 1/n, 1 <= n <= n_nodes
    # Node degrees: how many attractors each node is connected to
    node_degrees = list(map((lambda x: int(x+0.5)),1/M.max(axis=1).A1))
    # Get attractors
    attractors = [x for x in range(n_nodes) if M[x,x] > zero_thresh]
    attractor_degrees = [node_degrees[a] for a in attractors]
    if logger is not None:
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        logger('{tm} Found {a} attractors'.format(tm=time,a=len(attractors)))
    # Combine attractors into base of clusters. If there is a non-zero entry
    # for an arc from one attractor to another, they're in the same cluster.
    # Attractors can only be in clusters with attractors of the same degree (because
    # all attractors in a cluster are connected to all other attractors in that
    # cluster). Use this to boost performance.
    clusters = [[a] for a,d in zip(attractors,attractor_degrees) if d==1]
    partial_clusters = {d:[] for d in range(2,max(attractor_degrees)+1)}
    for att,deg in zip(attractors,attractor_degrees):
        if deg > 1: #We've already done degree 1
            for i,clu in enumerate(partial_clusters[deg]):
                if M[att,clu[0]] > zero_thresh:
                    clu.append(att)
                    if len(clu) == deg: # Check the cluster for completeness
                        clusters.append(clu)
                        partial_clusters[deg].pop(i)
                    break
            else: # for -> else
                # Because we're only looking at deg > 1, this never creates a
                # "complete" cluster. We don't have to check the length.
                partial_clusters[deg].append([att])
    if logger is not None:
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        logger('{tm} Formed {c} clusters'.format(tm=time,c=len(clusters)))
    # Now do the rest of them. Each remaining node mathematically guaranteed
    # to go to at least one attractor. If there's a non-zero entry for an arc
    # from a node to an attractor, then that node is in that attractor's cluster.
    non_attractors = [x for x in range(n_nodes) if M[x,x] <= zero_thresh]
    for c in clusters:
        c += [x for x in non_attractors if M[x,c[0]] > zero_thresh]
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
    
def MCL(M,E,R,loop_weight=1,labels=None,thresh=None,logger=None):
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
    logger - optional callable for logging steps of the process. Strings are
        passed to this function detailing the status of the process.
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
    if thresh is None:
        thresh = _DEFAULT_ROW_THRESH*n_nodes
    if logger is not None:
        iter = 0
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        logger('{tm} Start. nodes={n}, L1Thresh={thr:.1e}'.format(tm=time,n=n_nodes,thr=thresh))
    # Set up the matrix
    T = _make_stochastic(M+loop_weight*numpy.identity(n_nodes))
    # Loop through the algorithm with the supplied parameters until equilibrium
    check_deltas = [thresh+1]*_DEFAULT_CHECK_ITERATIONS
    for e,r in zip(E,R):
        T2 = _MCL_step(T,e,r)
        if logger is not None:
            iter += 1
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            l1delta = _delta(T,T2)
            maxdiff = _maxdiff(T,T2)
            logger('{tm} Iteration {i}. L1Delta={delta:.1e}, MaxDiff={md:.1e}'.format(tm=time,i=iter,delta=l1delta,md=maxdiff))
        check_deltas.insert(0,_delta(T,T2))
        check_deltas.pop()
        if max(check_deltas) < thresh:
            T = T2
            break
        T = T2
    else:
        raise Exception('Not enough iterations performed to reach equilibrium')
    # Interpret the results to form the clusters
    clusters = _create_clusters(T,logger)
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

def create_matrix(pairs,weights=None,labels=None,directed=False):
    """Creates a dense adjacency matrix based on the values provided, for use in
    the MCL process.
    Parameters:
    pairs - a list of 2-tuples for the edges.
    weights - optional. List of weight values for the edges.
    labels - optional. A list of all the nodes in graph this matrix represents.
        If None, values in I and J are assumed to be 0-based indices. If labels
        is provided, values in I and J should be listed in labels.
    directed - optional. If False (default) mirrored edges will be created for every
        edged provided. If True, only the exact edges specified will be created."""
    if (weights is not None) and (len(pairs)!=len(weights)):
        raise Exception('weights must be the same length as pairs')
    # For every valid index k, an edge will be drawn from pairs[k][0] to pairs[k][1] with
    # weight weights[k] (1, if no weights are provided). If directed==False, a
    # corresponding edge will be created from pairs[k][1] to pairs[k][0].
    if weights is None:
        weights = [1.0]*len(pairs)
    if labels is None:
        matsize = max(max(a,b) for a,b in pairs)+1
        row_idx = [i for i,_ in pairs]
        col_idx = [j for _,j in pairs]
    else:
        matsize = len(labels)
        label_dict = {x:i for i,x in enumerate(labels)}
        try:
            row_idx = [label_dict[a] for a,_ in pairs]
            col_idx = [label_dict[b] for _,b in pairs]
        except KeyError:
            raise Exception('All values in pairs must be present in labels')
    # Create dense matrix
    M = numpy.matrix(numpy.zeros((matsize,matsize)))
    for a,b,w in zip(row_idx,col_idx,weights):
        M[a,b] += w
        if (directed==False) and (a!=b):
            M[b,a] += w
    return M