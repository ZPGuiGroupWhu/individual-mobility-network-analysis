"""
Legacy compressibility implementation without random-walk entropy output.

Python translation of the provided MATLAB function rate_distortion,
with safe handling to avoid "divide by zero encountered in log2" warnings.

Dependencies: numpy, scipy
Usage:
    from rate_distortion import rate_distortion
    S, S_low, clusters, Gs = rate_distortion(G, heuristic=1, num_pairs=100)

Notes:
 - G: numpy.ndarray (N x N) adjacency (can be weighted, directed)
 - Returns:
    S: numpy array length N
    S_low: numpy array length N
    clusters: list length N; clusters[i] is list-of-lists for that clustering level
    Gs: list length N; Gs[i] is adjacency (or scaled) at that level
"""
import numpy as np
from scipy.sparse.linalg import eigs
from itertools import combinations
import random
import time
import networkx as nx
from scipy.sparse.linalg import eigs
import numpy as np
from itertools import combinations
import warnings
from paper_analysis.utils.io import read_graph
from numba import njit, prange


def rate_distortion(A, heuristic=7, num_pairs=100):
    """
    Computes rate-distortion curve when compressing a given graph G.
    Numba-accelerated version.

    Args:
        A (np.ndarray): NxN adjacency matrix.
        heuristic (int): Determines how pairs of nodes are chosen.
        num_pairs (int): Number of pairs to try combining at each iteration.

    Returns:
        S (list): Upper bound on entropy rate after clustering.
        S_low (list): Lower bound on entropy rate.
        clusters (list): clusters[n] lists the nodes in each of the n clusters.
        Gs (list): Gs[n] is the joint transition probability matrix for n clusters.
    """
    # Ensure float
    A = np.array(A, dtype=float)
    N = A.shape[0]
    E = np.sum(A) / 2.0

    S = np.zeros(N + 1)
    S_low = np.zeros(N + 1)
    clusters = [None] * (N + 1)
    Gs = [None] * (N + 1)

    # Transition probability
    row_sums = A.sum(axis=1, keepdims=True)
    P_old = np.divide(A, row_sums, out=np.zeros_like(A), where=row_sums != 0)

    # Steady-state distribution
    vals, vecs = np.linalg.eig(P_old.T)
    ind = np.argmax(vals.real)
    p_ss = vecs[:, ind].real
    p_ss = p_ss / np.sum(p_ss)
    p_ss_old = p_ss.copy()

    # Safe log2
    logP_old = safe_log2(P_old)
    S_old = -np.sum(p_ss_old * np.sum(P_old * logP_old, axis=1))

    P_joint = P_old * p_ss_old[:, np.newaxis]
    P_low = P_old.copy()

    # Initial
    S[N] = S_old
    S_low[N] = S_old
    clusters[N] = [[i] for i in range(N)]
    Gs[N] = A.copy()

    # Main loop
    for n in range(N - 1, 1, -1):
        curr_size = n + 1

        I, J = select_pairs(P_old, P_joint, p_ss_old, curr_size, heuristic, num_pairs)

        if len(I) == 0:
            continue

        # Compute entropy for all pairs in parallel
        S_all = compute_entropy_for_pairs(P_old, p_ss_old, I, J, S_old)

        # Pick minimum entropy
        min_inds = np.where(np.isclose(S_all, np.min(S_all), atol=1e-12))[0]
        min_ind = np.random.choice(min_inds)

        # Update
        S_old = S_all[min_ind]
        S[n] = S_old

        i_new, j_new = I[min_ind], J[min_ind]

        # Update clusters and P matrices
        all_inds = np.arange(curr_size)
        inds_not_ij = np.delete(all_inds, [i_new, j_new])
        p_ss_new = np.concatenate([p_ss_old[inds_not_ij], [p_ss_old[i_new] + p_ss_old[j_new]]])

        # Update P_joint
        block_11 = P_joint[np.ix_(inds_not_ij, inds_not_ij)]
        block_12 = np.sum(P_joint[np.ix_(inds_not_ij, [i_new, j_new])], axis=1, keepdims=True)
        block_21 = np.sum(P_joint[np.ix_([i_new, j_new], inds_not_ij)], axis=0, keepdims=True)
        block_22 = np.sum(P_joint[np.ix_([i_new, j_new], [i_new, j_new])], keepdims=True)
        P_joint = np.vstack([np.hstack([block_11, block_12]),
                             np.hstack([block_21, block_22])])

        denom = p_ss_new[:, np.newaxis]
        P_old = np.divide(P_joint, denom, out=np.zeros_like(P_joint), where=denom != 0)
        p_ss_old = p_ss_new
        logP_old = safe_log2(P_old)

        # Update clusters
        prev_clusters = clusters[n + 1]
        new_cluster_list = [prev_clusters[k] for k in inds_not_ij]
        merged_cluster = prev_clusters[i_new] + prev_clusters[j_new]
        new_cluster_list.append(merged_cluster)
        clusters[n] = new_cluster_list
        Gs[n] = P_joint * 2 * E

        # Update lower bound
        P_low_subset = P_low[:, inds_not_ij]
        P_low_combined = P_low[:, i_new] + P_low[:, j_new]
        P_low = np.hstack([P_low_subset, P_low_combined[:, np.newaxis]])
        logP_low = safe_log2(P_low)
        S_low[n] = -np.sum(p_ss * np.sum(P_low * logP_low, axis=1))

    return S[1:], S_low[1:], clusters[1:], Gs[1:]

# --------------------
# Helpers
# --------------------

def safe_log2(arr):
    arr = np.asarray(arr, dtype=float)
    res = np.zeros_like(arr)
    mask = arr > 0
    res[mask] = np.log2(arr[mask])
    return res

def select_pairs(P_old, P_joint, p_ss_old, curr_size, heuristic, num_pairs):
    # Returns arrays I, J of candidate pairs
    if heuristic == 1:
        pairs = list(combinations(range(curr_size), 2))
        pairs = np.array(pairs)
        I, J = pairs[:, 0], pairs[:, 1]
    elif heuristic == 2:
        pairs = list(combinations(range(curr_size), 2))
        num_to_sample = min(num_pairs, len(pairs))
        inds = np.random.choice(len(pairs), num_to_sample, replace=False)
        pairs = np.array(pairs)[inds]
        I, J = pairs[:, 0], pairs[:, 1]
    elif heuristic == 3:
        P_sym = P_old + P_old.T
        I, J = np.where(np.triu(P_sym, 1))
    elif heuristic == 4:
        P_sym = P_old + P_old.T
        possible_I, possible_J = np.where(np.triu(P_sym, 1))
        num_possible = len(possible_I)
        if num_possible > 0:
            pair_inds = np.random.choice(num_possible, min(num_pairs, num_possible), replace=False)
            I, J = possible_I[pair_inds], possible_J[pair_inds]
        else:
            I, J = np.array([], dtype=int), np.array([], dtype=int)
    elif heuristic == 5:
        P_joint_symm = np.triu(P_joint + P_joint.T, 1)
        flat = P_joint_symm.flatten()
        k = min(num_pairs, np.sum(P_joint_symm > 0))
        if k > 0:
            inds = np.argpartition(flat, -k)[-k:]
            I, J = np.unravel_index(inds, P_joint_symm.shape)
        else:
            I, J = np.array([], dtype=int), np.array([], dtype=int)
    elif heuristic == 6:
        diag_P = np.diag(P_joint)
        term1 = np.tile(diag_P, (curr_size, 1))
        term2 = np.tile(diag_P[:, np.newaxis], (1, curr_size))
        P_joint_symm = np.triu(P_joint + P_joint.T + term1 + term2, 1)
        flat = P_joint_symm.flatten()
        k = min(num_pairs, np.sum(P_joint_symm > 0))
        if k > 0:
            inds = np.argpartition(flat, -k)[-k:]
            I, J = np.unravel_index(inds, P_joint_symm.shape)
        else:
            I, J = np.array([], dtype=int), np.array([], dtype=int)
    elif heuristic == 7:
        p_ss_mat = np.tile(p_ss_old, (curr_size, 1))
        P_ss_temp = p_ss_mat + p_ss_mat.T
        iu, ju = np.triu_indices(curr_size, k=1)
        vals = P_ss_temp[iu, ju]
        k = min(num_pairs, len(vals))
        if k>0:
            inds = np.argpartition(vals, -k)[-k:]
            I, J = iu[inds], ju[inds]
        else:
            I, J = np.array([], dtype=int), np.array([], dtype=int)
    elif heuristic == 8:
        I = np.array([0])
        J = np.array([curr_size-1])
    else:
        raise ValueError('Invalid heuristic')
    return I, J

# --------------------
# Numba JIT parallelized core entropy calculation
# --------------------
@njit(parallel=True)
def compute_entropy_for_pairs(P_old, p_ss_old, I, J, S_old):
    num_pairs_temp = len(I)
    S_all = np.zeros(num_pairs_temp)
    curr_size = P_old.shape[0]

    for ind_pair in prange(num_pairs_temp):
        i = I[ind_pair]
        j = J[ind_pair]

        all_inds = np.arange(curr_size)
        inds_not_ij = np.delete(all_inds, [i, j])

        # New stationary
        p_ss_temp = np.zeros(len(inds_not_ij)+1)
        for idx, v in enumerate(inds_not_ij):
            p_ss_temp[idx] = p_ss_old[v]
        p_ss_temp[-1] = p_ss_old[i]+p_ss_old[j]

        # Top -> merged
        P_temp_1 = np.zeros(len(inds_not_ij))
        for idx, row in enumerate(inds_not_ij):
            P_temp_1[idx] = (p_ss_old[row]*P_old[row,i]+p_ss_old[row]*P_old[row,j])/p_ss_temp[idx] if p_ss_temp[idx]>0 else 0.0

        # Merged -> top
        P_temp_2 = np.zeros(len(inds_not_ij))
        for idx, col in enumerate(inds_not_ij):
            P_temp_2[idx] = (p_ss_old[i]*P_old[i,col]+p_ss_old[j]*P_old[j,col])/p_ss_temp[-1] if p_ss_temp[-1]>0 else 0.0

        # Merged -> merged
        P_temp_3 = (p_ss_old[i]*(P_old[i,i]+P_old[i,j])+p_ss_old[j]*(P_old[j,i]+P_old[j,j]))/p_ss_temp[-1] if p_ss_temp[-1]>0 else 0.0

        # safe log2
        logP_temp_1 = np.zeros_like(P_temp_1)
        for idx in range(len(P_temp_1)):
            if P_temp_1[idx]>0:
                logP_temp_1[idx]=np.log2(P_temp_1[idx])
        logP_temp_2 = np.zeros_like(P_temp_2)
        for idx in range(len(P_temp_2)):
            if P_temp_2[idx]>0:
                logP_temp_2[idx]=np.log2(P_temp_2[idx])
        logP_temp_3 = np.log2(P_temp_3) if P_temp_3>0 else 0.0

        # old terms
        term_old_i = 0.0
        term_old_j = 0.0
        term_old_row_i = 0.0
        term_old_row_j = 0.0
        sub_term_i = 0.0
        sub_term_j = 0.0

        for idx in range(curr_size):
            if P_old[idx,i]>0:
                term_old_i += p_ss_old[idx]*P_old[idx,i]*np.log2(P_old[idx,i])
            if P_old[idx,j]>0:
                term_old_j += p_ss_old[idx]*P_old[idx,j]*np.log2(P_old[idx,j])
        for idx in range(curr_size):
            if P_old[i,idx]>0:
                term_old_row_i += p_ss_old[i]*P_old[i,idx]*np.log2(P_old[i,idx])
            if P_old[j,idx]>0:
                term_old_row_j += p_ss_old[j]*P_old[j,idx]*np.log2(P_old[j,idx])

        if P_old[i,i]>0:
            sub_term_i += p_ss_old[i]*P_old[i,i]*np.log2(P_old[i,i])
        if P_old[i,j]>0:
            sub_term_i += p_ss_old[i]*P_old[i,j]*np.log2(P_old[i,j])
        if P_old[j,j]>0:
            sub_term_j += p_ss_old[j]*P_old[j,j]*np.log2(P_old[j,j])
        if P_old[j,i]>0:
            sub_term_j += p_ss_old[j]*P_old[j,i]*np.log2(P_old[j,i])

        dS = -np.sum(p_ss_temp[:-1]*P_temp_1*logP_temp_1) - p_ss_temp[-1]*np.sum(P_temp_2*logP_temp_2) - p_ss_temp[-1]*P_temp_3*logP_temp_3 + term_old_i + term_old_j + term_old_row_i + term_old_row_j - sub_term_i - sub_term_j

        S_all[ind_pair] = S_old + dS

    return S_all


def compute_compressibility(rd_upper):
    """
    Compute network compressibility from the rate-distortion upper bound.

    Parameters
    ----------
    rd_upper : ndarray of shape (N,)
        Rate-distortion upper bound curve.

    Returns
    -------
    network_compressibility : float
    """
    return np.mean(rd_upper[-1] - rd_upper)


def graph_compressibility(
    G,
    weight=None,
    mode="weighted"
):
    """
    Compute graph compressibility, with explicit handling of disconnected graphs.

    Parameters
    ----------
    G : networkx.Graph or DiGraph
        Input graph.
    weight : str or None
        Edge-weight attribute name.
    mode : {"weighted", "gcc", "all"}
        - "weighted": compute each connected component and weight by node count.
        - "gcc": compute only the giant connected component.
        - "all": compute the whole graph directly, assuming it is connected.

    Returns
    -------
    compressibility : float
        Graph compressibility.
    """

    if G.is_directed():
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    if mode == "all":
        A = nx.to_numpy_array(G, weight=weight)
        rd_upper, _, _, _ = rate_distortion(A)
        return compute_compressibility(rd_upper)

    if mode == "gcc":
        gcc_nodes = max(components, key=len)
        subG = G.subgraph(gcc_nodes)
        A = nx.to_numpy_array(subG, weight=weight)
        rd_upper, _, _, _ = rate_distortion(A)
        return compute_compressibility(rd_upper)

    if mode == "weighted":
        total_nodes = G.number_of_nodes()
        compressibility = 0.0

        for comp in components:
            subG = G.subgraph(comp)

            if subG.number_of_nodes() < 2:
                continue

            A = nx.to_numpy_array(subG, weight=weight)
            rd_upper, _, _, _ = rate_distortion(A)
            comp_value = compute_compressibility(rd_upper)

            compressibility += (
                comp_value * subG.number_of_nodes() / total_nodes
            )

        return compressibility

    raise ValueError(
        f"Unknown mode: {mode}. "
        "Choose from {'weighted', 'gcc', 'all'}."
    )


