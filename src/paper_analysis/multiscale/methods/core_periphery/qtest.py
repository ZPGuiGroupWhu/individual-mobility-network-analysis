"""Project-local q-test for core-periphery significance.

The test compares the observed core-periphery quality score with scores
obtained from randomized networks under a supplied null model.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from . import utils


def config_model(G):
    """Generate a random graph from the expected degree model."""
    deg = [d[1] for d in G.degree()]
    return nx.expected_degree_graph(deg)


def erdos_renyi(G):
    """Generate an Erdos-Renyi random graph with the same density as ``G``."""
    n = G.number_of_nodes()
    p = nx.density(G)
    return nx.fast_gnp_random_graph(n, p)


def qtest(
    pair_id,
    coreness,
    G,
    cpa,
    significance_level=0.05,
    null_model=erdos_renyi,
    num_of_rand_net=100,
    weight=None,
    equal_core_num=False,
    **params,
):
    """
    Test significance for a single core-periphery structure.

    Parameters
    ----------
    pair_id : dict
        Core-periphery pair assignment for each node.
    coreness : dict
        Binary or continuous coreness value for each node.
    G : networkx.Graph
        Original graph.
    cpa : CPAlgorithm-like object
        Core-periphery detection algorithm. It must implement ``detect`` and
        ``score``.
    significance_level : float
        Significance threshold for the one-sided test.
    null_model : callable
        Function that generates a randomized graph from ``G``.
    num_of_rand_net : int
        Number of randomized networks.
    weight : str or None
        Edge-weight attribute passed to the scoring function.
    equal_core_num : bool
        If true, randomized scores are computed with the same number of core
        nodes as the observed structure.
    **params
        Extra keyword arguments accepted for compatibility.

    Returns
    -------
    q_obs : float or list
        Observed core-periphery quality score.
    q_tilde : numpy.ndarray
        Quality scores from randomized networks.
    significant : bool
        Whether the observed structure is significant.
    p_value : float
        One-sided p-value, computed as ``Pr(Q_random >= Q_observed)``.
    """
    q_obs = cpa.score(G, pair_id, coreness, weight=weight)
    x = np.array(list(coreness.values()), dtype=float)
    n_core = int(np.sum(x))
    num_nodes = len(x)

    q_tilde = []
    if equal_core_num:
        for _ in range(num_of_rand_net):
            Gr = null_model(G)
            Ar, _ = utils.to_adjacency_matrix(Gr, weight=weight)

            strength = np.array(Ar.sum(axis=1)).reshape(-1)
            order = np.argsort(-strength)
            x_rand = np.zeros(num_nodes)
            x_rand[order[:n_core]] = 1
            q_rand = cpa._score(Ar, None, x_rand)
            q_tilde.append(q_rand)
    else:
        for _ in range(num_of_rand_net):
            Gr = null_model(G)
            Ar, _ = utils.to_adjacency_matrix(Gr, weight=weight)
            cpa.detect(Ar)
            q_tilde.append(cpa.Q_)

    q_tilde = np.array(q_tilde, dtype=float)
    p_value = float(np.mean(q_tilde >= q_obs))
    significant = p_value <= significance_level

    return q_obs, q_tilde, significant, p_value
