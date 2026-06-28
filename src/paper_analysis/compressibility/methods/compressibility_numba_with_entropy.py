"""Entropy-enabled graph compressibility interface.

This module keeps the legacy rate-distortion implementation in
``compressibility_numba.py`` and adds a metrics wrapper that also reports the
uncompressed random-walk entropy rate. It is the default interface used by the
release workflow for core-periphery module compressibility analysis.
"""

from __future__ import annotations

import networkx as nx

from paper_analysis.compressibility.methods.compressibility_numba import (
    compute_compressibility,
    rate_distortion,
)


def _component_metrics(G, weight=None):
    """Compute compressibility and entropy rate for a connected component."""
    A = nx.to_numpy_array(G, weight=weight)
    rd_upper, _, _, _ = rate_distortion(A)
    return {
        "compressibility": compute_compressibility(rd_upper),
        "random_walk_entropy": rd_upper[-1],
    }


def graph_rate_distortion_metrics(G, weight=None, mode="weighted"):
    """
    Compute graph compressibility and random-walk entropy rate.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.
    weight : str or None
        Edge attribute used as weight when constructing the adjacency matrix.
    mode : {"weighted", "gcc", "all"}
        Component handling policy. ``weighted`` averages component metrics by
        node count, ``gcc`` uses the largest connected component, and ``all``
        computes metrics on the graph as supplied.

    Returns
    -------
    dict
        Dictionary with ``compressibility`` and ``random_walk_entropy``.
    """
    if G.number_of_nodes() < 2:
        return {
            "compressibility": 0.0,
            "random_walk_entropy": 0.0,
        }

    if G.is_directed():
        components = list(nx.strongly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    if mode == "all":
        return _component_metrics(G, weight=weight)

    if mode == "gcc":
        gcc_nodes = max(components, key=len)
        return _component_metrics(G.subgraph(gcc_nodes), weight=weight)

    if mode == "weighted":
        total_nodes = G.number_of_nodes()
        compressibility = 0.0
        random_walk_entropy = 0.0

        for comp in components:
            subG = G.subgraph(comp)

            if subG.number_of_nodes() < 2:
                continue

            metrics = _component_metrics(subG, weight=weight)
            node_fraction = subG.number_of_nodes() / total_nodes
            compressibility += metrics["compressibility"] * node_fraction
            random_walk_entropy += metrics["random_walk_entropy"] * node_fraction

        return {
            "compressibility": compressibility,
            "random_walk_entropy": random_walk_entropy,
        }

    raise ValueError(
        f"Unknown mode: {mode}. "
        "Choose from {'weighted', 'gcc', 'all'}."
    )


def graph_compressibility(G, weight=None, mode="weighted"):
    """
    Compute graph compressibility using the entropy-enabled metrics wrapper.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.
    weight : str or None
        Edge-weight attribute name.
    mode : {"weighted", "gcc", "all"}
        Component handling policy.

    Returns
    -------
    float
        Graph compressibility.
    """
    return graph_rate_distortion_metrics(
        G,
        weight=weight,
        mode=mode,
    )["compressibility"]
