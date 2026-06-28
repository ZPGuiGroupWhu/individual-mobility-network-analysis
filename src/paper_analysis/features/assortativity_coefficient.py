import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, pearsonr
import networkx as nx


def xicor(x, y, ties="auto"):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = len(y)

    if len(x) != n:
        raise IndexError(
            f"x, y length mismatch: {len(x)}, {len(y)}"
        )

    if ties == "auto":
        ties = len(np.unique(y)) < n
    elif not isinstance(ties, bool):
        raise ValueError(
            f"expected ties either \"auto\" or boolean, "
            f"got {ties} ({type(ties)}) instead"
        )

    y = y[np.argsort(x)]
    r = rankdata(y, method="ordinal")
    nominator = np.sum(np.abs(np.diff(r)))

    if ties:
        l = rankdata(y, method="max")
        denominator = 2 * np.sum(l * (n - l))
        nominator *= n
    else:
        denominator = np.power(n, 2) - 1
        nominator *= 3

    statistic = 1 - nominator / denominator  # upper bound is (n - 2) / (n + 1)
    p_value = norm.sf(statistic, scale=2 / 5 / np.sqrt(n))

    return statistic, p_value


def node_degree_xyw(G, x="out", y="in", edge_weight=None):
    """
    An iterator that returns the degree-degree pairs of the corresponding nodes of all edges in G, and the weights of the edges.
    :param G: NetworkX graph
    :param x: The degree type for source node (directed graphs only).
    :param y: The degree type for target node (directed graphs only).
    :param edge_weight: The edge attribute key. if edge_weight=None, no weights are returned.
    :return: Generates 3-tuple of (degree, degree, weight) values.

    note:
    For undirected graphs each edge is produced twice, once for each edge representation (u, v) and (v, u), with the exception of self-loop edges which only appear once.

    """
    if G.is_directed():
        direction = {"out": G.out_degree, "in": G.in_degree}
        xdeg = direction[x]
        ydeg = direction[y]
    else:
        xdeg = ydeg = G.degree

    for u, nbrsdict in G.adjacency():
        uattr = xdeg[u]
        if G.is_multigraph():
            for v, keys in nbrsdict.items():
                vattr = ydeg[v]
                for _, vals in keys.items():
                    if edge_weight is None:
                        yield (uattr, vattr)
                    else:
                        ewgt = vals.get(edge_weight, None)
                        yield (uattr, vattr, ewgt)
        else:
            for v, vals in nbrsdict.items():
                vattr = ydeg[v]
                if edge_weight is None:
                    yield (uattr, vattr)
                else:
                    ewgt = vals.get(edge_weight, None)
                    yield (uattr, vattr, ewgt)


def node_attribute_xyw(G, attribute, edge_weight=None):
    """
    An iterator that returns the attribute-attribute pairs of the corresponding nodes of all edges in G, and the weights of the edges.
    :param G: NetworkX graph
    :param attribute: The node attribute key.
    :param edge_weight: The edge attribute key. if edge_weight=None, no weights are returned.
    :return:
    (x, y, weight): 3-tuple
    Generates 3-tuple of (attribute, attribute, weight) values.

    note:
    For undirected graphs each edge is produced twice, once for each edge representation (u, v) and (v, u), with the exception of self-loop edges which only appear once.
    refer:
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/assortativity/pairs.html#node_attribute_xy
    """
    Gnodes = G.nodes
    for u, nbrsdict in G.adjacency():
        uattr = Gnodes[u].get(attribute, None)
        if G.is_multigraph():
            for v, keys in nbrsdict.items():
                vattr = Gnodes[v].get(attribute, None)
                for _, vals in keys.items():
                    if edge_weight is None:
                        yield (uattr, vattr)
                    else:
                        ewgt = vals.get(edge_weight, None)
                        yield (uattr, vattr, ewgt)
        else:
            for v, vals in nbrsdict.items():
                vattr = Gnodes[v].get(attribute, None)
                if edge_weight is None:
                    yield (uattr, vattr)
                else:
                    ewgt = vals.get(edge_weight, None)
                    yield (uattr, vattr, ewgt)


def assortativity_coefficient(G, attribute, x='in', y='out', method='pearson', alternative='two-sided',edge_weight=None):
    """
    This function calculates the assortativity coefficient of a given graph G for a specified attribute. Assortativity is a measure of the tendency for nodes in a network to connect to other nodes with similar characteristics.

    :param G: NetworkX graph
    :param attribute: The node attribute key.
    :param x: The degree type for source node (directed graphs only).
    :param y: The degree type for target node (directed graphs only).
    :param method: The method to calculate the correlation coefficient, can be "pearson" or "chatterjee"
    :param alternative: {"two-sided", "greater", "less"}, optional.
            Defines the alternative hypothesis. Default is "two-sided".
    :param edge_weight: The weights of the edges must be integers. If the weights of the edges are provided, they will be used to generate duplicate (x, y) data to take the weights into account
    :return:  the calculated correlation coefficient and p_value.
    """
    # get xyw tuple
    if attribute == "degree":
        xyw = list(node_degree_xyw(G, x, y, edge_weight=edge_weight))
    else:
        xyw = list(node_attribute_xyw(G, attribute, edge_weight=edge_weight))

    # if edge_weight is not None, extend data according to weight.
    if edge_weight is not None:
        xyw_weighted = []
        for x, y, w in xyw:
            if not isinstance(w, int):
                raise ValueError(f"edge weight {w} is not an integer")
            xyw_weighted.extend([(x, y)] * w)
    else:
        xyw_weighted = xyw

    # calculate correlation coefficient
    x, y = zip(*xyw_weighted)
    if method == 'pearson':
        return pearsonr(x,y,alternative=alternative)
    elif method == 'chatterjee':
        return xicor(x,y)
    else:
        raise ValueError(f"method {method} is not supported")
