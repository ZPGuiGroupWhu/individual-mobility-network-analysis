import numba
import numpy as np
from . import utils
from .CPAlgorithm import CPAlgorithm


@numba.jit(nopython=True, cache=True)
def _score_(A_indptr, A_indices, A_data, num_nodes, x):
    Q = 0.0
    mcc = 0
    mpp = 0
    ncc = 0

    for i in range(num_nodes):
        neighbors = A_indices[A_indptr[i] : A_indptr[i + 1]]
        weights = A_data[A_indptr[i]: A_indptr[i + 1]]
        for j, nei in enumerate(neighbors):
            w = weights[j]
            mcc += w * x[i] * x[nei]
            mpp += w * (1 - x[i]) * (1 - x[nei])
        ncc += x[i]

    avg_edge_w = np.mean(A_data)

    Q = (ncc * (ncc-1) * avg_edge_w - mcc) + mpp
    Q = -Q
    # print(Q)
    return Q


class LipWeighted(CPAlgorithm):
    """Lip's algorithm.

    S. Z. W.~ Lip. A fast algorithm for the discrete core/periphery bipartitioning problem. arXiv, pages 1102.5511, 2011.

    .. highlight:: python
    .. code-block:: python

        >>> import cpnet
        >>> alg = cpnet.Lip()
        >>> alg.detect(G)
        >>> pair_id = alg.get_pair_id()
        >>> coreness = alg.get_coreness()

    .. note::

        - [x] weighted
        - [ ] directed
        - [ ] multiple groups of core-periphery pairs
        - [ ] continuous core-periphery structure
    """

    def __init__(self):
        pass

    def detect(self, G):
        """Detect core-periphery structure.

        :param G: Graph
        :type G: networkx.Graph or scipy sparse matrix
        :return: None
        :rtype: None
        """
        A, nodelabel = utils.to_adjacency_matrix(G,weight='weight')

        strength = np.array(A.sum(axis=1)).reshape(-1)
        avg_edge_w = A.data.mean()
        x = self._detect(strength,avg_edge_w)
        cids = np.zeros(A.shape[0]).astype(int)

        Q = self._score(A, None, x)
        self.nodelabel = nodelabel
        self.c_ = cids
        self.x_ = x
        self.Q_ = Q
        self.qs_ = Q

    def _detect(self, strength, avg_edge_w):
        M = np.sum(strength)/2
        N = len(strength)

        # print(M)
        order = np.argsort(-strength)
        Z = M
        Zbest = np.inf
        kbest = 0
        for k in range(N):
            Z = Z + (k * avg_edge_w) - strength[order[k]]
            if Z < Zbest:
                kbest = k
                Zbest = Z
        _x = np.zeros(N)
        _x[order[: kbest + 1]] = 1

        return _x

    def _score(self, A, c, x):
        """Calculate the strength of core-periphery pairs.

        :param A: Adjacency amtrix
        :type A: scipy sparse matrix
        :param c: group to which a node belongs
        :type c: dict
        :param x: core (x=1) or periphery (x=0)
        :type x: dict
        :return: strength of core-periphery
        :rtype: float
        """
        return [_score_(A.indptr, A.indices, A.data, A.shape[0], x)]


