import polars as pl
import networkx as nx
import numpy as np
from dataclasses import dataclass


INCREASE_DEGREE = 0
DECREASE_DEGREE = 1
MERGE = 2
SPLIT = 3
OVERLAP = 4
DISSOCIATE = 5

CHOICE_DISTRIBUTION = [
    # change degree
    1 / 6,
    1 / 6,
    # merge or split
    1 / 6,
    1 / 6,
    # overlap or dissociate
    1 / 6,
    1 / 6,
]


@dataclass
class HyperEdgeIndex:
    i: int
    reverse: bool


class HyperGraph:
    def __init__(self, topo_graph: nx.MultiGraph):
        self.data = pl.DataFrame(
            [
                {"nodes": [u, v], "edges": [key], "degree": 3, "score": None}
                for u, v, key, weight in topo_graph.edges(keys=True)
            ]
        ).with_row_index()
        self.topo = topo_graph

    def get_hyperedge(self, x: HyperEdgeIndex):
        nodes = self.data[x.i, "nodes"]
        edges = self.data[x.i, "edges"]
        if x.reverse:
            return (nodes[::-1], edges[::-1])
        return (nodes, edges)

    def merge(self, a: HyperEdgeIndex, b: HyperEdgeIndex):
        na, ea = self.get_hyperedge(b)
        nb, eb = self.get_hyperedge(a)
        assert na[-1] == nb[0]
        new_nodes = na[:-1] + nb[1:]
        new_edges = ea + eb
        self.data = self.data.remove(
            (pl.col("index") == a.i) | (pl.col("index") == b.i)
        ).extend(
            pl.DataFrame(
                {
                    "nodes": [new_nodes],
                    "edges": [new_edges],
                    "score": [None],
                }
            )
        )

    def split(self, a: HyperEdgeIndex, node: int):
        na, ea = self.get_hyperedge(a)
        j = na.index(node)
        n1 = na[: j + 1]
        n2 = na[j:]
        e1 = ea[:j]
        e2 = ea[j:]
        self.data = self.data.remove((pl.col("index") == a.i)).extend(
            pl.DataFrame(
                {
                    "nodes": [n1, n2],
                    "edges": [e1, e2],
                    "score": [None, None],
                }
            )
        )

    def overlap(self, a: HyperEdgeIndex, b: HyperEdgeIndex):
        na, ea = self.get_hyperedge(b)
        nb, eb = self.get_hyperedge(a)
        assert na[-1] == nb[0]
        assert len(nb) >= 2
        self.data
        self.data = self.data.remove((pl.col("index") == a.i)).extend(
            pl.DataFrame(
                {
                    "nodes": [na + [nb[1]]],
                    "edges": [ea + eb[0]],
                    "score": [None],
                }
            )
        )

    def dissociate(self, a: HyperEdgeIndex, b: HyperEdgeIndex):
        na, ea = self.get_hyperedge(b)
        nb, eb = self.get_hyperedge(a)
        assert na[-2] == nb[0]
        assert na[-1] == nb[1]
        assert ea[-1] == eb[0]
        self.data = self.data.remove((pl.col("index") == a.i)).extend(
            pl.DataFrame(
                {
                    "nodes": [na[:-1]],
                    "edges": [ea[:-1]],
                    "score": [None],
                }
            )
        )

    def increase_degree(self, i: int):
        d = self.data[i, "degree"]
        assert d == 2
        self.data[i, "degree"] = 3
        self.data[i, "score"] = None

    def decrease_degree(self, i: int):
        d = self.data[i, "degree"]
        assert d == 3
        self.data[i, "degree"] = 2
        self.data[i, "score"] = None

    def random_merge(self):
        """
        To do a random merge, we need to find one node A such that there are 2 hyperedges with extremity A
        """

    def random_split(self):
        """
        To do a random split, we need to find any hyperedge containing at least 3 nodes
        """

    def random_overlap(self):
        """
        To do a ranodm overlap, we need to find 2 hyperedges U, V such that U is completely contained in V
        """

    def random_dissociate(self):
        """
        To do a random dissociation, we need to find 2 hyperedges such that U is completely contained in V
        """

    def random_perturbation(self):
        choice = np.random.choice(range(3), p=CHOICE_DISTRIBUTION)
        if choice == INCREASE_DEGREE:
            i_h = np.random.choice(self.data.index)
            self.increase_degree(i_h)
        if choice == DECREASE_DEGREE:
            i_h = np.random.choice(self.data.index)
            self.decrease_degree(i_h)
        if choice == OVERLAP:
            self.random_overlap()
        if choice == DISSOCIATE:
            self.random_dissociate()
        if choice == MERGE:
            self.random_merge()
        if choice == SPLIT:
            self.random_split()
