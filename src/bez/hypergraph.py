import networkx as nx
from dataclasses import dataclass
import itertools
import numpy as np

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

HYPER = 0
EDGE = 1
NODE = 2


@dataclass
class HyperEdge:
    degree: int
    score: float | None
    edges: list[int]


class HyperGraph:
    def __init__(self, topo_graph: nx.MultiGraph):
        self.counter = itertools.count(start=0)
        self.hyper2edge = nx.DiGraph()
        self.topo = topo_graph
        self.table = dict()
        for u, v, key in topo_graph.edges(keys=True):
            i_h = next(self.counter)
            self.table[i_h] = HyperEdge(degree=3, score=None, edges=[(u, v, key)])
            self.hyper2edge.add_edge(i_h, (u, v, key), extremity=True)

    def add_hyperedge(self, degree: int, edges: list[int]):
        i_new = next(self.counter)
        self.table[i_new] = HyperEdge(degree=degree, edges=edges, score=None)
        for edge in edges:
            self.hyper2edge.add_edge(i_new, edge)
        self.hyper2edges.edges[(i_new, edges[0])]["extremity"] = True
        self.hyper2edges.edges[(i_new, edges[-1])]["extremity"] = True

    def merge(self, ha: int, hb: int, node: int):
        hyper_a = self.table[ha]
        hyper_b = self.table[hb]
        if self.table[ha].edges[-1][1] != node:
            hyper_a.edges.reverse()
        if self.table[hb].edges[0][0] != node:
            hyper_b.edges.reverse()
        d = max(self.table[ha].degree, self.table[hb].degree)
        s_merge = hyper_a.edges + hyper_b.edges
        self.add_hyperedge(degree=d, edges=s_merge)

        del self.table[ha]
        del self.table[hb]
        self.hyper2edge.remove_node(ha)
        self.hyper2edge.remove_node(hb)

    def split(self, h0: int, node: int):
        hyper = self.table[h0]
        i_split = [i for i, edge in enumerate(hyper.edges) if edge[1] == node][0]
        sa = hyper.edges[: i_split + 1]
        sb = hyper.edges[i_split + 1 :]
        self.add_hyperedge(degree=hyper.degree, edges=sa)
        self.add_hyperedge(degree=hyper.degree, edges=sb)
        self.hyper2edge.remove_node(h0)
        del self.table[h0]

    def overlap(self, ha, hb, edge):
        """
        we suppose that hyper-edge `ha` starts or ends with u
        where (u, v) = edge
        """
        (u, v, k) = edge
        hyper_a = self.table[ha]
        _hyper_b = self.table[hb]
        if hyper_a.edges[-1][1] != u:
            hyper_a.edges.reverse()
        hyper_a.edges.append(edge)
        self.add_hyperedge(degree=hyper_a.degree, edges=hyper_a.edges)
        self.hyper2edge.remove_node(ha)
        del self.table[ha]

    def dissociate(self, ha, hb, edge):
        """
        We suppose that the hyper-edge starts with (v, u) or ends with (u, v)
        where (u, v) = edge
        """
        (u, v, k) = edge
        hyper_a = self.table[ha]
        hyper_b = self.table[hb]
        assert (u, v, k) in hyper_b.edges or (v, u, k) in hyper_b.edges
        if hyper_a.edges[-1] != (u, v, k):
            hyper_a.edges.reverse()
        hyper_a.edges.pop()
        self.add_hyperedge(degree=hyper_a.degree, edges=hyper_a.edges)
        self.hyper2edge.remove_node(ha)
        del self.table[ha]

    def increase_degree(self, i: int):
        d = self.table[i].degree
        assert d < 3
        self.table[i].degree = d + 1
        self.table[i].score = None

    def decrease_degree(self, i: int):
        d = self.table[i].degree
        assert d > 2
        self.table[i].degree = d - 1
        self.table[i].score = None

    def random_merge(self):
        """
        To do a random merge, we need to find one node A such that there are 2 hyperedges with extremity A
        """

    def random_split(self):
        candidates = [i for (i, h) in self.table if len(h.edges) >= 2]
        i = np.random.choice(candidates)
        node = np.random.choice([u for (u, v, k) in self.table[i].edges[1:]])
        self.split(i, node)

    def random_overlap(self):
        """
        To do a ranodm overlap, we need to find 2 hyperedges U, V such that U is completely contained in V
        """

    def random_dissociate(self):
        """
        To do a random dissociation, we need to find 2 hyperedges such that U is completely contained in V
        """

    def random_perturbation(self):
        choice = np.random.choice(range(6), p=CHOICE_DISTRIBUTION)
        if choice == INCREASE_DEGREE:
            candidates = [i for i, h in self.table.items() if h.degree < 3]
            i_h = np.random.choice(candidates)
            self.increase_degree(i_h)
        if choice == DECREASE_DEGREE:
            candidates = [i for i, h in self.table.items() if h.degree > 1]
            i_h = np.random.choice(candidates)
            self.decrease_degree(i_h)
        if choice == OVERLAP:
            self.random_overlap()
        if choice == DISSOCIATE:
            self.random_dissociate()
        if choice == MERGE:
            self.random_merge()
        if choice == SPLIT:
            self.random_split()
