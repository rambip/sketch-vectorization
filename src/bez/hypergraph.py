import networkx as nx
from dataclasses import dataclass
import itertools
import numpy as np
from collections import Counter
import random

INCREASE_DEGREE = 0
DECREASE_DEGREE = 1
MERGE = 2
SPLIT = 3
OVERLAP = 4
DISSOCIATE = 5

SOURCE = "*"

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


#@dataclass
class HyperEdge:
    edges: list[tuple[int, int, int]]
    degree: int
    score: float | None

    def __init__(self, edges, degree, score):
        self.edges = edges
        self.degree = degree
        self.score = score

    def reverse(self):
        self.edges = [(v, u, k) for (u, v, k) in reversed(self.edges)]

    def __repr__(self):
        tokens = []
        tokens.append(f"{self.edges[0][0]}")
        print(self.edges)
        for (u, v, k) in self.edges:
            tokens.append(f"- [{k}] -> {v}")
        tokens.append(f"( degree = {self.degree})")
        return " ".join(tokens)

class SampleError(BaseException):
    ...


class HyperGraph:
    def __init__(self, topo_graph: nx.MultiGraph):
        self.g = nx.DiGraph()
        self.topo = topo_graph
        for u, v, key in topo_graph.edges(keys=True):
            h = HyperEdge(degree=3, score=None, edges=[(u, v, key)])
            self.g.add_edge(h, u)
            self.g.add_edge(h, v)
            self.g.add_edge(SOURCE, h)


    def add_hyperedge(self, edges: list[tuple[int, int, int]], degree: int):
        h = HyperEdge(degree=degree, edges=edges, score=None)
        self.g.add_edge(SOURCE, h)
        for (u, v, k) in edges:
            self.g.add_edge(h, u)
            self.g.add_edge(h, v)

    def merge(self, a: HyperEdge, b: HyperEdge, node: int):
        if a.edges[-1][1] != node:
            a.reverse()
        if b.edges[0][0] != node:
            b.reverse()
        d = max(a.degree, b.degree)
        s_merge = a.edges + b.edges
        self.add_hyperedge(edges=s_merge, degree=d)

        self.g.remove_node(a)
        self.g.remove_node(b)

    def split(self, h: HyperEdge, node: int):
        i_split = [i for i, edge in enumerate(h.edges) if edge[1] == node][0]
        sa = h.edges[: i_split + 1]
        sb = h.edges[i_split + 1 :]
        self.add_hyperedge(degree=h.degree, edges=sa)
        self.add_hyperedge(degree=h.degree, edges=sb)
        self.g.remove_node(h)

    def overlap(self, a: HyperEdge, b: HyperEdge, edge):
        """
        we suppose that hyper-edge `ha` starts or ends with u
        where (u, v) = edge
        """
        (u, v, k) = edge
        if a.edges[-1][1] != u:
            a.reverse()
        a.edges.append(edge)
        # we remove it and add it again to update the "hyperedge -> node" information
        self.add_hyperedge(edges=a.edges, degree=a.degree)
        self.g.remove_node(a)

    def dissociate(self, a: HyperEdge, b: HyperEdge, edge):
        """
        We suppose that the hyper-edge starts with (v, u) or ends with (u, v)
        where (u, v) = edge
        """
        (u, v, k) = edge
        assert (u, v, k) in a.edges or (v, u, k) in b.edges
        if a.edges[-1] != (u, v, k):
            a.reverse()
        a.edges.pop()
        self.add_hyperedge(edges=a.edges, degree=a.degree)
        self.g.remove_node(a)

    def sample_t(self):
        a = np.random.choice(self.g.succ[SOURCE])
        x, y = [a.edges[0][0], a.edges[-1][1]]
        if self.g.degree[x] > self.g.degree[y]:
            node = x
        else:
            node = y
        b = np.random.choice(self.g.pred[node])
        if a == b:
            raise SampleError("a = b in sample")
        return (a, b, node)

    def increase_degree(self, h: HyperEdge):
        d = h.degree
        assert d < 3
        h.degree = d + 1
        h.score = None

    def decrease_degree(self, h: HyperEdge):
        d = h.degree
        assert d > 2
        h.degree = d - 1
        h.score = None

    def score(self):
        ...

    def do_perturbation(self, action):
        ...

    def undo_perturbation(self, action):
        ...

    def try_get_random_perturbation(self):
        choice = np.random.choice(range(6), p=CHOICE_DISTRIBUTION)
        a, b, node = self.sample_t()
        if choice == INCREASE_DEGREE:
            if a.degree >= 3:
                raise SampleError(f"degree too high to increase\n a={a}")
            return (choice, a)
        if choice == DECREASE_DEGREE:
            if a.degree <= 1:
                raise SampleError(f"Degree too low to decrease\na={a}")
            return (choice, a)
        if choice == OVERLAP:
            candidates = [edge for edge in b.edges if node in edge]
            # FIXME: np.random.choice does not work
            edge = random.choice(candidates)
            return (choice, a, b, edge)
        if choice == DISSOCIATE:
            if a.edges[0][0] != node:
                a.reverse()
            (u, v, k) = a.edges[0]
            if (u, v, k) not in b.edges and (v, u, k) not in b.edges:
                raise SampleError(f"hypereges do not share an edge for dissociate\na={a}\nb={b}")
            return (choice, a, b, (u, v, k))
        if choice == MERGE:
            if b.edges[0][0] != node and b.edges[-1][1] != node:
                raise SampleError(f"hyperedges do not share an extremity for merge\na={a}\nb={b}\nnode={node}")
            return (choice, a, b, node)
        if choice == SPLIT:
            if len(b.edges) < 3:
                    raise SampleError(f"hyperedge is too short for split\nb={b}")
            if b.edges[0][0] == node or b.edges[-1][1] == node:
                raise SampleError(f"node is not at extremity for split\nb={b}")
            return (choice, b, node)

    def random_perturbation(self):
        while True:
            try:
                self.random_perturbation()
                return
