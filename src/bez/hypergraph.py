import networkx as nx
from dataclasses import dataclass
import itertools

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

    def merge(self, ha: int, hb: int, node: int):
        sa = self.table[ha].edges
        sb = self.table[hb].edges
        if self.table[ha].edges[-1][1] != node:
            sa = sa[::-1]
        if self.table[hb].edges[0][0] != node:
            sb = sb[::-1]
        d = max(self.table[ha].degree, self.table[hb].degree)
        s_merge = sa + sb
        h_new = next(self.counter)
        self.table[h_new] = HyperEdge(degree=d, edges=s_merge, score=None)
        del self.table[ha]
        del self.table[hb]
        self.hyper2edge.remove_node(ha)
        self.hyper2edge.remove_node(hb)
        for edge in s_merge:
            self.hyper2edge.add_edge(h_new, edge)
        self.tripartite.edges[(h_new, s_merge[0])]["extremity"] = True
        self.tripartite.edges[(h_new, s_merge[-1])]["extremity"] = True

    def split(self, h0: int, node: int):
        hyper = self.table[h0]
        i_split = hyper.edges.find(lambda edge: edge[1] == node)
        sa = hyper[: i_split + 1]
        sb = hyper[i_split + 1 :]
        h_new_a = next(self.counter)
        h_new_b = next(self.counter)
        self.table[h_new_a] = HyperEdge(edges=sa, degree=hyper.degree, score=None)
        self.table[h_new_b] = HyperEdge(edges=sb, degree=hyper.degree, score=None)
        self.hyper2edge.remove_node(h0)
        for edge in sa:
            self.hyper2edge.add_edge((HYPER, h_new_a), (EDGE, edge))
        self.hyper2edge.edges[(h_new_a, sa[0])]["extremity"] = True
        self.hyper2edge.edges[(h_new_a, sa[-1])]["extremity"] = True
        self.hyper2edge.edges[(h_new_b, sb[0])]["extremity"] = True
        self.hyper2edge.edges[(h_new_b, sb[-1])]["extremity"] = True
        for edge in sb:
            self.tripartite.add_edge((HYPER, h_new_b), (EDGE, edge))

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

    # def random_merge(self):
    #     """
    #     To do a random merge, we need to find one node A such that there are 2 hyperedges with extremity A
    #     """

    # def random_split(self):
    #     """
    #     To do a random split, we need to find any hyperedge containing at least 3 nodes
    #     """

    # def random_overlap(self):
    #     """
    #     To do a ranodm overlap, we need to find 2 hyperedges U, V such that U is completely contained in V
    #     """

    # def random_dissociate(self):
    #     """
    #     To do a random dissociation, we need to find 2 hyperedges such that U is completely contained in V
    #     """

    # def random_perturbation(self):
    #     choice = np.random.choice(range(3), p=CHOICE_DISTRIBUTION)
    #     if choice == INCREASE_DEGREE:
    #         i_h = np.random.choice(self.data.index)
    #         self.increase_degree(i_h)
    #     if choice == DECREASE_DEGREE:
    #         i_h = np.random.choice(self.data.index)
    #         self.decrease_degree(i_h)
    #     if choice == OVERLAP:
    #         self.random_overlap()
    #     if choice == DISSOCIATE:
    #         self.random_dissociate()
    #     if choice == MERGE:
    #         self.random_merge()
    #     if choice == SPLIT:
    #         self.random_split()
