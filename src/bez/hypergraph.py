import networkx as nx
import numpy as np
from enum import IntEnum


class Perturbation(IntEnum):
    INCREASE_DEGREE = 0
    DECREASE_DEGREE = 1
    MERGE = 2
    SPLIT = 3
    OVERLAP = 4
    DISSOCIATE = 5
    REVERSE = 6


SOURCE = "*"

CHOICE_DISTRIBUTION = [
    # change degree
    1 / 8,
    1 / 8,
    # merge or split
    1 / 8,
    1 / 8,
    # overlap or dissociate
    1 / 8,
    1 / 8,
    # reverse edge
    2 / 8,
]

HYPER = 0
EDGE = 1
NODE = 2


class HyperEdge:
    edges: list[tuple[tuple[int], tuple[int], int]]
    pixels: list[tuple[int, int]]
    degree: int
    score: float | None

    def __init__(self, edges, degree, pixels):
        self.edges = edges
        self.degree = degree
        self.pixels = list(pixels)

    def reverse(self):
        self.edges = [(v, u, k) for (u, v, k) in self.edges[::-1]]
        self.pixels.reverse()

    def __repr__(self):
        tokens = []
        tokens.append(f"{self.edges[0][0]}")
        for u, v, k in self.edges:
            tokens.append(f"- [{k}] -> {v}")
        tokens.append(f"( degree = {self.degree}, n_pixels = {len(self.pixels)})")
        return " ".join(tokens)


class SampleError(BaseException): ...


class HyperGraph:
    def __init__(self, topo_graph: nx.MultiGraph):
        self.g = nx.DiGraph()
        self.topo = topo_graph
        self.edge2pixel = {}
        for u, v, key, pixels in topo_graph.edges(keys=True, data="pixels"):
            if tuple(pixels[0]) != u:
                pixels = pixels[::-1]
            assert tuple(pixels[0]) == u
            self.edge2pixel[(u, v, key)] = pixels
            self.edge2pixel[(v, u, key)] = pixels[::-1]
            d = min(len(pixels) - 1, 3)
            h = HyperEdge(degree=d, edges=[(u, v, key)], pixels=pixels)
            self.g.add_edge(h, tuple(u))
            self.g.add_edge(h, tuple(v))
            self.g.add_edge(SOURCE, h)

    def all_hyperedges(self):
        return self.g.succ[SOURCE]

    def create_hyperedge(self, edges, degree):
        pixels = []
        pixels.append(edges[0][0])
        for u, v, k in edges:
            assert tuple(pixels[-1]) == u
            pixels.extend(self.edge2pixel[(u, v, k)][1:])
        return HyperEdge(edges=edges, degree=degree, pixels=pixels)

    def perturbate(
        self, old_hyperedges: list[HyperEdge], new_hyperedges: list[HyperEdge]
    ):
        for old in old_hyperedges:
            self.g.remove_node(old)
        for new in new_hyperedges:
            self.g.add_edge(SOURCE, new)
            for u, v, k in new.edges:
                self.g.add_edge(new, tuple(u))
                self.g.add_edge(new, tuple(v))

    def merge(self, a: HyperEdge, b: HyperEdge):
        assert a.edges[-1][1] == b.edges[0][0]
        d = max(a.degree, b.degree)
        return self.create_hyperedge(a.edges + b.edges, d)

    def split(self, h: HyperEdge, node: int):
        i_split = [i for i, (u, v, k) in enumerate(h.edges) if u == node][0]
        sa = h.edges[:i_split]
        sb = h.edges[i_split:]
        a = self.create_hyperedge(sa, h.degree)
        b = self.create_hyperedge(sb, h.degree)
        return (a, b)

    def overlap(self, a: HyperEdge, b: HyperEdge):
        node = a.edges[-1][1]
        (u, v, k) = [(u, v, k) for (u, v, k) in b.edges if u == node][0]
        sa2 = a.edges + [(u, v, k)]
        return self.create_hyperedge(sa2, a.degree)

    def dissociate(self, a: HyperEdge, b: HyperEdge):
        (u, v, k) = a.edges[-1]
        assert (u, v, k) in b.edges
        assert len(a.edges) >= 2
        sa2 = a.edges[:-1]
        return self.create_hyperedge(sa2, a.degree)
        self.g.remove_node(a)

    def sample_t(self):
        a = np.random.choice(self.g.succ[SOURCE])
        node = a.edges[-1][1]
        b = np.random.choice(self.g.pred[node])
        return (a, b, node)

    def increase_degree(self, h: HyperEdge):
        d = h.degree
        assert d < 3
        return HyperEdge(edges=h.edges, degree=d + 1, pixels=h.pixels)

    def decrease_degree(self, h: HyperEdge):
        d = h.degree
        assert d > 1
        return HyperEdge(edges=h.edges, degree=d - 1, pixels=h.pixels)

    def try_propose_random_perturbation(self):
        choice = np.random.choice(Perturbation, p=CHOICE_DISTRIBUTION)
        choice = Perturbation(choice)
        a, b, node = self.sample_t()
        if choice == Perturbation.REVERSE:
            a.reverse()
            return choice, [], []
        if choice == Perturbation.INCREASE_DEGREE:
            if a.degree >= 3:
                raise SampleError(f"degree too high to increase\n a={a}")
            return choice, [a], [self.increase_degree(a)]
        if choice == Perturbation.DECREASE_DEGREE:
            if a.degree <= 1:
                raise SampleError(f"Degree too low to decrease\na={a}")
            return choice, [a], [self.decrease_degree(a)]
        if choice == Perturbation.SPLIT:
            if len(b.edges) < 2:
                raise SampleError(f"hyperedge is too short for split\nb={b}")
            if b.edges[0][0] == node or b.edges[-1][1] == node:
                raise SampleError(f"node is at extremity for split\nb={b}")
            return choice, [b], list(self.split(b, node))
        if choice == Perturbation.MERGE:
            if b.edges[0][0] != node:
                raise SampleError(
                    f"hyperedges do not share an extremity for merge\na={a}\nb={b}\nnode={node}"
                )
            if a == b:
                raise SampleError(f"This merge would create a cycle\na={a}")
            return choice, [a, b], [self.merge(a, b)]
        if choice == Perturbation.OVERLAP:
            if b.edges[-1][1] == node:
                raise SampleError(
                    f"overlap impossible: node is at the end of b\na={a}\nb={b}\nnode={node}"
                )
            return choice, [a], [self.overlap(a, b)]
        if choice == Perturbation.DISSOCIATE:
            (u, v, k) = a.edges[-1]
            if a == b:
                raise SampleError(f"a = b for dissociate")
            if len(a.edges) < 2:
                raise SampleError(
                    f"hyperedge is too short for dissociate\na={a}\nb={b}"
                )
            if (u, v, k) not in b.edges:
                raise SampleError(
                    f"hypereges do not share an edge for dissociate\na={a}\nb={b}"
                )
            return choice, [a], [self.dissociate(a, b)]

    def propose_random_perturbation(self):
        while True:
            try:
                return self.try_propose_random_perturbation()
            except SampleError as e:
                pass
