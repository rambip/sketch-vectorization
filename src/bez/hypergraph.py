import numpy as np
from dataclasses import dataclass


@dataclass
class HyperEdgeIndex:
    i: int
    reverse: bool


class HyperGraph:
    def __init__(self, size: int, edges):
        # TODO: degrees
        self.hedges = []
        for a, b in edges:
            self.hedges.append([a, b])
        self.tip_counts = np.zeros((1,))

    def update_counts(self):
        self.tip_counts.fill(0)
        for h in self.hedges:
            if len(h) == 1:
                self.tip_counts[h[0]] += 1
            first = h[0]
            last = h[-1]
            self.tip_counts[first] += 1
            self.tip_counts[last] += 1

    def get_hyperedge(self, x: HyperEdgeIndex):
        result = self.hedges[x.i]
        if x.reverse:
            return result[::-1]
        return result

    def merge(self, a: HyperEdgeIndex, b: HyperEdgeIndex):
        ha = self.get_hyperedge(b)
        hb = self.get_hyperedge(a)
        assert ha[-1] == hb[0]
        new = ha[:-1] + hb[1:]
        self.hedges[a.i] = new
        self.hedges[b.i] = self.hedges.pop()

    def split(self, a: HyperEdgeIndex, node: int):
        ha = self.get_hyperedge(a)
        j = ha.index(node)
        result1 = ha[: j + 1]
        result2 = ha[j:]
        self.hedges[a.i] = result1
        self.hedges.append(result2)

    def overlap(self, a: HyperEdgeIndex, b: HyperEdgeIndex):
        ha = self.get_hyperedge(b)
        hb = self.get_hyperedge(a)
        assert ha[-1] == hb[0]
        assert len(hb) >= 2
        self.hedges[a.i] = ha + [hb[1]]

    def dissociate(self, a: HyperEdgeIndex, b: HyperEdgeIndex):
        ha = self.get_hyperedge(b)
        hb = self.get_hyperedge(a)
        assert ha[-2] == hb[0]
        assert ha[-1] == hb[1]
        self.hedges[a.i] = ha[:-1]

    def random_perturbation(self):
        # Problem: we want to evaluate the energy of the new config.
        # Is it a good idea to generate the perturbation, evaluate and go back ?
        ...
