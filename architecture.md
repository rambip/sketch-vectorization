# Goal

This file contains all the step needed to implement our algorith, along with some implementation details.

# Overview

```
                                                              (3)                               (4)
                                                         ┌───────────┐                     ┌───────────┐
                                                         │           │                     │           │
                                                         ▼           │                     ▼           │
                       ┌────────────────┐            ┌───────────────────┐       ┌────────────┐        │
original image ───────►│  pixel graph   │ ────────►  │ topological graph ├──────►│ hypergraph │ ───────┘
                 (1)   └────────────────┘    (2)     └───────────────────┘       └────────────┘ ─────────►   end result
                                                                                                    (5)

                          nodes = pixels           nodes = clusters of pixels    nodes = same
                       edges = 8-neighbourhood     edges = linked clusters       edges = set of nodes
                                                   +path of pixels
                                                    for each edge

                                                   ⚠️ multigraph
```

The algorithm has 4 main parts:
- 1) image pre-processing and skeletonization
- 2) computing topology
- 3) curve network creation and refinement
- 4) hypergraph optimization
- 5) finalization and svg export

Before we describe all the steps, we will present the data structures for each graph.

# Structure and objects

## Original image
Our original image is a bitmap image in gray level (0-255), as a matrix of integers

## Pixel graph

The pixel graph is a graph $(V_p, E_p)$ where:
- $V_p$ is the set of pixel positions that are in the drawing
- $E_p$ is the set of edges between pixels according to the 8-connexity

In practise, we will only store the boolean pixel matrix.
We will also store a matrix of integers containing the "width" of the line drawn for each pixel.

## Topological graph

The topological graph is a multigraph $(V_t, E_t)$ where:
- $V_t$ is a set of identifiers
- $E_t$ is a family of $(a, b)$ with $a$ and $b$ in $V_t$

We also store for each a sequence of pixels in the pixel graph.

We will store it as an adjacency list, where each edge has an identifier.

```python
import networkx as nx
topo_g = nx.MultiGraph()
topo_g.add_nodes(...)

# chain of pixels
topo_c = ...
chain = [(0, 0), (1, 1), (1, 2)]
key_for_chain = len(topo_c)
topo_c.append(chain)
# we need the key to know what pixels are in this edge.
topo_s.add_edge(a, b, key_for_chain)
```

Rq: it's a multigraph with loops.
Example 1: a loop
```
 XXXX
O    X
 XXXX
```

Example 2: a multigraph. Here, the topological graph nodes have 2 different edges.
```
   XXXX
XXO    OXX
   XXXX
```

## Hypergraph

The hypergraph contains information about how topological edges are arranged together.


Rq: This is not a hypergraph in the mathematical sense. Indeed, if we have a triangle of topological nodes like so
```
     ┌─┐
     │A│
     └─┘
     ╱ ╲
    ╱   ╲
   ╱     ╲
  ╱       ╲
┌─┐       ┌─┐
│B│ ------│C│
└─┘       └─┘
```

Then, the hyperedge (A,B,C) is ambiguous. It could refer to the super-edge A->B->C, B->C->A or C->A->B

For this reason, we will use the term "super-edge" instead of "hyper-edge".

We will represent the hypergraph as a list of sequences of topological nodes.

```


                                   ┌───┐
                                   │|F|│
                                   └───┘
                                   /
                                 /
                            ┌───┐
                            │|C││
       ┌───┐    ┌───┐  /--/ └───┘      ┌───┐    ┌───┐
       │|A│├────┤|B│├─/         \  ----│|D││ -- │|E││
       └───┘    └───┘                  └───┘    └───┘

```

In this situation, if we want to group (A, B, C, F) in the same super-edge and (E, D, C) in another, we just store `[A B C F]` and `[E D C]`

The datastructure needs to implement 4 particular operations:
- merging / spliting
- overlap / dissociation

The operations are explained below

# Steps

## Preprocessing

We use a threshold to transform our image into a binary image.

We use erosion to know the line width at each pixel.

Then, we increase the size of the lines with a circular dilation. To know the right parameter of the dilation, we take the max width.

## Skeleton

We use an existing skeletonization algorithm, from scipy.

## Topology computation

The algorithm is as follows:
1) compute the number of neighbours of each pixel (8-neighbourhood)
2) split the result into 2 subgraphs: one $G_2$ with the pixels with exactly 2 neighbours, and one $G_+$ with the rest
3) compute the connected components in $G_2$. Create a new vertex $v_t$ for each one of these
  - use 4-connexity, see below for why
4) Compute the chains
For this step, we use 8-connexity. We start at any unvisited node of degree 2, and we initiate the procedure "follow chain" on each side.
We use a double-ended queue to store the elements in the right order.
```
follow_chain_right(last_node, node, dqueue):
    while node is degree 2:
        find the neighbour that is not last_node
        node = this neighbourgh
        add node to the righ of dqueue
        last_node = current_node

    return the connected component this node is in

(same for follow_chain_left)
```

Rq: this case could be a problem:
```

 xxxx
 x
 x
```

The pixel at the top left has 2 neighbours, but we would like it to be a node of the topological graph.


## Initial curve network

To fit the bezier square error, we can write it as a function of (P0x, P0y, P1x, P1y, P2x, P2y, P3x, P3y)
This function is quadratic, so we just need to find when it's derivative is equal to 0 for each variable.

## Refined curve network

- to know if we split or not, we compute
$v = \frac{1}{N}\sum (1-\frac{w_p}{2}) \|B^e(t_p) - p \|^2_2$

Where $N$ is the number of pixels in the chain.
Then, we compute $l = \sqrt(v)$ the standard deviation length, and we compare it to 2 pixels.
Rq: 2 pixels seems a high treshold, but it may be due to the fact that the bezier will not fit the points "at the right time".

## Hypergraph

We must define 2 main operations:

*merge*

We want to merge $c = (x_0, ..., x_n)$ with some other.
First, find other $c'$ which starts or ends with $x_n$. Remove it, and add all the elements to $c$, in reverse.

*overlap*

We want to overlap $c = (x_0, ..., x_n)$ with some other.
First, find other $c'$ which contains $x_n$. Look at one of the neighbours in the chain. add this neighbour to $c$
