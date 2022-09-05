import numpy as np
from park import core
from park.spaces.rng import np_random
from park.utils.directed_graph import DirectedGraph


class EdgeInGraph(core.Space):
    """
    The element of this space is an edge in a DirectedGraph object.
    """
    def __init__(self, graph=None):
        if graph is not None:
            assert type(graph) == DirectedGraph
        self.graph = graph
        self.valid_set = None
        core.Space.__init__(self, 'graph_float32', (), np.float32)

    def update_graph(self, graph):
        assert type(graph) == DirectedGraph
        self.graph = graph

    def update_valid_set(self, valid_set):
        self.valid_set = valid_set

    def sample(self):
        if self.valid_set is None:
            edges = list(self.graph.edges())
        else:
            assert len(self.valid_set) <= self.graph.number_of_edges()
            edges = list(self.valid_set)

        if len(edges) > 0:
            return edges[np_random.randint(len(edges))]
        else:
            return None

    def contains(self, x):
        if self.valid_set is None:
            return self.graph.has_edge(x)
        else:
            if x is None:
                # no valid action exists because
                # valid set is empty
                return len(self.valid_set) == 0
            else:
                return (x in self.valid_set)
