import numpy

class Graph(object):
    """Representation of a directed, unweighted graph"""
    def __init__(self,graph_dict=None):
        if graph_dict is None:
            graph_dict = {}
        self._graph_dict = {}
        for v in graph_dict:
            self._graph_dict[v] = list(graph_dict[v])
    
    def vertices(self):
        return list(self._graph_dict.keys())
    
    def edges(self):
        E = []
        for v1 in self._graph_dict:
            for v2 in self._graph_dict[v1]:
                E.append((v1,v2))
        return E
    
    def add_vertex(self,v):
        if v in self._graph_dict.keys():
            raise Exception('Vertex "{}" already exists'.format(v))
        self._graph_dict[v] = []
    
    def add_edge(self,e):
        v1, v2 = tuple(e)
        if (v1 in self._graph_dict) and (v2 in self._graph_dict):
            self._graph_dict[v1].append(v2)
        else:
            raise Exception('One or more vertices not in graph. Add all vertices first')
    
    def matrix(self):
        vertices = self.vertices()
        M = numpy.matrix([[0]*len(vertices)]*len(vertices))
        for v1,v2 in self.edges():
            M[vertices.index(v1),vertices.index(v2)] += 1
        return M,vertices
