from typing import List
from Node import Node

class Graph:
    """
    Class supporting the Graph data structure of SIMPAT-BRAVE PROJECT
    
    Attributes:
    nodes : list
        List of nodes in the graph
    edges : list
        List of edges in the graph
    """
    
    def __init__(self, nodes : List[Node], edges : List) -> None:
        self.nodes = nodes
        self.edges = edges
    
    def add_node(self, node : Node)->None:
        """
        Add a node to the graph
        
        Parameters:
        node : any
            Node to be added to the graph
        """
        self.nodes.append(node)
        
    def add_edge(self, edge)->None:
        """
        Add an edge to the graph
        
        Parameters:
        edge : tuple
            Edge to be added to the graph
        """
        self.edges.append(edge)
        
    def __str__(self)->str:
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"