from typing import List
from Node import Node

class Graph:
    """
    Class supporting the Graph data structure of SIMPAT-BRAVE PROJECT
    Represents the graph of the project where the UAV will navigate
    """
    
    def __init__(self, nodes : List[Node], edges : List) -> None:
        """
        Initialize the graph
        
        Parameters:
        nodes : list
            List of nodes in the graph
        edges : list
            List of edges in the graph
        """
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
        
    def get_num_nodes(self)->int:
        """
        Get the number of nodes in the graph
        
        Returns:
        int
            Number of nodes in the graph
        """
        return len(self.nodes)
    
    def get_nodes(self)->List[Node]:
        """
        Get the nodes in the graph
        
        Returns:
        list
            List of nodes in the graph
        """
        return self.nodes
    
    def get_num_edges(self)->int:
        """
        Get the number of edges in the graph
        
        Returns:
        int
            Number of edges in the graph
        """
        return len(self.edges)
    
    def get_num_users(self)->int:
        """
        Get the number of users in the graph
        
        Returns:
        int
            Number of users in the graph
        """
        num_users = 0
        for node in self.nodes:
            num_users += len(node.user_list)
        return num_users
        
    def __str__(self)->str:
        return f"Nodes: {self.nodes}\nEdges: {self.edges}"