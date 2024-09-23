from Node import Node

class Uav:
    """
    Class supporting the UAV data structure of SIMPAT-BRAVE PROJECT
    Represents the UAV that will navigate itself in the graph and process user data
    """
    
    def __init__(self, uav_id: int, initial_node_id: int, final_node_id: int, energy_level: float = 100, total_data_processing_capacity: float = 1000)->None:
        """
        Initialize the UAV
        
        Parameters:
        uav_id : int
            ID of the UAV
        energy_level : float
            Energy level of the UAV
        initial_node_id : int
            ID of the initial node
        final_node_id : int
            ID of the final node
        total_data_processing_capacity : float
            Total data processing capacity of the UAV
        """
        self.uav_id = uav_id
        self.energy_level = energy_level
        self.initial_node_id = initial_node_id
        self.final_node_id = final_node_id
        self.total_data_processing_capacity = total_data_processing_capacity
        self.visited_nodes = []
        
        
    def adjust_energy(self, energy_used: float)->None:
        """
        Adjust the energy level of the UAV
        
        Parameters:
        energy : float
            Energy level to be adjusted
        """
        if self.energy_level - energy_used < 0:
            print("Energy level goes under 0 - Action not allowed")
        else:
            self.energy_level -= energy_used
            
    def get_energy_level(self)->float:
        """
        Get the energy level of the UAV
        
        Returns:
        float
            Energy level of the UAV
        """
        return self.energy_level
    
    def check_if_initial_node(self, node: Node)->bool:
        """
        Check if the UAV is in the initial node
        
        Parameters:
        node : Node
            Node to be checked
        
        Returns:
        bool
            True if the UAV is in the initial node, False otherwise
        """
        return node.node_id == self.initial_node_id
    
    def check_if_final_node(self, node: Node)->bool:
        """
        Check if the UAV is in the final node
        
        Parameters:
        node : Node
            Node to be checked
        
        Returns:
        bool
            True if the UAV is in the final node, False otherwise
        """
        return node.node_id == self.final_node_id
    
    def update_visited_nodes(self, node: Node)->None:
        """
        Update the list of visited nodes
        
        Parameters:
        node : Node
            Node to be added to the list of visited nodes
        """
        self.visited_nodes.append(node.get_node_id())

        
    def __str__(self)->str:
        return f"UAV ID: {self.uav_id}, Energy Level: {self.energy_level}"