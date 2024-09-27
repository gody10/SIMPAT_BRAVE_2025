from Node import Node
import jax.numpy as jnp
import jax
import logging

class Uav:
    """
    Class supporting the UAV data structure of SIMPAT-BRAVE PROJECT
    Represents the UAV that will navigate itself in the graph and process user data
    """
    
    def __init__(self, uav_id: int, initial_node: Node, final_node: Node, capacity: float = 100, total_data_processing_capacity: float = 1000, velocity : float = 1, uav_system_bandwidth: float = 0.9, cpu_frequency: float = 2, height: float = 100)->None:
        """
        Initialize the UAV
        
        Parameters:
        uav_id : int
            ID of the UAV
        capacity : float
            Total Capacity of the UAV
        initial_node_id : int
            ID of the initial node
        final_node_id : int
            ID of the final node
        total_data_processing_capacity : float
            Total data processing capacity of the UAV
        """
        self.uav_id = uav_id
        self.energy_level = capacity * 11.1 * 3.6
        self.initial_node = initial_node
        self.final_node = final_node
        self.total_data_processing_capacity = total_data_processing_capacity
        self.uav_system_bandwidth = uav_system_bandwidth
        self.velocity = velocity
        self.height = height
        self.cpu_frequency = cpu_frequency
        self.finished_business_in_node = False
        self.visited_nodes = []
        self.update_visited_nodes(self.initial_node)
        self.set_current_coordinates(self.initial_node.get_coordinates())
        
    def set_finished_business_in_node(self, finished: bool)->None:
        """
        Set the finished business in the node
        
        Parameters:
        finished : bool
            True if the business is finished, False otherwise
        """
        self.finished_business_in_node = finished
    
    def get_finished_business_in_node(self)->bool:
        """
        Get the finished business in the node
        
        Returns:
        bool
            True if the business is finished, False otherwise
        """
        return self.finished_business_in_node
        
    def get_total_data_processing_capacity(self)->float:
        """
        Get the total data processing capacity of the UAV
        
        Returns:
        float
            Total data processing capacity of the UAV
        """
        return self.total_data_processing_capacity
        
    def get_height(self)->float:
        """
        Get the height of the UAV
        
        Returns:
        float
            Height of the UAV
        """
        return self.height
        
    def get_uav_bandwidth(self)->float:
        """
        Get the UAV bandwidth
        
        Returns:
        float
            UAV bandwidth
        """
        return self.uav_system_bandwidth
    
    def get_cpu_frequency(self)->float:
        """
        Get the UAV CPU frequency
        
        Returns:
        float
            UAV CPU frequency
        """
        return self.cpu_frequency
        
    def adjust_energy(self, energy_used: float)->bool:
        """
        Adjust the energy level of the UAV
        
        Parameters:
        energy : float
            Energy level to be adjusted
        """
        if self.energy_level - energy_used < 0:
            logging.info("Energy level goes under 0 - Action not allowed")
            return False
        else:
            self.energy_level -= energy_used
            return True
            
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
        return node.node_id == self.initial_node
    
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
        return node.get_node_id() == self.final_node.get_node_id()
    
    def update_visited_nodes(self, node: Node)->None:
        """
        Update the list of visited nodes
        
        Parameters:
        node : Node
            Node to be added to the list of visited nodes
        """
        self.visited_nodes.append(node)
        
    def get_visited_nodes(self)->list:
        """
        Get the list of visited nodes
        
        Returns:
        list
            List of visited nodes
        """
        return self.visited_nodes
    
    def get_current_coordinates(self)->tuple:
        """
        Get the current coordinates of the UAV
        
        Returns:
        tuple
            Current coordinates of the UAV
        """
        return self.current_coordinates
    
    def set_current_coordinates(self, coordinates: tuple)->None:
        """
        Set the current coordinates of the UAV
        
        Parameters:
        coordinates : tuple
            Coordinates to be set
        """
        self.current_coordinates = coordinates
        
    def get_current_node(self)->Node:
        """
        Get the current node of the UAV
        
        Returns:
        Node
            Current node of the UAV
        """
        return self.visited_nodes[-1]
        
    def travel_to_node(self, node: Node)->bool:
        """
        Travel to a node
        
        Parameters:
        node : Node
            Node to travel to
        """
        
        if not self.finished_business_in_node:
            logging.info("UAV has to first process the data in the node before moving to another node")
            return False
        
        # Calculate the distance between the current node and the next node and the time to travel from one to the other
        distance = jnp.sqrt((self.current_coordinates[0] - node.get_coordinates()[0])**2 + (self.current_coordinates[1] - node.get_coordinates()[1])**2 + (self.current_coordinates[2] - node.get_coordinates()[2])**2)
        time_travel = distance / self.velocity
        
        # Calculate the Energy wasted to travel from one node to the other
        energy_travel = (308.709 * time_travel) - 0.85
        
        # Adjust the energy level of the UAV
        
        if not self.adjust_energy(energy_travel):
            logging.info(f"Available energy level: {self.energy_level} - while energy needed: {energy_travel}")
            return False
        
        self.set_current_coordinates(node.get_coordinates())
        self.update_visited_nodes(node)
        self.set_finished_business_in_node(False)
        return True
        
    def hover_over_node(self, time_hover: float)->bool:
        """
        Hover over a node
        
        Parameters:
        time_hover : float
            Time to hover over the node
        height : float
            Height to hover over the node in meters
        """
        # Calculate the energy wasted to hover over the node
        energy_hover = ((4.917 * self.get_height()) - 275.204)*time_hover
        
        # Adjust the energy level of the UAV
        if not self.adjust_energy(energy_hover):
            logging.info("Energy level goes under 0 - Action not allowed")
            return False

    def process_data(self, energy_coefficient: float, cpu_frequency: float, phi: list, strategy: list, bits: list)->bool:
        """
        Process data
        
        Parameters:
        data : float
            Data to be processed
        """
        # Calculate the energy wasted to process the data
        energy_process = energy_coefficient * cpu_frequency * jnp.sum(jnp.multiply(phi, strategy, bits))
        
        # Adjust the energy level of the UAV
        if not self.adjust_energy(energy_process):
            logging.info("Energy level goes under 0 - Action not allowed")
            return False
        else:
            self.set_finished_business_in_node(True)
            
    
            
    def get_random_unvisited_next_node(self, nodes: list, key)->Node:
        """
        Get a random unvisited next node
        
        Parameters:
        nodes : list
            List of nodes
        
        Returns:
        Node
            Random unvisited next node
        """
        # Get all nodes that haven't been visited yet
        unvisited_nodes = [node for node in nodes if node not in self.get_visited_nodes()]
        
        # Check if there are any unvisited nodes left
        if not unvisited_nodes:
            return None
        
        # Generate a random index using jax
        idx = jax.random.randint(key, shape=(), minval=0, maxval=len(unvisited_nodes))
        return unvisited_nodes[idx]
            
        
    def __str__(self)->str:
        return f"UAV ID: {self.uav_id}, Energy Level: {self.energy_level}"