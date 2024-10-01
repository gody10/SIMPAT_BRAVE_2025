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
        self.energy_level = capacity #* 11.1 * 3.6
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
    
    def get_final_node(self)->Node:
        """
        Get the final node
        
        Returns:
        Node
            Final node
        """
        return self.final_node
    
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
    
    def get_velocity(self)->float:
        """
        Get the velocity of the UAV
        
        Returns:
        float
            Velocity of the UAV
        """
        return self.velocity
    
    def calculate_time_to_travel(self, node_to_start: Node, node_to_end_up: Node)->float:
        """
        Calculate the time to travel from the current node to the next node
        
        Parameters:
        node : Node
            Node to travel to
        
        Returns:
        float
            Time to travel from the current node to the next node
        """
        distance = jnp.sqrt((node_to_start.get_coordinates()[0] - node_to_end_up.get_coordinates()[0])**2 + (node_to_start.get_coordinates()[1] - node_to_end_up.get_coordinates()[1])**2 + (node_to_start.get_coordinates()[2] - node_to_end_up.get_coordinates()[2])**2)
        return distance / self.get_velocity()
        
    def travel_to_node(self, node: Node)->bool:
        """
        Travel to a node
        
        Parameters:
        node : Node
            Node to travel to
        """
        
        # Log the UAV energy level
        logging.info("UAV Energy Level before travelling: %s", self.energy_level)
        
        if not self.finished_business_in_node:
            logging.info("UAV has to first process the data in the node before moving to another node")
            return False
        
        # Calculate the time to travel from the current node to the next node
        time_travel = self.calculate_time_to_travel(node_to_start= self.get_current_node(), node_to_end_up= node)
        
        # Calculate the Energy wasted to travel from one node to the other
        energy_travel = (308.709 * time_travel) - 0.85
        
        # Adjust the energy level of the UAV
        
        if not self.adjust_energy(energy_travel):
            logging.info("Available energy level: %s - while energy needed: %s", self.energy_level, energy_travel)
            return False
        
        # Log the UAV energy level
        logging.info("UAV Energy Level after travelling: %s", self.energy_level)
        
        self.set_current_coordinates(node.get_coordinates())
        self.update_visited_nodes(node)
        print("Updated Visited Nodes")
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
        logging.info("Energy level is now: %s and the energy wasted to hover over the node is: %s", self.energy_level, energy_hover)
        return True

    def energy_to_process_data(self, energy_coefficient: float)->bool:
        """
        Process data
        
        Parameters:
        data : float
            Data to be processed
        """
        # Calculate the total bits based on the strategy that the node's users have chosen
        node_total_bits_based_on_strategy = 0
        for user in self.get_current_node().get_user_list():
            node_total_bits_based_on_strategy += user.get_user_bits() * user.get_user_strategy()
        
        # Calculate the energy wasted to process the data
        energy_process = energy_coefficient * self.get_cpu_frequency() * node_total_bits_based_on_strategy
        
        # Adjust the energy level of the UAV
        if not self.adjust_energy(energy_process):
            logging.info("Energy level goes under 0 - Action not allowed")
            return False
        else:
            logging.info("Energy level is now: %s and the energy wasted to process the data is: %s", self.energy_level, energy_process)
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
        
        # Remove the final node from the list of unvisited nodes if there are more than one unvisited nodes
        if len(unvisited_nodes) > 1:
            unvisited_nodes = [node for node in unvisited_nodes if node.get_node_id() != self.final_node.get_node_id()]
            
        # Check if there arent any unvisited nodes left except the final node
        if not unvisited_nodes:
            return self.get_final_node()
            
        # Calculate the total bits available in each unvisited node
        total_bits = [node.get_node_total_data() for node in unvisited_nodes]
        
        # Calculate the energy needed to travel to each unvisited node
        energy_travel = jnp.array([308.709 * self.calculate_time_to_travel(node_to_start= self.get_current_node(), node_to_end_up= node) - 0.85 for node in unvisited_nodes])
        
        energy_process = jnp.array([node.calculate_total_energy_for_all_user_data_processing() for node in unvisited_nodes])
        
        energy_hover = jnp.array([(4.917 * self.get_height() - 275.204) * 1 for node in unvisited_nodes])
        energy_to_travel_to_final_node = jnp.array([308.709 * self.calculate_time_to_travel(node_to_start= node, node_to_end_up= self.get_final_node()) - 0.85 for node in unvisited_nodes])
        
        # Print all the energies
        print(f"Energy Travel: {energy_travel}")
        print(f"Energy Process: {energy_process}")
        print(f"Energy Hover: {energy_hover}")
        print(f"Energy to Travel to Final Node: {energy_to_travel_to_final_node}")
        print(f"Total Bits: {total_bits}")
        
        # Calculate the total energy needed to travel to each unvisited node
        total_energy = jnp.add(energy_travel, jnp.add(energy_process, energy_hover))
        
        # Remove from unvisited nodes the ones that the energy needed to travel to them and then to the final node is higher than the available energy of the UAV
        unvisited_nodes = [node for i, node in enumerate(unvisited_nodes) if total_energy[i] + energy_to_travel_to_final_node[i] < self.get_energy_level()]
        
        # Check if there arent any unvisited nodes left except the final node
        if not unvisited_nodes:
            return self.get_final_node()
        
        # Generate a random index using jax
        idx = jax.random.randint(key, shape=(), minval=0, maxval=len(unvisited_nodes))
        return unvisited_nodes[idx]
    
    def get_brave_greedy_next_node(self, nodes: list)->Node:
        """
        Get the next node using the Brave Greedy algorithm
        
        Parameters:
        nodes : list
            List of nodes
        
        Returns:
        Node
            Next node using the Brave Greedy algorithm
        """
        next_node = None
        
        # Get all nodes that haven't been visited yet
        unvisited_nodes = [node for node in nodes if node not in self.get_visited_nodes()]
        
        # Remove the final node from the list of unvisited nodes if there are more than one unvisited nodes
        if len(unvisited_nodes) > 1:
            unvisited_nodes = [node for node in unvisited_nodes if node.get_node_id() != self.final_node.get_node_id()]
        
        # Calculate the total bits available in each unvisited node
        total_bits = [node.get_node_total_data() for node in unvisited_nodes]
        
        # Order the nodes by the total bits available
        unvisited_nodes = [node for _, node in sorted(zip(total_bits, unvisited_nodes), key=lambda pair: pair[0], reverse=True)]
        
        # Calculate the energy needed to travel to each unvisited node
        energy_travel = jnp.array([308.709 * self.calculate_time_to_travel(node_to_start= self.get_current_node(), node_to_end_up= node) - 0.85 for node in unvisited_nodes])
        
        energy_process = jnp.array([node.calculate_total_energy_for_all_user_data_processing() for node in unvisited_nodes])
        
        energy_hover = jnp.array([(4.917 * self.get_height() - 275.204) * 1 for node in unvisited_nodes])
        energy_to_travel_to_final_node = jnp.array([308.709 * self.calculate_time_to_travel(node_to_start= node, node_to_end_up= self.get_final_node()) - 0.85 for node in unvisited_nodes])
        
        # Print all the energies
        print(f"Energy Travel: {energy_travel}")
        print(f"Energy Process: {energy_process}")
        print(f"Energy Hover: {energy_hover}")
        print(f"Energy to Travel to Final Node: {energy_to_travel_to_final_node}")
        print(f"Total Bits: {total_bits}")
        
        # Calculate the total energy needed to travel to each unvisited node
        total_energy = jnp.add(energy_travel, jnp.add(energy_process, energy_hover))
            
        # Do the same with enumerate instead of iterate
        chosen_a_node_to_go = False
        for i, node in enumerate(unvisited_nodes):
            if total_energy[i] + energy_to_travel_to_final_node[i] < self.get_energy_level():
                next_node = node
                chosen_a_node_to_go = True
                break
            else:
                print(f"The energy needed to go to the node {node.get_node_id()} is {total_energy[i]} and to go to the final node {energy_to_travel_to_final_node[i]}, but the available energy is {self.get_energy_level()}")
        if not chosen_a_node_to_go:
            return self.get_final_node()

        return next_node
            
        
    def __str__(self)->str:
        return f"UAV ID: {self.uav_id}, Energy Level: {self.energy_level}"