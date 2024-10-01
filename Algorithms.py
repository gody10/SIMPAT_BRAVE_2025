from Uav import Uav
from Graph import Graph
import logging
import jax
import jax.numpy as jnp
from jax import random
from Utility_functions import generate_node_coordinates
from AoiUser import AoiUser
from Edge import Edge
from Node import Node

class Algorithms:
    """
    Class that contains the algorithms to be used by the UAV to navigate through the graph
    """

    def __init__(self, convergence_threshold: float = 1e-3)->None:
        """
        Initialize the algorithms class
        
        Parameters:
        number_of_users : float
            Number of users in the system
        number_of_nodes : float
            Number of nodes in the system
        key : jax.random.PRNGKey
            Key for random number generation
        convergence_threshold : float
            Convergence threshold for the algorithms
        """
        self.convergence_threshold = convergence_threshold
        self.graph = None
        self.uav = None
        self.number_of_users = None
        self.number_of_nodes = None
        self.key = None
        self.uav_height = None
        self.min_distance_between_nodes = None
        self.node_radius = None
        self.uav_energy_capacity = None
        self.uav_cpu_frequency = None
        self.uav_bandwidth = None
        self.uav_processing_capacity = None
        self.uav_velocity = None
        
    def get_uav(self)->Uav:
        """
        Get the UAV
        
        Returns:
        Uav
            UAV
        """
        return self.uav
    
    def get_graph(self)->Graph:
        """
        Get the graph
        
        Returns:
        Graph
            Graph
        """
        return self.graph
        
    def get_convergence_threshold(self)->float:
        """
        Get the convergence threshold
        
        Returns:
        float
            Convergence threshold
        """
        return self.convergence_threshold
    
    def get_number_of_users(self)->list:
        """
        Get the number of users
        
        Returns:
        list
            Number of users for each node
        """
        return self.number_of_users
    
    def get_number_of_nodes(self)->float:
        """
        Get the number of nodes
        
        Returns:
        float
            Number of nodes
        """
        return self.number_of_nodes
    
    def get_key(self)->jax.random.PRNGKey:
        """
        Get the key
        
        Returns:
        float
            Key
        """
        return self.key
    
    def sort_nodes_based_on_total_bits(self)->list:
        """
        Sort the nodes based on the total bits of data they have
        
        Returns:
        list
            Sorted nodes
        """
        nodes = self.graph.get_nodes()
        sorted_nodes = sorted(nodes, key= lambda x: x.get_node_total_data(), reverse= True)
        logging.info("The nodes have been sorted based on the total bits of data they have")
        for node in sorted_nodes:
            logging.info("Node %s has %s bits of data", node.get_node_id(), node.get_node_total_data())
        return sorted_nodes
    
    def setup_experiment(self, number_of_users: list, number_of_nodes: float, key: jax.random.PRNGKey, uav_height: float, min_distance_between_nodes: float, node_radius: float, uav_energy_capacity: float, 
                         uav_bandwidth: float, uav_processing_capacity: float, uav_cpu_frequency: float, uav_velocity: float)->None:
        """
        Setup the experiment
        
        Parameters:
        number_of_users : float
            Number of users in the system
        number_of_nodes : float
            Number of nodes in the system
        key : jax.random.PRNGKey
            Key for random number generation
        uav_height : float
            Height of the UAV
        min_distance_between_nodes : float
            Minimum distance between nodes
        node_radius : float
            Radius of the node
        """
        self.number_of_users = number_of_users
        self.number_of_nodes = number_of_nodes
        self.key = key
        self.uav_height = uav_height
        self.min_distance_between_nodes = min_distance_between_nodes
        self.node_radius = node_radius
        self.uav_energy_capacity = uav_energy_capacity
        self.uav_bandwidth = uav_bandwidth
        self.uav_processing_capacity = uav_processing_capacity
        self.uav_cpu_frequency = uav_cpu_frequency
        self.uav_velocity = uav_velocity
        
        nodes = []
        for i in range(number_of_nodes):
            # Generate random center coordinates for the node
            node_coords = generate_node_coordinates(key, nodes, min_distance_between_nodes)
            
            users = []
            for j in range(number_of_users[number_of_nodes]):
                # Generate random polar coordinates (r, theta, phi) within the radius of the node
                r = node_radius * random.uniform(random.split(key)[0], (1,))[0]  # distance from the center within radius
                theta = random.uniform(random.split(key)[0], (1,))[0] * 2 * jnp.pi  # azimuthal angle (0 to 2*pi)
                phi = random.uniform(random.split(key)[0], (1,))[0] * jnp.pi  # polar angle (0 to pi)
                
                # Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z)
                x = r * jnp.sin(phi) * jnp.cos(theta)
                y = r * jnp.sin(phi) * jnp.sin(theta)
                z = r * jnp.cos(phi)
                
                # User coordinates relative to the node center
                user_coords = (node_coords[0] + x, node_coords[1] + y, node_coords[2] + z)
                
                users.append(AoiUser(
                    user_id=j,
                    data_in_bits= random.uniform(random.split(key + i + j)[0], (1,))[0] * 1000,
                    transmit_power= random.uniform(random.split(key + i + j)[0], (1,))[0] * 100,
                    energy_level= 4000,
                    task_intensity= random.uniform(random.split(key + i + j)[0], (1,))[0] * 100,
                    carrier_frequency= random.uniform(random.split(key + i + j)[0], (1,))[0] * 100,
                    coordinates=user_coords
                ))
            
            nodes.append(Node(
                node_id=i,
                users=users,
                coordinates=node_coords
            ))
            
        # Create edges between all nodes with random weights
        edges = []
        for i in range(number_of_nodes):
            for j in range(i+1, number_of_nodes):
                edges.append(Edge(nodes[i].user_list[0], nodes[j].user_list[0], random.normal(key, (1,))))
                
        # Create the graph
        self.graph = Graph(nodes= nodes, edges= edges)

        # Get number of nodes and edges
        logging.info("Number of Nodes: %s", self.graph.get_num_nodes())
        logging.info("Number of Edges: %s", self.graph.get_num_edges())
        logging.info("Number of Users: %s", self.graph.get_num_users())

        # Create a UAV
        self.uav = Uav(uav_id= 1, initial_node= nodes[0], final_node= nodes[len(nodes)-1], capacity= self.uav_energy_capacity, total_data_processing_capacity= self.uav_processing_capacity, 
                       velocity= self.uav_velocity, uav_system_bandwidth= self.uav_bandwidth, cpu_frequency= self.uav_cpu_frequency, height= uav_height)
        
    def reset(self)->None:
        """
        Reset the experiment so that it can be run again by the same or a different algorithm
        """
        self.graph = None
        self.uav = None
        self.setup_experiment(number_of_users= self.number_of_users, number_of_nodes= self.number_of_nodes, key= self.key, uav_height= self.uav_height, min_distance_between_nodes= self.min_distance_between_nodes, node_radius= self.node_radius,
                              uav_energy_capacity= self.uav_energy_capacity, uav_bandwidth= self.uav_bandwidth, uav_processing_capacity= self.uav_processing_capacity, uav_cpu_frequency= self.uav_cpu_frequency, uav_velocity= self.uav_velocity)
        
    def run_random_walk_algorithm(self, solving_method: str)->bool:
        """
        Algorithm that makes the UAV navigate through the graph randomly
        Every time it needs to move from one node to another, it will randomly choose the next node from a set of unvisited nodes
            
        Parameters:
        solving_method : str
            Solving method to be used for the submodular game (cvxpy or scipy)
        
        Returns:
            bool
                True if the UAV has reached the final node, False otherwise
        """

        logging.info("Running the Random Walk Algorithm")
        uav = self.get_uav()
        graph = self.get_graph()
        U = self.get_number_of_users()
        key = self.get_key()
        convergence_threshold = self.get_convergence_threshold()
        T = 2
        
        uav_has_reached_final_node = False
        break_flag = False

        while(not break_flag):
            
            if(uav.check_if_final_node(uav.get_current_node())):
                uav_has_reached_final_node = True
            
            # Check if the game has been played in the current node
            if(not uav.get_finished_business_in_node()):
                # Start playing the game inside the current node
                done = False
                temp_U = U[uav.get_current_node().get_node_id()]
                user_strategies = jnp.ones(temp_U) * 0.1  # Strategies for all users
                uav_bandwidth = uav.get_uav_bandwidth()
                uav_cpu_frequency = uav.get_cpu_frequency()
                uav_total_data_processing_capacity = uav.get_total_data_processing_capacity()

                user_channel_gains = jnp.zeros(temp_U)
                user_transmit_powers = jnp.zeros(temp_U)
                user_data_in_bits = jnp.zeros(temp_U)

                for idx, user in enumerate(uav.get_current_node().get_user_list()):
                    # Assing the channel gain and transmit power to the user
                    user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
                    user_transmit_powers = user_transmit_powers.at[idx].set(user.get_transmit_power())
                    user_data_in_bits = user_data_in_bits.at[idx].set(user.get_user_bits())
                    user.set_user_strategy(user_strategies[idx])
                    c = 0.08
                    b = 0.4
                    
                    # Initialize data rate, current strategy, channel gain, time overhead, current consumed energy and total overhead for each user
                    user.calculate_data_rate(uav_bandwidth, user_channel_gains[idx], user_transmit_powers[idx])
                    user.calculate_channel_gain(uav.get_current_coordinates(), uav.get_height())
                    user.calculate_time_overhead(other_user_strategies= user_strategies, other_user_bits= user_data_in_bits, uav_total_capacity= uav_total_data_processing_capacity, uav_cpu_frequency= uav_cpu_frequency)
                    user.calculate_consumed_energy()
                    user.calculate_total_overhead(2)
                    
                iteration_counter = 0
                while(not done):
                    
                    iteration_counter += 1
                    previous_strategies = user_strategies

                    # Iterate over users and pass the other users' strategies
                    for idx, user in enumerate(uav.get_current_node().get_user_list()):
                        # Exclude the current user's strategy
                        #print("Playing game with user: ", idx)
                        
                        # Exclude the current user's strategy
                        other_user_strategies = jnp.concatenate([user_strategies[:idx], user_strategies[idx+1:]])
                        
                        # Exclude the current user's channel gain, transmit power and data in bits
                        other_user_channel_gains = jnp.concatenate([user_channel_gains[:idx], user_channel_gains[idx+1:]])
                        other_user_transmit_powers = jnp.concatenate([user_transmit_powers[:idx], user_transmit_powers[idx+1:]])
                        other_user_data_in_bits = jnp.concatenate([user_data_in_bits[:idx], user_data_in_bits[idx+1:]])
                        
                        
                        if solving_method == "cvxpy":
                            # Play the submodular game
                            maximized_utility, percentage_offloaded = user.play_submodular_game_cvxpy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
                                                                                                    uav_cpu_frequency, uav_total_data_processing_capacity, T, uav.get_current_coordinates(), uav.get_height())
                            
                            logging.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
                            logging.info("User %d has maximized its utility to %s", idx, maximized_utility)
                            
                            # Update the user's strategy
                            user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
                            
                            # Update user's channel gain
                            user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
                            
                        elif solving_method == "scipy":
                            # Play the submodular game
                            maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
                                                                                                    uav_cpu_frequency, uav_total_data_processing_capacity, 2, uav.get_current_coordinates(), uav.get_height())
    
                            logging.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
                            logging.info("User %d has maximized its utility to %s", idx, maximized_utility)
                            
                            # Update the user's strategy
                            user_strategies = user_strategies.at[idx].set(percentage_offloaded[0])
                            
                            # Update user's channel gain
                            user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())

                    # Check how different the strategies are from the previous iteration    
                    strategy_difference = jnp.linalg.norm(user_strategies - previous_strategies)
                    
                    # Check if the strategies have converged
                    if strategy_difference < convergence_threshold:
                        # Calculate the consumed energy for all users based on the strategies they have chosen and adjust the energy
                        for idx, user in enumerate(uav.get_current_node().get_user_list()):
                            user.calculate_consumed_energy()
                            user.adjust_energy(user.get_current_consumed_energy())
                        
                        # Adjust UAV energy for hovering over the node
                        uav.hover_over_node(time_hover= T)
                        
                        # Adjust UAV energy for processing the offloaded data
                        uav.energy_to_process_data(energy_coefficient= 0.1)
                        
                        done = True
                        
                uav.set_finished_business_in_node(True)
                uav.hover_over_node(time_hover= T)
                logging.info("The UAV has finished its business in the current node")
                logging.info("The strategies at node %s have converged to: %s", uav.get_current_node().get_node_id(), user_strategies)
                # Log the task_intensity of the users
                task_intensities = []
                for user in uav.get_current_node().get_user_list():
                    task_intensities.append(user.get_task_intensity())
                logging.info("The task intensities of the users at node %s are: %s", uav.get_current_node().get_node_id(), task_intensities)
                logging.info("Converged with strategy difference: %s in %d iterations", strategy_difference, iteration_counter)
                
                if (uav_has_reached_final_node):
                    logging.info("The UAV has reached the final node and has finished its business")
                    break_flag = True
                #print(f"Final Strategies: {user_strategies}")
            else:
                # Decide to which node to move next randomly from the ones availalbe that are not visited
                next_node = uav.get_random_unvisited_next_node(nodes= graph.get_nodes(), key= key)
                if (next_node is not None):
                    if (uav.travel_to_node(next_node)):
                        logging.info("The UAV has reached the next node")
                        logging.info("The UAV energy level is: %s after going to the next node", uav.get_energy_level())
                        # Check if the UAV has reached the final node
                        if uav.check_if_final_node(uav.get_current_node()):
                            logging.info("The UAV has reached the final node")
                            uav_has_reached_final_node = True
                    else:
                        logging.info("The UAV has not reached the next node because it has not enough energy")
                        break_flag = True
                else:
                    logging.info("The UAV has visited all the nodes")
                    break_flag = True
        trajectory = uav.get_visited_nodes()
        trajectory_ids = []
        for node in trajectory:
            trajectory_ids.append(node.get_node_id())
            
        logging.info("The UAV trajectory is: %s", trajectory_ids)
        
        if uav_has_reached_final_node:
            return True
        else:
            return False
        
    def brave_greedy(self, solving_method:str)->bool:
        """
        Algorithm that makes the UAV navigate through the graph by selecting the Area of Interest (AoI) with the highest amount of data to offload
        Every time it needs to move from one node to another, it will choose the node that has the highest amount of data to offload
            
        Parameters:
        solving_method : str
            Solving method to be used for the submodular game (cvxpy or scipy)
        
        Returns:
            bool
                True if the UAV has reached the final node, False otherwise
        """
    
        logging.info("Running the Brave Greedy Algorithm")
        uav = self.get_uav()
        graph = self.get_graph()
        U = self.get_number_of_users()
        convergence_threshold = self.get_convergence_threshold()
        T = 2
        
        uav_has_reached_final_node = False
        break_flag = False

        while(not break_flag):
            
            if(uav.check_if_final_node(uav.get_current_node())):
                uav_has_reached_final_node = True
            
            # Check if the game has been played in the current node
            if(not uav.get_finished_business_in_node()):
                # Start playing the game inside the current node
                done = False
                temp_U = U[uav.get_current_node().get_node_id()]
                user_strategies = jnp.ones(temp_U) * 0.1  # Strategies for all users
                uav_bandwidth = uav.get_uav_bandwidth()
                uav_cpu_frequency = uav.get_cpu_frequency()
                uav_total_data_processing_capacity = uav.get_total_data_processing_capacity()

                user_channel_gains = jnp.zeros(temp_U)
                user_transmit_powers = jnp.zeros(temp_U)
                user_data_in_bits = jnp.zeros(temp_U)

                for idx, user in enumerate(uav.get_current_node().get_user_list()):
                    # Assing the channel gain and transmit power to the user
                    user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
                    user_transmit_powers = user_transmit_powers.at[idx].set(user.get_transmit_power())
                    user_data_in_bits = user_data_in_bits.at[idx].set(user.get_user_bits())
                    user.set_user_strategy(user_strategies[idx])
                    
                c = 0.08
                b = 0.4
                    
                iteration_counter = 0
                while(not done):
                    
                    iteration_counter += 1
                    previous_strategies = user_strategies

                    # Iterate over users and pass the other users' strategies
                    for idx, user in enumerate(uav.get_current_node().get_user_list()):
                        # Exclude the current user's strategy
                        #print("Playing game with user: ", idx)
                        
                        # Exclude the current user's strategy
                        other_user_strategies = jnp.concatenate([user_strategies[:idx], user_strategies[idx+1:]])
                        
                        # Exclude the current user's channel gain, transmit power and data in bits
                        other_user_channel_gains = jnp.concatenate([user_channel_gains[:idx], user_channel_gains[idx+1:]])
                        other_user_transmit_powers = jnp.concatenate([user_transmit_powers[:idx], user_transmit_powers[idx+1:]])
                        other_user_data_in_bits = jnp.concatenate([user_data_in_bits[:idx], user_data_in_bits[idx+1:]])
                        
                        
                        if solving_method == "cvxpy":
                            # Play the submodular game
                            maximized_utility, percentage_offloaded = user.play_submodular_game_cvxpy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
                                                                                                    uav_cpu_frequency, uav_total_data_processing_capacity, T, uav.get_current_coordinates(), uav.get_height())
                            
                            logging.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
                            logging.info("User %d has maximized its utility to %s", idx, maximized_utility)
                            
                            # Update the user's strategy
                            user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
                            
                            # Update user's channel gain
                            user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
                            
                        elif solving_method == "scipy":
                            # Play the submodular game
                            maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
                                                                                                    uav_cpu_frequency, uav_total_data_processing_capacity, 2, uav.get_current_coordinates(), uav.get_height())
    
                            logging.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
                            logging.info("User %d has maximized its utility to %s", idx, maximized_utility)
                            
                            # Update the user's strategy
                            user_strategies = user_strategies.at[idx].set(percentage_offloaded[0])
                            
                            # Update user's channel gain
                            user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())

                    # Check how different the strategies are from the previous iteration    
                    strategy_difference = jnp.linalg.norm(user_strategies - previous_strategies)
                    
                    # Check if the strategies have converged
                    if strategy_difference < convergence_threshold:
                        # Calculate the consumed energy for all users based on the strategies they have chosen and adjust the energy
                        for idx, user in enumerate(uav.get_current_node().get_user_list()):
                            user.calculate_consumed_energy()
                            user.adjust_energy(user.get_current_consumed_energy())
                            
                        # Adjust UAV energy for hovering over the node
                        uav.hover_over_node(time_hover= T)
                        
                        # Adjust UAV energy for processing the offloaded data
                        uav.energy_to_process_data(energy_coefficient= 0.1)
                        
                        done = True
                        
                uav.set_finished_business_in_node(True)
                uav.hover_over_node(time_hover= T)
                logging.info("The UAV has finished its business in the current node")
                logging.info("The strategies at node %s have converged to: %s", uav.get_current_node().get_node_id(), user_strategies)
                # Log the task_intensity of the users
                task_intensities = []
                for user in uav.get_current_node().get_user_list():
                    task_intensities.append(user.get_task_intensity())
                logging.info("The task intensities of the users at node %s are: %s", uav.get_current_node().get_node_id(), task_intensities)
                logging.info("Converged with strategy difference: %s in %d iterations", strategy_difference, iteration_counter)
                
                if (uav_has_reached_final_node):
                    logging.info("The UAV has reached the final node and has finished its business")
                    break_flag = True
                #print(f"Final Strategies: {user_strategies}")
            else:
                # Decide to which node to move next randomly from the ones availalbe that are not visited
                next_node = uav.get_brave_greedy_next_node(nodes= graph.get_nodes())
                if (next_node is not None):
                    if (uav.travel_to_node(next_node)):
                        logging.info("The UAV has reached the next node")
                        # Check if the UAV has reached the final node
                        if uav.check_if_final_node(uav.get_current_node()):
                            logging.info("The UAV has reached the final node")
                            uav_has_reached_final_node = True
                    else:
                        logging.info("The UAV has not reached the next node because it has not enough energy")
                        break_flag = True
                else:
                    logging.info("The UAV has visited all the nodes")
                    break_flag = True
        trajectory = uav.get_visited_nodes()
        trajectory_ids = []
        for node in trajectory:
            trajectory_ids.append(node.get_node_id())
            
        logging.info("The UAV trajectory is: %s", trajectory_ids)
        
        if uav_has_reached_final_node:
            return True
        else:
            return False
        
    def q_brave(self, solving_method:str)->bool:
        """
        Algorithm that makes the UAV navigate through the graph by using Q-Learning to select the next node to visit
        The decision-making process of the RL agent, i.e., UAV, for data collection and processing is represented using a Markov decision process (MDP)
        characterized by a four-component tuple (S, A, f, r), with S set of states, A set of actions, f : S × A → S state transition function, 
        and r : S × A → R is the reward function, with R denoting the real value reward.
            
        Parameters:
        solving_method : str
            Solving method to be used for the submodular game (cvxpy or scipy)
        
        Returns:
            bool
                True if the UAV has reached the final node, False otherwise
        """

        logging.info("Running the Q-Brave Algorithm")
        uav = self.get_uav()
        graph = self.get_graph()
        U = self.get_number_of_users()
        convergence_threshold = self.get_convergence_threshold()
        T = 2
        
        uav_has_reached_final_node = False
        break_flag = False

        while(not break_flag):
            
            if(uav.check_if_final_node(uav.get_current_node())):
                uav_has_reached_final_node = True
            
            # Check if the game has been played in the current node
            if(not uav.get_finished_business_in_node()):
                # Start playing the game inside the current node
                done = False
                temp_U = U[uav.get_current_node().get_node_id()]
                user_strategies = jnp.ones(temp_U) * 0.1  # Strategies for all users
                uav_bandwidth = uav.get_uav_bandwidth()
                uav_cpu_frequency = uav.get_cpu_frequency()
                uav_total_data_processing_capacity = uav.get_total_data_processing_capacity()

                user_channel_gains = jnp.zeros(temp_U)
                user_transmit_powers = jnp.zeros(temp_U)
                user_data_in_bits = jnp.zeros(temp_U)

                for idx, user in enumerate(uav.get_current_node().get_user_list()):
                    # Assing the channel gain and transmit power to the user
                    user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
                    user_transmit_powers = user_transmit_powers.at[idx].set(user.get_transmit_power())
                    user_data_in_bits = user_data_in_bits.at[idx].set(user.get_user_bits())
                    user.set_user_strategy(user_strategies[idx])
                    
                c = 0.08
                b = 0.4
                    
                iteration_counter = 0
                while(not done):
                    
                    iteration_counter += 1
                    previous_strategies = user_strategies

                    # Iterate over users and pass the other users' strategies
                    for idx, user in enumerate(uav.get_current_node().get_user_list()):
                        # Exclude the current user's strategy
                        #print("Playing game with user: ", idx)
                        
                        # Exclude the current user's strategy
                        other_user_strategies = jnp.concatenate([user_strategies[:idx], user_strategies[idx+1:]])
                        
                        # Exclude the current user's channel gain, transmit power and data in bits
                        other_user_channel_gains = jnp.concatenate([user_channel_gains[:idx], user_channel_gains[idx+1:]])
                        other_user_transmit_powers = jnp.concatenate([user_transmit_powers[:idx], user_transmit_powers[idx+1:]])
                        other_user_data_in_bits = jnp.concatenate([user_data_in_bits[:idx], user_data_in_bits[idx+1:]])
                        
                        
                        if solving_method == "cvxpy":
                            # Play the submodular game
                            maximized_utility, percentage_offloaded = user.play_submodular_game_cvxpy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
                                                                                                    uav_cpu_frequency, uav_total_data_processing_capacity, T, uav.get_current_coordinates(), uav.get_height())
                            
                            logging.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
                            logging.info("User %d has maximized its utility to %s", idx, maximized_utility)
                            
                            # Update the user's strategy
                            user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
                            
                            # Update user's channel gain
                            user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
                            
                        elif solving_method == "scipy":
                            # Play the submodular game
                            maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
                                                                                                    uav_cpu_frequency, uav_total_data_processing_capacity, 2, uav.get_current_coordinates(), uav.get_height())

                            logging.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
                            logging.info("User %d has maximized its utility to %s", idx, maximized_utility)
                            
                            # Update the user's strategy
                            user_strategies = user_strategies.at[idx].set(percentage_offloaded[0])
                            
                            # Update user's channel gain
                            user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())

                    # Check how different the strategies are from the previous iteration    
                    strategy_difference = jnp.linalg.norm(user_strategies - previous_strategies)
                    
                    # Check if the strategies have converged
                    if strategy_difference < convergence_threshold:
                        # Calculate the consumed energy for all users based on the strategies they have chosen and adjust the energy
                        for idx, user in enumerate(uav.get_current_node().get_user_list()):
                            user.calculate_consumed_energy()
                            user.adjust_energy(user.get_current_consumed_energy())
                            
                        # Adjust UAV energy for hovering over the node
                        uav.hover_over_node(time_hover= T)
                        
                        # Adjust UAV energy for processing the offloaded data
                        uav.energy_to_process_data(energy_coefficient= 0.1)
                        
                        done = True
                        
                uav.set_finished_business_in_node(True)
                uav.hover_over_node(time_hover= T)
                logging.info("The UAV has finished its business in the current node")
                logging.info("The strategies at node %s have converged to: %s", uav.get_current_node().get_node_id(), user_strategies)
                # Log the task_intensity of the users
                task_intensities = []
                for user in uav.get_current_node().get_user_list():
                    task_intensities.append(user.get_task_intensity())
                logging.info("The task intensities of the users at node %s are: %s", uav.get_current_node().get_node_id(), task_intensities)
                logging.info("Converged with strategy difference: %s in %d iterations", strategy_difference, iteration_counter)
                
                if (uav_has_reached_final_node):
                    logging.info("The UAV has reached the final node and has finished its business")
                    break_flag = True
                #print(f"Final Strategies: {user_strategies}")
            else:
                # Decide to which node to move next randomly from the ones availalbe that are not visited
                next_node = uav.get_brave_greedy_next_node(nodes= graph.get_nodes())
                if (next_node is not None):
                    if (uav.travel_to_node(next_node)):
                        logging.info("The UAV has reached the next node")
                        # Check if the UAV has reached the final node
                        if uav.check_if_final_node(uav.get_current_node()):
                            logging.info("The UAV has reached the final node")
                            uav_has_reached_final_node = True
                    else:
                        logging.info("The UAV has not reached the next node because it has not enough energy")
                        break_flag = True
                else:
                    logging.info("The UAV has visited all the nodes")
                    break_flag = True
        trajectory = uav.get_visited_nodes()
        trajectory_ids = []
        for node in trajectory:
            trajectory_ids.append(node.get_node_id())
            
        logging.info("The UAV trajectory is: %s", trajectory_ids)
        
        if uav_has_reached_final_node:
            return True
        else:
            return False