from Uav import Uav
from Graph import Graph
import logging
import jax
import jax.numpy as jnp

class Algorithms:
    """
    Class that contains the algorithms to be used by the UAV to navigate through the graph
    """

    def __init__(self, number_of_users: float, number_of_nodes: float, uav: Uav, graph: Graph, key: jax.random.PRNGKey, convergence_threshold: float = 1e-3)->None:
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
        self.number_of_users = number_of_users
        self.number_of_nodes = number_of_nodes
        self.graph = graph
        self.saved_graph_version = graph
        self.key = key
        self.uav = uav
        self.saved_uav_version = uav
        
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
            logging.info(f"Node {node.get_node_id()} has {node.get_node_total_data()} bits of data")
        return sorted_nodes
    
    def reset(self)->None:
        """
        Reset the algorithm to the initial state
        """
        self.uav = self.saved_uav_version
        self.graph = self.saved_graph_version   
        
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
                            
                            logging.info(f"User {idx} has offloaded {percentage_offloaded[0]} of its data")
                            logging.info(f"User {idx} has maximized its utility to {maximized_utility}")
                            
                            # Update the user's strategy
                            user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
                            
                            # Update user's channel gain
                            user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
                            
                        elif solving_method == "scipy":
                            # Play the submodular game
                            maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
                                                                                                    uav_cpu_frequency, uav_total_data_processing_capacity, 2, uav.get_current_coordinates(), uav.get_height())
    
                            logging.info(f"User {idx} has offloaded {percentage_offloaded[0]} of its data")
                            logging.info(f"User {idx} has maximized its utility to {maximized_utility}")
                            
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
                logging.info(f"The strategies at node {uav.get_current_node().get_node_id()} have converged to: {user_strategies}")
                # Log the task_intensity of the users
                task_intensities = []
                for user in uav.get_current_node().get_user_list():
                    task_intensities.append(user.get_task_intensity())
                logging.info(f"The task intensities of the users at node {uav.get_current_node().get_node_id()} are: {task_intensities}")
                logging.info(f"Converged with strategy difference: {strategy_difference} in {iteration_counter} iterations")
                
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
            
        logging.info(f"The UAV trajectory is: {trajectory_ids}")
        
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
                            
                            logging.info(f"User {idx} has offloaded {percentage_offloaded[0]} of its data")
                            logging.info(f"User {idx} has maximized its utility to {maximized_utility}")
                            
                            # Update the user's strategy
                            user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
                            
                            # Update user's channel gain
                            user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
                            
                        elif solving_method == "scipy":
                            # Play the submodular game
                            maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
                                                                                                    uav_cpu_frequency, uav_total_data_processing_capacity, 2, uav.get_current_coordinates(), uav.get_height())
    
                            logging.info(f"User {idx} has offloaded {percentage_offloaded[0]} of its data")
                            logging.info(f"User {idx} has maximized its utility to {maximized_utility}")
                            
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
                logging.info(f"The strategies at node {uav.get_current_node().get_node_id()} have converged to: {user_strategies}")
                # Log the task_intensity of the users
                task_intensities = []
                for user in uav.get_current_node().get_user_list():
                    task_intensities.append(user.get_task_intensity())
                logging.info(f"The task intensities of the users at node {uav.get_current_node().get_node_id()} are: {task_intensities}")
                logging.info(f"Converged with strategy difference: {strategy_difference} in {iteration_counter} iterations")
                
                if (uav_has_reached_final_node):
                    logging.info("The UAV has reached the final node and has finished its business")
                    break_flag = True
                #print(f"Final Strategies: {user_strategies}")
            else:
                # Decide to which node to move next randomly from the ones availalbe that are not visited
                next_node = uav.get_brave_greedy_next_node(nodes= graph.get_nodes(), key= key)
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
            
        logging.info(f"The UAV trajectory is: {trajectory_ids}")
        
        if uav_has_reached_final_node:
            return True
        else:
            return False
        
    def q_brave(self, solving_method:str)->bool:
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
                                
                                logging.info(f"User {idx} has offloaded {percentage_offloaded[0]} of its data")
                                logging.info(f"User {idx} has maximized its utility to {maximized_utility}")
                                
                                # Update the user's strategy
                                user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
                                
                                # Update user's channel gain
                                user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
                                
                            elif solving_method == "scipy":
                                # Play the submodular game
                                maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
                                                                                                        uav_cpu_frequency, uav_total_data_processing_capacity, 2, uav.get_current_coordinates(), uav.get_height())
        
                                logging.info(f"User {idx} has offloaded {percentage_offloaded[0]} of its data")
                                logging.info(f"User {idx} has maximized its utility to {maximized_utility}")
                                
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
                    logging.info(f"The strategies at node {uav.get_current_node().get_node_id()} have converged to: {user_strategies}")
                    # Log the task_intensity of the users
                    task_intensities = []
                    for user in uav.get_current_node().get_user_list():
                        task_intensities.append(user.get_task_intensity())
                    logging.info(f"The task intensities of the users at node {uav.get_current_node().get_node_id()} are: {task_intensities}")
                    logging.info(f"Converged with strategy difference: {strategy_difference} in {iteration_counter} iterations")
                    
                    if (uav_has_reached_final_node):
                        logging.info("The UAV has reached the final node and has finished its business")
                        break_flag = True
                    #print(f"Final Strategies: {user_strategies}")
                else:
                    # Decide to which node to move next randomly from the ones availalbe that are not visited
                    next_node = uav.get_brave_greedy_next_node(nodes= graph.get_nodes(), key= key)
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
                
            logging.info(f"The UAV trajectory is: {trajectory_ids}")
            
            if uav_has_reached_final_node:
                return True
            else:
                return False