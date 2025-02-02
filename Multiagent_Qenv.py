import gymnasium as gym
from Graph import Graph
from Uav import Uav
from Node import Node
from typing import Any, Tuple
import logging
import jax.numpy as jnp
from AoiUser import AoiUser

class Multiagent_Qenv(gym.Env):
	"""
	Environment for Q-learning algorithm.
	Includes the Graph and the Uav interaction with the Nodes.
	Also includes the submodular game that the users play within the nodes.
	"""
	
	def __init__(self, graph: Graph, uavs: list[Uav], number_of_users: list, convergence_threshold: float, n_actions: float,
				n_observations: float, solving_method: str, T: float, c: float, b: float, max_iter: int) -> None:
		"""
		Initialize the environment.
		
		Parameters:
			graph (Graph): The graph of the environment.
			uav (Uav): The UAV in the environment.
			number_of_users (list): The number of users in each node.
			convergence_threshold (float): The threshold for convergence.
			n_actions (float): The number of actions.
			n_observations (float): The number of observations.
			solving_method (str): The method to use for solving the game.
		"""
		self.graph = graph
		self.uavs = uavs
		self.number_of_uavs = len(uavs)
		self.number_of_users = number_of_users
		self.convergence_threshold = convergence_threshold
		self.n_actions = n_actions
		self.n_observations = n_observations
		self.solving_method = solving_method
		self.T = T
		self.done = False
		self.c = c
		self.b = b
		self.max_iter = max_iter
		
		#self.action_space = gym.spaces.Discrete(n_actions)
		self.action_space = gym.spaces.MultiDiscrete([n_actions] * self.number_of_uavs)
		self.observation_space = gym.spaces.MultiDiscrete([n_observations] * self.number_of_uavs)
		
		self.reset(uavs= self.uavs, graph= graph)
		#logging.info("Environment has been successfully initialized!")
	
	def step(self, action: list) -> Tuple[Node, float, bool, dict]:
		"""
		Perform an action in the environment and calculate the reward.
		
		Parameters:
			action (list): The actions to perform.

		Returns:
			observation (Node): The new observation.
			reward (float): The reward for the action.
			done (bool): Whether the episode is done.
			info (dict): Additional information.
		"""

		info = {}
		
		# Initialize the reward
		temp_reward = jnp.zeros(len(self.uavs))
	
		# Create an empty list with lists to store the visited nodes of each UAV
		visited_nodes = [[] for i in range(len(self.uavs))]

		# Perform the action for all uavs
		observation_list = []
		for i in range(len(self.uavs)):
			self.uavs[i].travel_to_node(action[i])
			observation_list.append(self.uavs[i].get_current_node())
   
		self.observation = observation_list
		
		for i in range(len(self.uavs)):
  
			# Start playing the game inside the current node
			done_game = False
			temp_U = self.number_of_users[0][self.uavs[i].get_current_node().get_node_id()]
			user_strategies = jnp.ones(temp_U) * 0.1  # Strategies for all users
			uav_bandwidth = self.uavs[i].get_uav_bandwidth()
			uav_cpu_frequency = self.uavs[i].get_cpu_frequency()
			uav_total_data_processing_capacity = self.uavs[i].get_total_data_processing_capacity()

			user_channel_gains = jnp.zeros(temp_U)
			user_transmit_powers = jnp.zeros(temp_U)
			user_data_in_bits = jnp.zeros(temp_U)

			for idx, user in enumerate(self.uavs[i].get_current_node().get_user_list()):
				# Assing the channel gain and transmit power to the user
				user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
				user_transmit_powers = user_transmit_powers.at[idx].set(user.get_transmit_power())
				user_data_in_bits = user_data_in_bits.at[idx].set(user.get_user_bits())
				user.set_user_strategy(user_strategies[idx])
			
			# Play the submodular game
			c = self.c
			b = self.b
				
			iteration_counter = 0
			while(not done_game):
				
				iteration_counter += 1
				previous_strategies = user_strategies

				# Iterate over users and pass the other users' strategies
				for idx, user in enumerate(self.uavs[i].get_current_node().get_user_list()):
					# Exclude the current user's strategy
					#print("Playing game with user: ", idx)
					
					# Exclude the current user's strategy
					other_user_strategies = jnp.concatenate([user_strategies[:idx], user_strategies[idx+1:]])
					
					# Exclude the current user's channel gain, transmit power and data in bits
					other_user_channel_gains = jnp.concatenate([user_channel_gains[:idx], user_channel_gains[idx+1:]])
					other_user_transmit_powers = jnp.concatenate([user_transmit_powers[:idx], user_transmit_powers[idx+1:]])
					other_user_data_in_bits = jnp.concatenate([user_data_in_bits[:idx], user_data_in_bits[idx+1:]])
					
					
					if self.solving_method == "cvxpy":
						# Play the submodular game
						maximized_utility, percentage_offloaded = user.play_submodular_game_cvxpy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
																								uav_cpu_frequency, uav_total_data_processing_capacity, self.T, self.uavs[i].get_current_coordinates(), self.uavs[i].get_height())
						
						#logging.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
						#logging.info("User %d has maximized its utility to %s", idx, maximized_utility)
						
						# Update the user's strategy
						user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
						
						# Update user's channel gain
						user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
						
					elif self.solving_method == "scipy":
						# Play the submodular game
						maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
																								uav_cpu_frequency, uav_total_data_processing_capacity, self.T, self.uavs[i].get_current_coordinates(), self.uavs[i].get_height())

						# #logging.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
						# #logging.info("User %d has maximized its utility to %s", idx, maximized_utility)
						
						# Update the user's strategy
						user_strategies = user_strategies.at[idx].set(percentage_offloaded[0])
						
						# Update user's channel gain
						user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())

				# Check how different the strategies are from the previous iteration    
				strategy_difference = jnp.linalg.norm(user_strategies - previous_strategies)
				
				# Check if the strategies have converged
				if strategy_difference < self.convergence_threshold:
					# Calculate the consumed energy for all users based on the strategies they have chosen and adjust the energy
					# for idx, user in enumerate(uav.get_current_node().get_user_list()):
					#     user.calculate_consumed_energy()
					#     user.adjust_energy(user.get_current_consumed_energy())
					
					# Adjust UAV energy for processing the offloaded data
					self.uavs[i].energy_to_process_data(energy_coefficient= 0.1)
					
					done_game = True
		
				if iteration_counter > 19:
					#logging.info("The game has not converged after 100 iterations!")
					done_game = True
					
			self.uavs[i].set_finished_business_in_node(True)
			self.uavs[i].hover_over_node(time_hover= self.T)
			#logging.info("The UAV has finished its business in the current node")
			#logging.info("The strategies at node %s have converged to: %s", self.uav.get_current_node().get_node_id(), user_strategies)
			# Log the task_intensity of the users
			task_intensities = []
			for user in self.uavs[i].get_current_node().get_user_list():
				task_intensities.append(user.get_task_intensity())
			#logging.info("The task intensities of the users at node %s are: %s", self.uav.get_current_node().get_node_id(), task_intensities)
			#logging.info("Converged with strategy difference: %s in %d iterations", strategy_difference, iteration_counter)
					
			# Get the strategies of the users and calculate the new remaining data for the users
			for idx, user in enumerate(self.uavs[i].get_current_node().get_user_list()):
				user.set_user_strategy(user_strategies[idx])
				user.calculate_remaining_data() # Calculate the remaining data for the user
			self.uavs[i].get_current_node().calculate_total_bit_data()
			
			# Calculate total data in bits processed by the UAV in this node
			total_data_processed = 0
			for user in self.uavs[i].get_current_node().get_user_list():
				total_data_processed += user.get_current_strategy() * user.get_user_bits()
			
			self.uavs[i].update_total_processed_data(total_data_processed)

			# Calculate the reward
			temp_reward = temp_reward.at[i].set(total_data_processed)

			# Divide the reward by the number of uavs that are on the same node as this uav
			uavs_in_node = 0
			for j in range(len(self.uavs)):
				if (self.uavs[j].get_current_node().get_node_id() == self.uavs[i].get_current_node().get_node_id()):
					uavs_in_node += 1

			temp_reward = temp_reward.at[i].set(temp_reward[i] / uavs_in_node)
		
			self.reward = self.reward.at[i].add(temp_reward[i])
   
			# Update the visited nodes for each UAV
			visited_nodes[i] = self.uavs[i].get_visited_nodes()
  
		info["visited_nodes"] = visited_nodes

		# Check UAV situation
		available_uavs = []

		# Check if the UAV has reached the final node to end the episode
		for i in range(len(self.uavs)):
			if (self.uavs[i].get_current_node().get_node_id() != self.uavs[i].get_final_node().get_node_id()):
				#logging.info("The UAV has reached the final node!")
				available_uavs.append(self.uavs[i])
			else:
				#print("UAV with id: ", self.uavs[i].get_uav_id(), " has reached the final node!")
				pass
			
		remaining_uavs = []
		# Check if a UAV has run out of energy
		for i in range(len(available_uavs)):
			if (self.uavs[i].get_energy_level() <= 0):
				#print("UAV with id: ", self.uavs[i].get_uav_id(), " has run out of energy!")
				pass
			else:
				remaining_uavs.append(available_uavs[i])
			
		available_uavs = remaining_uavs
  
		remaining_uavs = []
		# If the UAV has exceeded max_iter actions, end the episode
		for i in range(len(available_uavs)):
			if (self.uavs[i].get_number_of_actions() >= self.max_iter):
				#print("UAV with id: ", self.uavs[i].get_uav_id(), " has exceeded the maximum number of actions!")
				pass
			else:
				remaining_uavs.append(available_uavs[i])
		
		available_uavs = remaining_uavs

		if len(available_uavs) == 0:
			self.done = True
			self.uavs = available_uavs
			#logging.info("The episode has ended!")
		else:
			self.uavs = available_uavs

		info["Available UAVs"] = self.uavs
		
		return (self.observation, self.reward, self.done, info)
	
	def reset(self, uavs: list, graph: Graph) -> Node:
		"""
		Reset the environment to the initial state.
		"""
		self.uavs = uavs
		self.graph = graph
				
		# if (self.uav.get_current_node() != self.uav.get_initial_node()):
		# 	print("STOP THE INITIAL NODE IS NOT THE CURRENT NODE") 
		
		# Perform the action for all uavs
		observation_list = []
		for i in range(self.number_of_uavs):
			observation_list.append(self.uavs[i].get_current_node())
   
		self.observation = observation_list
  
		self.reward = jnp.zeros(self.number_of_uavs)
		self.done = False
		
		return self.observation