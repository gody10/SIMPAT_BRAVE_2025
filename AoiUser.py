import jax.numpy as jnp
import cvxpy as cp
from typing import Tuple
import logging
from scipy.optimize import minimize
import numpy as np
import Node

class AoiUser:
	"""
	Class supporting the User data structure of SIMPAT-BRAVE PROJECT
	Represents a user in an Area of Interest (AOI)
	"""

	def __init__(self, user_id:int, data_in_bits : float, transmit_power: float, energy_level: float, task_intensity: float, coordinates: Tuple = (0,0,0), carrier_frequency: float = 5)->None:
		"""
		Initialize the user
		
		Parameters:
		user_id : int
			ID of the user
		data_in_bits : float
			Amount of data in bits of the user
		transmit_power : float
			Transmit power of the user
		channel_gain : float
			Channel gain of the user
		total_energy_capacity : float
			Total energy capacity of the user
		task_intensity : float
			Task intensity of the user
		"""
		self.user_id = user_id
		self.data_in_bits = data_in_bits
		self.total_bits = data_in_bits
		self.transmit_power = transmit_power
		self.channel_gain = 0
		self.energy_level = energy_level
		self.total_capacity = energy_level
		self.task_intensity = task_intensity
		self.coordinates = coordinates
		self.carrier_frequency = carrier_frequency
		self.white_noise = 1e-9
		self.current_strategy = 0
		self.current_data_rate = 0
		self.current_time_overhead = 0
		self.current_consumed_energy = 0
		self.current_total_overhead = 0
		self.user_utility = 0
		self.distance = 0
		self.other_user_strategies = []
		self.other_user_transmit_powers = []
		self.other_user_bits = []
		self.uav_total_capacity = 0
		self.T = 0

	def get_user_bits(self)->float:
		"""
		Get the amount of data in bits of the user
		
		Returns:
		float
			Amount of data in bits of the user
		"""
		return self.data_in_bits

	def get_total_bits(self)->float:
		"""
		Get the total amount of data in bits of the user
		
		Returns:
		float
			Total amount of data in bits of the user
		"""
		return self.total_bits
	
	def get_total_capacity(self)->float:
		"""
		Get the total capacity of the user
		
		Returns:
		float
			Total capacity of the user
		"""
		return self.total_capacity
	
	def get_carrier_frequency(self)->float:
		"""
		Get the carrier frequency of the user
		
		Returns:
		float
			Carrier frequency of the user
		"""
		return self.carrier_frequency
	
	def get_user_id(self)->int:
		"""
		Get the ID of the user
		
		Returns:
		int
			ID of the user
		"""
		return self.user_id
	
	def get_task_intensity(self)->float:
		"""
		Get the task intensity of the user
		
		Returns:
		float
			Task intensity of the user
		"""
		return self.task_intensity
	
	def get_transmit_power(self)->float:
		"""
		Get the transmit power of the user
		
		Returns:
		float
			Transmit power of the user
		"""
		return self.transmit_power

	def get_user_utility(self)->float:
		"""
		Get the utility of the user
		
		Returns:
		float
			Utility of the user
		"""
		return self.user_utility

	def set_user_utility(self, utility: float)->None:
		"""
		Set the utility of the user
		
		Parameters:
		utility : float
			Utility of the user
		"""
		self.user_utility = utility
	
	def get_channel_gain(self)->float:
		"""
		Get the channel gain of the user
		
		Returns:
		float
			Channel gain of the user
		"""
		return self.channel_gain 
	
	def calculate_remaining_data(self)->None:
		"""
		Calculate the remaining data of the user
		"""
		self.data_in_bits = self.data_in_bits * (1 - self.current_strategy)
	
	def calculate_channel_gain(self, uav_coordinates: Tuple, uav_height: float)->None:
		"""
		Calculate the channel gain of the user
		
		Parameters:
		uav_coordinates : Tuple
			Coordinates of the UAV
		uav_height : float
			Height of the UAV
		
		Returns:
		float
			Channel gain of the user
		"""
		distance = self.get_distance()
		##logging.info("Distance between User %d and UAV is %f", self.get_user_id(), distance)
		pl_loss = 20*jnp.log(distance) + 20 * jnp.log(self.get_carrier_frequency()) + 2*jnp.log((4*jnp.pi)/ 3 * 10*18) + 2
		pl_nloss = 20*jnp.log(distance) + 20 * jnp.log (self.get_carrier_frequency()) + 2*jnp.log((4*jnp.pi)/ 3 * 10*18) + 21
		#theta = jnp.arcsin(uav_height/distance)
		theta = jnp.arcsin(jnp.clip(uav_height / distance, -1, 1))

		pr_loss = 1 / (1 + 0.136 * (jnp.exp(-11.95 * (theta-0.136))))
		pl = pr_loss * pl_loss + (1 - pr_loss) * pl_nloss
		
		self.channel_gain = 1/(10**(pl/10))
		##logging.info("User %d has Channel Gain %f", self.get_user_id(), self.channel_gain)
	
	def get_coordinates(self)->Tuple:
		"""
		Get the coordinates of the user
		
		Returns:
		Tuple
			Coordinates of the user
		"""
		return self.coordinates

	def calculate_distance(self, node: Node)->None:
		"""
		Calculate the distance of the user from the node
  
		Parameters:
			node : Node
  		"""
		self.distance = jnp.sqrt((node.get_coordinates()[0] - self.coordinates[0])**2 + (node.get_coordinates()[1] - self.coordinates[1])**2 + (node.get_coordinates()[2] - self.coordinates[2])**2)
  
	def get_distance(self)->float:
		"""
		Get the distance of the user from the node
		
		Returns:
		float
			Distance of the user from the node
		"""
		return self.distance

	def set_distance(self, distance: float)->None:
		"""
		Set the distance of the user from the node
		
		Parameters:
		distance : float
			Distance of the user from the node
		"""
		self.distance = distance
  
	
	def set_user_strategy(self, strategy: float)->None:
		"""
		Set the strategy of the user
		
		Parameters:
		strategy : float
			Strategy of the user
		"""
		self.current_strategy = strategy
		
	def get_user_strategy(self)->float:
		"""
		Get the strategy of the user
		
		Returns:
		float
			Strategy of the user
		"""
		return self.current_strategy
	
	def adjust_energy(self, energy_used: float)->bool:
		"""
		Adjust the energy level of the UAV
		
		Parameters:
		energy : float
			Energy level to be adjusted
		"""
		if self.energy_level - energy_used < 0:
			#logging.info("Energy required for user %d is greater than the current energy level. The energy_level is %f and the energy_used is %f", self.get_user_id(), self.energy_level, energy_used)
			return False
		else:
			self.energy_level -= energy_used
			return True
		
	def get_current_strategy(self)->float:
		"""
		Get the current strategy of the user
		
		Returns:
		float
			Current strategy of the user
		"""
		return self.current_strategy
		
	def get_current_energy_level(self)->float:
		"""
		Get the current energy level of the user
		
		Returns:
		float
			Current energy level of the user
		"""
		return self.energy_level
	
	def get_current_data_rate(self)->float:
		"""
		Get the current data rate of the user
		
		Returns:
		float
			Current data rate of the user
		"""
		return self.current_data_rate
	
	def get_current_time_overhead(self)->float:
		"""
		Get the current time overhead of the user
		
		Returns:
		float
			Current time overhead of the user
		"""
		return self.current_time_overhead
	
	def get_current_consumed_energy(self)->float:
		"""
		Get the current consumed energy of the user
		
		Returns:
		float
			Current consumed energy of the user
		"""
		return self.current_consumed_energy

	def get_data_offloaded(self)->float:
		"""
		Get the data offloaded by the user
		
		Returns:
		float
			Data offloaded by the user
		"""
		return self.total_bits * self.current_strategy
	
	def get_current_total_overhead(self)->float:
		"""
		Get the current total overhead of the user
		
		Returns:
		float
			Current total overhead of the user
		"""
		return self.current_total_overhead
		
	def calculate_data_rate(self, uav_bandwidth: float, other_users_transmit_powers: list, other_users_channel_gains: list)->None:
		"""
		Calculate the data rate of the user
		
		Parameters:
		uav_bandwidth : float
			Bandwidth of the UAV
		
		Returns:
		float
			Data rate of the user
		"""
		self.current_data_rate = uav_bandwidth* jnp.log(1 + ((self.transmit_power * self.get_channel_gain()) / (self.white_noise + jnp.sum(other_users_transmit_powers * other_users_channel_gains))))
		##logging.info("User %d has Current data rate %f", self.get_user_id(), self.current_data_rate)
	
	def calculate_time_overhead(self, other_user_strategies: list, other_user_bits: list, uav_total_capacity: float, uav_cpu_frequency: float)->None:
		"""
		Calculate the time overhead of the user
		
		Parameters:
		other_user_strategies : list
			List of the strategies of the other users
		other_user_bits : list
			List of the bits of the other users
		uav_total_capacity : float
			Total capacity of the UAV
		uav_cpu_frequency : float
			CPU frequency of the UAV
		"""
		
		data_rate = self.get_current_data_rate()
		denominator = 1 - (jnp.sum(other_user_strategies * other_user_bits) / uav_total_capacity)

		self.current_time_overhead = (
			((self.data_in_bits * self.get_user_strategy()) / data_rate) + 
			((self.get_task_intensity() * self.get_user_strategy() * self.data_in_bits) / (denominator * uav_cpu_frequency))
)
		##logging.info("User %d has Current time overhead %f", self.get_user_id(), self.current_time_overhead)
	
	def calculate_consumed_energy(self)->None:
		"""
		Calculate the consumed energy of the user during the offloading process based on the current strategy
		"""
		data_rate = self.get_current_data_rate()
		self.current_consumed_energy = ((self.get_user_strategy() * self.get_total_bits()) / (data_rate)) * self.get_transmit_power()
		#self.current_consumed_energy = self.get_user_strategy() * self.get_total_bits()
		##logging.info("User %d has Current consumed energy %f", self.get_user_id(), self.current_consumed_energy)
		
	def calculate_total_overhead(self, T: float)->None:
		"""
		Calculate the total overhead of the user during the offloading process over a period T
		
		Parameters:
		T : float
			Time that that timeslot t lasted
		"""
		term1 = self.get_current_time_overhead()/T
		term2 = self.get_current_consumed_energy()/(self.total_capacity)
  
		##logging.info("Term 1: %f, Term 2: %f", term1, term2)
		self.current_total_overhead = term1 + term2
		##logging.info("User %d has Current total overhead %f", self.get_user_id(), self.current_total_overhead)
		
	def play_submodular_game_cvxpy(self, other_people_strategies: list, c: float, b: float, uav_bandwidth: float, other_users_transmit_powers: list, other_users_channel_gains: list, 
								   other_user_data_in_bits: list, uav_cpu_frequency: float, uav_total_data_processing_capacity: float, T: float, uav_coordinates: Tuple, uav_height: float)->float:
		"""
		Define the submodular game that the user will play with the other users
		
		Parameters:
		other_people_strategies : list
			List of the strategies of the other users
		c : float
		"""
		
		# Set as Variable the bits that the user will send to each MEC server
		percentage_offloaded = cp.Variable((1, 1), name = 'percentage_offloaded', nonneg=True)
		
		# Calculate channel gain of the user
		self.calculate_channel_gain(uav_coordinates= uav_coordinates, uav_height= uav_height)
		#print("Channel Gain: ", self.get_channel_gain())
		
		# Calculate the data rate of the user
		self.calculate_data_rate(uav_bandwidth, other_users_transmit_powers, other_users_channel_gains)
		#print("Data Rate: ", self.get_current_data_rate())
		
		# Calculate the time overhead of the user
		self.calculate_time_overhead(other_people_strategies, other_user_data_in_bits, uav_total_data_processing_capacity, uav_cpu_frequency)
		#print("Time Overhead: ", self.get_current_time_overhead())
		
		# Calculate the total overhead of the user
		self.calculate_total_overhead(T)
		#print(" Total Overhead: ", self.get_current_total_overhead())
		
		#print("Consumed Energy: ", self.get_current_consumed_energy())

	   # Define your variable
		percentage_offloaded = cp.Variable((1, 1), nonneg=True)

		# Define the objective function with constants and parameter
		objective = cp.Maximize(
			(b * cp.exp(percentage_offloaded / cp.sum(other_people_strategies))) 
			- (c * cp.exp(self.get_current_total_overhead()))
		)

		# Define constraints
		constraints = [percentage_offloaded >= 0.1, percentage_offloaded <= 1]

		# Create the problem
		prob = cp.Problem(objective, constraints)

		# Solve the problem
		solution = prob.solve(verbose= True, qcp=True)
		
		# Update user energy
		self.set_user_strategy(percentage_offloaded.value)
  
		# Calculate the consumed energy of the user
		self.calculate_consumed_energy()
		
		return (solution, percentage_offloaded.value)
		
	def play_submodular_game_scipy(self, other_people_strategies: list, c: float, b: float, uav_bandwidth: float, other_users_transmit_powers: list, other_users_channel_gains: list, 
								   other_user_data_in_bits: list, uav_cpu_frequency: float, uav_total_data_processing_capacity: float, T: float, uav_coordinates: Tuple, uav_height: float)->float:
		"""
		Define the submodular game that the user will play with the other users
		
		Parameters:
		other_people_strategies : list
			List of the strategies of the other users
		c : float
		"""
		self.other_user_strategies = other_people_strategies
		self.other_user_transmit_powers = other_users_transmit_powers
		self.other_user_bits = other_user_data_in_bits
		self.uav_total_capacity = uav_total_data_processing_capacity
		self.T = T
  
  
		# Calculate channel gain of the user
		self.calculate_channel_gain(uav_coordinates= uav_coordinates, uav_height= uav_height)
		#print("Channel Gain: ", self.get_channel_gain())
		
		# Calculate the data rate of the user
		self.calculate_data_rate(uav_bandwidth, other_users_transmit_powers, other_users_channel_gains)
		#print("Data Rate: ", self.get_current_data_rate())
		
		# Calculate the consumed energy of the user
		self.calculate_consumed_energy()
  
		# Calculate the time overhead of the user
		self.calculate_time_overhead(other_people_strategies, other_user_data_in_bits, uav_total_data_processing_capacity, uav_cpu_frequency)
		#print("Time Overhead: ", self.get_current_time_overhead())
		
		# Calculate the total overhead of the user
		self.calculate_total_overhead(T)
		#print(" Total Overhead: ", self.get_current_total_overhead())
		
		
		#print("Consumed Energy: ", self.get_current_consumed_energy())
		
		def constraint_positive(percentage_offloaded):
			return percentage_offloaded - 0.2  # Each value should be >= 0
	
		def constraint_upper_bound(percentage_offloaded):
			return 0.8 - percentage_offloaded  # Each value should be <= 1

		def time_variance_constraint(percentage_offloaded):
			# Calculate the consumed energy of the user
			
			data_rate = self.get_current_data_rate()
			denominator = 1 - (jnp.sum(self.other_user_strategies * self.other_user_bits) / self.uav_total_capacity)

			current_time_overhead = ( ((self.data_in_bits * percentage_offloaded) / data_rate) + ((self.get_task_intensity() * percentage_offloaded * self.data_in_bits) / (denominator * uav_cpu_frequency)))
			#print("Time Overhead: ", current_time_overhead/self.T)
			#logging.info("CONSTRAINT: Time Overhead: %f", current_time_overhead/self.T)

			return self.T - current_time_overhead # Time overhead should be less than T #0.5 and noone can go high

		def energy_variance_constraint(percentage_offloaded):
			
			current_consumed_energy = ((percentage_offloaded * self.get_total_bits()) / (self.get_current_data_rate())) * self.get_transmit_power()
   
			#logging.info("CONSTRAINT: Consumed Energy: %f", current_consumed_energy/self.total_capacity)
			#print("Energy Overhead: ", 1 - (current_consumed_energy/self.total_capacity))
			return self.total_capacity - current_consumed_energy  # Energy overhead should be less than total capacity
		
		constraints = [
			{'type': 'ineq', 'fun': constraint_positive},  # percentage_offloaded >= 0
			{'type': 'ineq', 'fun': constraint_upper_bound},  # percentage_offloaded <= 1
			{'type': 'ineq', 'fun': time_variance_constraint},  # Time overhead should be less than T
			{'type': 'ineq', 'fun': energy_variance_constraint}  # Energy overhead should be less than total capacity
		]
		
		# Objective function to maximize (but converted to a minimization problem)
		def objective_function(percentage_offloaded):
			
			# Set an extremely small value to avoid overflow in the exponential
			#overflow_avoidance_factor = 1e-60
			
			# Set the weight for the x/sum(other_strategies) term
			w_s = 3
			w_o = 0.00000000000000001
			
			#Clip total overhead to avoid numerical issues
			#self.current_total_overhead = np.clip(self.current_total_overhead, 3, 50)
			
			# Reshape to match expected shape (if necessary)
			#percentage_offloaded = np.array(percentage_offloaded).reshape(-1, 1)
			
			# Calculate terms based on the provided expression
			term1 = b * np.exp( percentage_offloaded / np.sum(other_people_strategies))
			# Cap the term to avoid numerical issues
			#term1_clipped = jnp.clip(term1, 0, 1e18)
   
			term2 = c * np.exp( self.get_current_total_overhead())
			#term2_clipped = jnp.clip(term2, 0, 1e18)  # Clip to avoid numerical issues
   
			return -(term1 - term2)  # Negate for minimization

		# Solve the optimization problem using SLSQP
		result = minimize(fun= objective_function, x0= float(self.get_user_strategy()), constraints= constraints, method='SLSQP')
		
		# Extract the solution
		solution = result.x
		solution = jnp.maximum(solution, 0)  # Ensure solution is non-negative
		solution = jnp.minimum(solution, 1)  # Ensure solution is at most 1
		
		 # Update user strategy
		self.set_user_strategy(solution)

		# Calculate the consumed energy of the user
		#self.calculate_consumed_energy()
		
		# The maximum utility achieved (negated to reverse minimization)
		maximized_utility= -result.fun
  
		# Set the user utility obtained
		self.set_user_utility(maximized_utility)
		
		return (maximized_utility, solution)
	
	def play_sla_game(self, other_people_strategies: list, b: float, c: float, lr: float, uav_bandwidth: float, other_users_transmit_powers: list, other_users_channel_gains: list, 
								   other_users_data_in_bits: list, uav_cpu_frequency: float, uav_total_data_processing_capacity: float, T: float, uav_coordinates: Tuple, uav_height: float,
								   user_strategy_probabilities: list, selected_strategy: float, strategy_id: int, strategy_space: list)->float:
		"""
		Define the submodular game that the user will play with the other users
		
		Parameters:
		other_people_strategies : list
			List of the strategies of the other users
		c : float
		"""
		#self.other_user_strategies = other_people_strategies
		self.other_user_transmit_powers = other_users_transmit_powers
		self.uav_total_capacity = uav_total_data_processing_capacity
		self.T = T
		self.other_user_strategies = user_strategy_probabilities
		self.b = b
		self.c = c
		self.lr = lr

		utility_experienced_per_strategy = jnp.zeros(len(strategy_space))
		reward_experienced_per_strategy = jnp.zeros(len(strategy_space))
		normalized_reward_experienced_per_strategy = jnp.zeros(len(strategy_space))

		for i in range(len(strategy_space)):

			# Set the strategy of the user
			self.set_user_strategy(strategy_space[i])
	
			# Calculate channel gain of the user
			self.calculate_channel_gain(uav_coordinates= uav_coordinates, uav_height= uav_height)
			#print("Channel Gain: ", self.get_channel_gain())
			
			# Calculate the data rate of the user
			self.calculate_data_rate(uav_bandwidth, other_users_transmit_powers, other_users_channel_gains)
			#print("Data Rate: ", self.get_current_data_rate())
			
			# Calculate the consumed energy of the user
			self.calculate_consumed_energy()
	
			# Calculate the time overhead of the user
			self.calculate_time_overhead(other_people_strategies, other_users_data_in_bits, uav_total_data_processing_capacity, uav_cpu_frequency)
			#print("Time Overhead: ", self.get_current_time_overhead())
			
			# Calculate the total overhead of the user
			self.calculate_total_overhead(T)
			#print(" Total Overhead: ", self.get_current_total_overhead())
			
			# Calculate utility of the user
			self.user_utility = b * np.exp( selected_strategy / np.sum(other_people_strategies)) - c * np.exp( self.get_current_total_overhead())

			if self.user_utility < 0:
				self.user_utility = 0.05
			elif self.user_utility > 10000:
				self.user_utility = 1000

			# Append the utility if the user had selected this strategy
			utility_experienced_per_strategy = utility_experienced_per_strategy.at[i].set(self.user_utility)

		for i in range(len(utility_experienced_per_strategy)):

			reward = utility_experienced_per_strategy[i] / jnp.sum(utility_experienced_per_strategy)
			reward_experienced_per_strategy = reward_experienced_per_strategy.at[i].set(reward)

		for i in range(len(reward_experienced_per_strategy)):

			normalized_reward = reward_experienced_per_strategy[i] / jnp.sum(reward_experienced_per_strategy)
			normalized_reward_experienced_per_strategy = normalized_reward_experienced_per_strategy.at[i].set(normalized_reward)

		for i in range(len(normalized_reward_experienced_per_strategy)):

			# Update the probability of the user selecting this strategy
			if i == strategy_id:
				user_strategy_probabilities = user_strategy_probabilities.at[i].set(user_strategy_probabilities[i] + (lr * normalized_reward_experienced_per_strategy[i] * (1 - user_strategy_probabilities[i])))
			else:
				user_strategy_probabilities = user_strategy_probabilities.at[i].set(user_strategy_probabilities[i] - (lr * normalized_reward_experienced_per_strategy[i] * user_strategy_probabilities[i]))

		# Check if the probabilities sum to 1 or close to 1
		if jnp.sum(user_strategy_probabilities) > 1.1 or jnp.sum(user_strategy_probabilities) < 0.9:
			print(f"User {self.get_user_id()} Probabilities do not sum to 1, they sum to: ", jnp.sum(user_strategy_probabilities))

		return user_strategy_probabilities
		
	def __str__(self)->str:
		return f"User ID: {self.user_id}, Data in Bits: {self.data_in_bits}, Transmit Power: {self.transmit_power}, Energy Level: {self.energy_level}, Task Intensity: {self.task_intensity}, Coordinates: {self.coordinates}, Carrier Frequency: {self.carrier_frequency}"