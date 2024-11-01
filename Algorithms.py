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
from Qenv import Qenv
from Multiagent_Qenv import Multiagent_Qenv
from tqdm import tqdm
import matplotlib.pyplot as plt

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
		self.trajectory = []
		self.expended_energy = 0
		self.visited_nodes = 0
		self.most_processed_bits = 0
		self.energy_level = 0
		self.min_bits = 0
		self.max_bits = 0
		self.distance_min = 0
		self.distance_max = 0
		self.logger = None
		self.number_of_uavs = 1
  
	def get_uav(self)->Uav:
		"""
		Get the UAV
		
		Returns:
		Uav
			UAV
		"""
		return self.uav

	def get_uavs(self)->list:
		"""
		Get the UAVs
		
		Returns:
		list
			UAVs
		"""
		return self.uavs
	
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

	def get_trajectory(self)->list:
		"""
		Get the UAV trajectory
		
		Returns:
		list
			UAV trajectory
		"""
		return self.trajectory

	def set_trajectory(self, trajectory: list)->None:
		"""
		Set the UAV trajectory
		
		Parameters:
		trajectory : list
			UAV trajectory
		"""
		self.trajectory = trajectory
  
	def get_most_processed_bits(self)->float:
		"""
		Get the most processed bits
		
		Returns:
		float
			Most processed bits
		"""
		return self.most_processed_bits

	def get_most_expended_energy(self)->float:
		"""
		Get the expended energy
		
		Returns:
		float
			Expended energy
		"""
		return self.expended_energy

	def get_most_visited_nodes(self)->int:
		"""
		Get the total visited nodes
		
		Returns:
		int
			Total visited nodes
		"""
		return self.visited_nodes

	def set_most_processed_bits(self, most_processed_bits: float)->None:
		"""
		Set the most processed bits
		
		Parameters:
		most_processed_bits : float
			Most processed bits
		"""
		self.most_processed_bits = most_processed_bits
  
	def set_most_energy_expended(self, expended_energy: float)->None:
		"""
		Set the expended energy
		
		Parameters:
		expended_energy : float
			Expended energy
		"""
		self.expended_energy = expended_energy
  
	def set_best_trajectory(self, trajectory: list)->None:
		"""
		Set the best trajectory
		
		Parameters:
		trajectory : list
			Best trajectory
		"""
		self.trajectory = trajectory
  
	def set_best_trajectories(self, trajectories: list)->None:
		"""
		Set the best trajectories
		
		Parameters:
		trajectories : list
			Best trajectories
		"""
		self.trajectories = trajectories
  
	def get_best_trajectories(self)->list:
		"""
		Get the best trajectories
		
		Returns:
		list
			Best trajectories
		"""
		return self.trajectories
  
	def get_best_trajectory(self)->list:
		"""
		Get the best trajectory
		
		Returns:
		list
			Best trajectory
		"""
		return self.trajectory
  
	def set_most_visited_nodes(self, visited_nodes: int)->None:
		"""
		Set the total visited nodes
		
		Parameters:
		visited_nodes : int
			Total visited nodes
		"""
		self.visited_nodes = visited_nodes
	
	def sort_nodes_based_on_total_bits(self)->list:
		"""
		Sort the nodes based on the total bits of data they have
		
		Returns:
		list
			Sorted nodes
		"""
		nodes = self.graph.get_nodes()
		sorted_nodes = sorted(nodes, key= lambda x: x.get_node_total_data(), reverse= True)
		self.logger.info("The nodes have been sorted based on the total bits of data they have")
		for node in sorted_nodes:
			self.logger.info("Node %s has %s bits of data", node.get_node_id(), node.get_node_total_data())
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
		#self.logger.info("Number of Nodes: %s", self.graph.get_num_nodes())
		#self.logger.info("Number of Edges: %s", self.graph.get_num_edges())
		#self.logger.info("Number of Users: %s", self.graph.get_num_users())

		# Create a UAV
		self.uav = Uav(uav_id= 1, initial_node= nodes[0], final_node= nodes[len(nodes)-1], capacity= self.uav_energy_capacity, total_data_processing_capacity= self.uav_processing_capacity, 
						velocity= self.uav_velocity, uav_system_bandwidth= self.uav_bandwidth, cpu_frequency= self.uav_cpu_frequency, height= uav_height)
  
	def setup_algorithm_experiment(self, number_of_users: list, number_of_nodes: float, key: jax.random.PRNGKey, uav_height: float, min_distance_between_nodes: float, node_radius: float, uav_energy_capacity: float, 
                         uav_bandwidth: float, uav_processing_capacity: float, uav_cpu_frequency: float, uav_velocity: float, energy_level: float, min_bits: float, max_bits: float, distance_min: float, distance_max: float)->None:
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
		self.energy_level = energy_level
		self.min_bits = min_bits
		self.max_bits = max_bits
		self.distance_min = distance_min
		self.distance_max = distance_max
		self.trajectory = None

		
		nodes = []
		for i in range(number_of_nodes):
			# Generate random center coordinates for the node
			node_coords = generate_node_coordinates(key, nodes, min_distance_between_nodes)

			max_bits = max_bits  # Maximum bits for the highest user ID 900000000
			min_bits = min_bits    # Minimum bits for the lowest user ID
			bit_range = max_bits - min_bits

			#max_distance = 2  # Set maximum distance to 10

			users = []
			# Convert the number of users to an integer if it's an array
			num_users = int(number_of_users[number_of_nodes])

			# Split the key into enough subkeys for the number of users
			subkeys = jax.random.split(key, num_users)

			for j in range(number_of_users[number_of_nodes]):
				# Use a unique subkey for each user to get different random values
				data_in_bits = jax.random.uniform(subkeys[j], minval=min_bits, maxval=max_bits)
				
				# Sharpen the decrease in `r` to make higher ID users much closer to the center
				r_scale = (j / number_of_users[number_of_nodes]) ** 2  # Exponential decrease for sharper differences
				r = r_scale
				theta = random.uniform(random.split(key)[0], (1,))[0] * 2 * jnp.pi  # azimuthal angle (0 to 2*pi)
				phi = random.uniform(random.split(key)[0], (1,))[0] * jnp.pi  # polar angle (0 to pi)

				# Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z)
				x = r * jnp.sin(phi) * jnp.cos(theta)
				y = r * jnp.sin(phi) * jnp.sin(theta)
				z = r * jnp.cos(phi)

				# User coordinates relative to the node center
				user_coords = (node_coords[0] + x, node_coords[1] + y, node_coords[2] + z)
				
				# Create user object with random data_in_bits
				user = AoiUser(
					user_id=j,
					data_in_bits=data_in_bits,  # Assign random bits to each user
					transmit_power=1.5,
					energy_level=energy_level,
					task_intensity=1,
					carrier_frequency=5,
					coordinates=user_coords
				)

				users.append(user)
			
			nodes.append(Node(
				node_id=i,
				users=users,
				coordinates=node_coords
			))
   
		# Calculate the distance between all users and the UAV
		for node in nodes:
			node_distances = []
			for user in node.get_user_list():
				user.calculate_distance(node)
				node_distances.append(user.get_distance())
			# Scale the distances to be between 100 and 1000
			max_distance = max(node_distances)
			min_distance = min(node_distances)
			distance_range = max_distance - min_distance
			for user in node.get_user_list():
				user.set_distance(distance_min + distance_max * (user.get_distance() - min_distance) / distance_range)
		# Create edges between all nodes with random weights
		edges = []
		for i in range(number_of_nodes):
			for j in range(i+1, number_of_nodes):
				edges.append(Edge(nodes[i].user_list[0], nodes[j].user_list[0], random.normal(key, (1,))))
				
		# Create the graph
		self.graph = Graph(nodes=nodes, edges=edges)

		# Get number of nodes and edges
		#self.logger.info("Number of Nodes: %s", self.graph.get_num_nodes())
		#self.logger.info("Number of Edges: %s", self.graph.get_num_edges())
		#self.logger.info("Number of Users: %s", self.graph.get_num_users())
  
		# Create two subkeys for the initial and final node selection
		key, subkey1, subkey2 = random.split(key, 3)

		# # Randomly select a node object to be the final and the initial node
		# initial_node = random.randint(subkey1, (1,), 0, number_of_nodes)[0]
		# initial_node = nodes[initial_node]
	
		# final_node = random.randint(subkey2, (1,), 0, number_of_nodes)[0]

		# # Check if initial and final nodes are the same
		# while final_node == initial_node:
		# 	subkey3 = random.split(key)[0]
		# 	final_node = random.randint(subkey3, (1,), 0, number_of_nodes)[0]
	
		# final_node = nodes[final_node]

		#print("The initial node is %s", initial_node.get_node_id())
		#print("The final node is %s", final_node.get_node_id())

		initial_node = nodes[0]
		final_node = nodes[len(nodes)-1]	
		# Create a UAV
		self.uav = Uav(uav_id=1, initial_node= initial_node, final_node= final_node, capacity=self.uav_energy_capacity, total_data_processing_capacity=self.uav_processing_capacity, 
					velocity=self.uav_velocity, uav_system_bandwidth=self.uav_bandwidth, cpu_frequency=self.uav_cpu_frequency, height=uav_height)
		
	def setup_realistic_scenario(self, number_of_users: list, number_of_nodes: float, key: jax.random.PRNGKey, uav_height: float, min_distance_between_nodes: float, node_radius: float, uav_energy_capacity: float, 
                         uav_bandwidth: float, uav_processing_capacity: float, uav_cpu_frequency: float, uav_velocity: float, energy_level: float, min_bits: float, max_bits: float, distance_min: float, distance_max: float)->None:
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
		self.energy_level = energy_level
		self.min_bits = min_bits
		self.max_bits = max_bits
		self.distance_min = distance_min
		self.distance_max = distance_max

		
		nodes = []
		sum_of_bits = []
		for i in range(number_of_nodes):
			# Generate random center coordinates for the node
			node_coords = generate_node_coordinates(key, nodes, min_distance_between_nodes)

			# Determine bit size range for each node: 3 nodes with high bits, 2 nodes with low bits
			if i in [1,2,4]:  # Nodes with a larger number of bits
				bits_range = (max_bits * 0.85, max_bits)  # Upper range of bit capacity
			else:  # Nodes with a smaller number of bits
				bits_range = (min_bits, min_bits * 1.2)  # Lower range of bit capacity

			users = []
			num_users = int(number_of_users[number_of_nodes-1])

			# Split the key into enough subkeys for the number of users
			subkeys = jax.random.split(key, num_users)
			bits_sum = 0
			for j in range(num_users):
				data_in_bits = jax.random.uniform(subkeys[j], minval=bits_range[0], maxval=bits_range[1])
				bits_sum += data_in_bits
				r_scale = (j / num_users) ** 2  # Exponential decrease for sharper differences
				r = r_scale
				theta = random.uniform(random.split(key)[0], (1,))[0] * 2 * jnp.pi
				phi = random.uniform(random.split(key)[0], (1,))[0] * jnp.pi

				x = r * jnp.sin(phi) * jnp.cos(theta)
				y = r * jnp.sin(phi) * jnp.sin(theta)
				z = r * jnp.cos(phi)

				user_coords = (node_coords[0] + x, node_coords[1] + y, node_coords[2] + z)

				user = AoiUser(
					user_id=j,
					data_in_bits=data_in_bits,
					transmit_power=1.5,
					energy_level=energy_level,
					task_intensity=1,
					carrier_frequency=5,
					coordinates=user_coords
				)

				users.append(user)
			sum_of_bits.append(bits_sum)
			nodes.append(Node(
				node_id=i,
				users=users,
				coordinates=node_coords
			))

		# Calculate the distance between all users and the UAV and set distances
		for node in nodes:
			node_distances = []
			for user in node.get_user_list():
				user.calculate_distance(node)
				node_distances.append(user.get_distance())
			max_distance = max(node_distances)
			min_distance = min(node_distances)
			distance_range = max_distance - min_distance
			for user in node.get_user_list():
				user.set_distance(distance_min + distance_max * (user.get_distance() - min_distance) / distance_range)

		# Create edges between all nodes with random weights
		edges = []
		for i in range(number_of_nodes):
			for j in range(i + 1, number_of_nodes):
				edges.append(Edge(nodes[i], nodes[j], random.normal(key, (1,))))

		# Create the graph
		self.graph = Graph(nodes=nodes, edges=edges)
		self.graph.plot_3d_graph()
  
		# Plot the sum of bits of each node
		plt.figure(figsize=(10, 5))
  
		plt.bar(jnp.arange(number_of_nodes), sum_of_bits, color='skyblue')
  
		plt.xlabel('Node ID')
		plt.ylabel('Sum of Bits')
  
		plt.title('Sum of Bits of Each Node')
  
		plt.savefig('sum_of_bits.png')
  

		initial_node = nodes[0]
		final_node = nodes[-1]
  
		self.uav = Uav(
			uav_id=1, initial_node=initial_node, final_node=final_node,
			capacity=self.uav_energy_capacity, total_data_processing_capacity=self.uav_processing_capacity,
			velocity=self.uav_velocity, uav_system_bandwidth=self.uav_bandwidth, cpu_frequency=self.uav_cpu_frequency,
			height=uav_height
		)
  
	def setup_multiagent_scenario(self, number_of_users: list, number_of_nodes: float, key: jax.random.PRNGKey, uav_height: float, min_distance_between_nodes: float, node_radius: float, uav_energy_capacity: float, 
                         uav_bandwidth: float, uav_processing_capacity: float, uav_cpu_frequency: float, uav_velocity: float, energy_level: float, min_bits: float, max_bits: float, distance_min: float, distance_max: float, 
                         number_of_uavs: int)->None:
		"""
		Setup the experiment
		
		Parameters:
		number_of_users : list
			Number of users in the system per node
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
		self.energy_level = energy_level
		self.min_bits = min_bits
		self.max_bits = max_bits
		self.distance_min = distance_min
		self.distance_max = distance_max
		self.number_of_uavs = number_of_uavs

		nodes = []
		for i in range(number_of_nodes):
			# Generate random center coordinates for the node
			node_coords = generate_node_coordinates(key, nodes, min_distance_between_nodes)

			max_bits = max_bits  # Maximum bits for the highest user ID 900000000
			min_bits = min_bits    # Minimum bits for the lowest user ID
			bit_range = max_bits - min_bits

			#max_distance = 2  # Set maximum distance to 10

			users = []
			# Convert the number of users to an integer if it's an array

			num_users = int(number_of_users[0][number_of_nodes])

			# Split the key into enough subkeys for the number of users
			subkeys = jax.random.split(key, num_users)

			for j in range(num_users):
				# Use a unique subkey for each user to get different random values
				data_in_bits = jax.random.uniform(subkeys[j], minval=min_bits, maxval=max_bits)
				
				# Sharpen the decrease in `r` to make higher ID users much closer to the center
				r_scale = (j / number_of_users[0][number_of_nodes]) ** 2  # Exponential decrease for sharper differences
				r = r_scale
				theta = random.uniform(random.split(key)[0], (1,))[0] * 2 * jnp.pi  # azimuthal angle (0 to 2*pi)
				phi = random.uniform(random.split(key)[0], (1,))[0] * jnp.pi  # polar angle (0 to pi)

				# Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z)
				x = r * jnp.sin(phi) * jnp.cos(theta)
				y = r * jnp.sin(phi) * jnp.sin(theta)
				z = r * jnp.cos(phi)

				# User coordinates relative to the node center
				user_coords = (node_coords[0] + x, node_coords[1] + y, node_coords[2] + z)
				
				# Create user object with random data_in_bits
				user = AoiUser(
					user_id=j,
					data_in_bits=data_in_bits,  # Assign random bits to each user
					transmit_power=1.5,
					energy_level=energy_level,
					task_intensity=1,
					carrier_frequency=5,
					coordinates=user_coords
				)

				users.append(user)
			
			nodes.append(Node(
				node_id=i,
				users=users,
				coordinates=node_coords
			))
   
		# Calculate the distance between all users and the UAV
		for node in nodes:
			node_distances = []
			for user in node.get_user_list():
				user.calculate_distance(node)
				node_distances.append(user.get_distance())
			# Scale the distances to be between 100 and 1000
			max_distance = max(node_distances)
			min_distance = min(node_distances)
			distance_range = max_distance - min_distance
			for user in node.get_user_list():
				user.set_distance(distance_min + distance_max * (user.get_distance() - min_distance) / distance_range)
		# Create edges between all nodes with random weights
		edges = []
		for i in range(number_of_nodes):
			for j in range(i+1, number_of_nodes):
				edges.append(Edge(nodes[i], nodes[j], random.normal(key, (1,))))
				
		# Create the graph
		self.graph = Graph(nodes=nodes, edges=edges)

		# Initialize UAVs on random nodes
		self.uavs = []
		node_indices = jax.random.choice(key, jnp.arange(number_of_nodes), shape=(number_of_uavs,), replace=False)
		for uav_id, node_index in enumerate(node_indices):
			node = nodes[node_index]
			uav = Uav(
				uav_id=uav_id,
				initial_node=node,
				final_node=node,  # Set the final node; adjust if necessary
				capacity=self.uav_energy_capacity,
				total_data_processing_capacity=self.uav_processing_capacity,
				velocity=self.uav_velocity,
				uav_system_bandwidth=self.uav_bandwidth,
				cpu_frequency=self.uav_cpu_frequency,
				height=uav_height
			)
			self.uavs.append(uav)
		
  
	def setup_singular_experiment(self, number_of_users: list, number_of_nodes: float, key: jax.random.PRNGKey, uav_height: float, min_distance_between_nodes: float, node_radius: float, uav_energy_capacity: float, 
                         uav_bandwidth: float, uav_processing_capacity: float, uav_cpu_frequency: float, uav_velocity: float, energy_level: float, min_bits: float, max_bits: float, distance_min: float, distance_max: float,
						 logger: logging.Logger)->None:
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
		self.energy_level = energy_level
		self.min_bits = min_bits
		self.max_bits = max_bits
		self.distance_min = distance_min
		self.distance_max = distance_max
		self.logger = logger

		
		nodes = []
		for i in range(number_of_nodes):
			# Generate random center coordinates for the node
			node_coords = generate_node_coordinates(key, nodes, min_distance_between_nodes)

			max_bits = max_bits  # Maximum bits for the highest user ID 900000000
			min_bits = min_bits    # Minimum bits for the lowest user ID
			bit_range = max_bits - min_bits

			#max_distance = 2  # Set maximum distance to 10

			users = []
			for j in range(number_of_users[number_of_nodes]):

				data_step = 1000  # Amplify the data step size for bigger differences
				base_data_in_bits = 1000  # Set a base value for data bits
				data_in_bits = max_bits - bit_range * (j / number_of_users[number_of_nodes])**0.5  # Square root-based smooth decrease

				# Sharpen the decrease in `r` to make higher ID users much closer to the center
				r_scale = (j / number_of_users[number_of_nodes]) ** 2  # Exponential decrease for sharper differences
				#r = r_scale * max_distance  # Scale `r` to reach the maximum distance of 10
				r = r_scale
				theta = random.uniform(random.split(key)[0], (1,))[0] * 2 * jnp.pi  # azimuthal angle (0 to 2*pi)
				phi = random.uniform(random.split(key)[0], (1,))[0] * jnp.pi  # polar angle (0 to pi)
				
				# Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z)
				x = r * jnp.sin(phi) * jnp.cos(theta)
				y = r * jnp.sin(phi) * jnp.sin(theta)
				z = r * jnp.cos(phi)
    
				# User coordinates relative to the node center
				user_coords = (node_coords[0] + x, node_coords[1] + y, node_coords[2] + z)
    
				# Amplified differences in data bits
				#data_in_bits = base_data_in_bits + j * (data_step / number_of_users[number_of_nodes])
				#Smoothed data bits using a square root-based increase
				#data_in_bits = min_bits + bit_range * (j / number_of_users[number_of_nodes])**0.5  # Square root-based smooth increase
				
				user = AoiUser(
					user_id=j,
					data_in_bits=data_in_bits,
					transmit_power=1.5,
					energy_level=energy_level,
					task_intensity=1,
					carrier_frequency=5,
					coordinates=user_coords
				)
    
				users.append(user)
			
			nodes.append(Node(
				node_id=i,
				users=users,
				coordinates=node_coords
			))
   
		# Calculate the distance between all users and the UAV
		for node in nodes:
			node_distances = []
			for user in node.get_user_list():
				user.calculate_distance(node)
				node_distances.append(user.get_distance())
			# Scale the distances to be between 100 and 1000
			max_distance = max(node_distances)
			min_distance = min(node_distances)
			distance_range = max_distance - min_distance
			for user in node.get_user_list():
				user.set_distance(distance_min + distance_max * (user.get_distance() - min_distance) / distance_range)
		# Create edges between all nodes with random weights
		edges = []
		for i in range(number_of_nodes):
			for j in range(i+1, number_of_nodes):
				edges.append(Edge(nodes[i].user_list[0], nodes[j].user_list[0], random.normal(key, (1,))))
				
		# Create the graph
		self.graph = Graph(nodes=nodes, edges=edges)

		# Get number of nodes and edges
		self.logger.info("Number of Nodes: %s", self.graph.get_num_nodes())
		self.logger.info("Number of Edges: %s", self.graph.get_num_edges())
		self.logger.info("Number of Users: %s", self.graph.get_num_users())

		# Create a UAV
		self.uav = Uav(uav_id=1, initial_node=nodes[0], final_node=nodes[len(nodes)-1], capacity=self.uav_energy_capacity, total_data_processing_capacity=self.uav_processing_capacity, 
					velocity=self.uav_velocity, uav_system_bandwidth=self.uav_bandwidth, cpu_frequency=self.uav_cpu_frequency, height=uav_height)


	def reset(self)->None:
		"""
		Reset the experiment so that it can be run again by the same or a different algorithm
		"""
		if self.number_of_uavs==1:
			self.graph = None
			self.uav = None
			self.setup_realistic_scenario(number_of_users= self.number_of_users, number_of_nodes= self.number_of_nodes, key= self.key, uav_height= self.uav_height, min_distance_between_nodes= self.min_distance_between_nodes, node_radius= self.node_radius,
								uav_energy_capacity= self.uav_energy_capacity, uav_bandwidth= self.uav_bandwidth, uav_processing_capacity= self.uav_processing_capacity, uav_cpu_frequency= self.uav_cpu_frequency, uav_velocity= self.uav_velocity
								, energy_level= self.energy_level, min_bits= self.min_bits, max_bits= self.max_bits, distance_min= self.distance_min, distance_max= self.distance_max)
		else:
			self.graph = None
			self.uav = None
			self.setup_multiagent_scenario(number_of_users= self.number_of_users, number_of_nodes= self.number_of_nodes, key= self.key, uav_height= self.uav_height, min_distance_between_nodes= self.min_distance_between_nodes, node_radius= self.node_radius,
								uav_energy_capacity= self.uav_energy_capacity, uav_bandwidth= self.uav_bandwidth, uav_processing_capacity= self.uav_processing_capacity, uav_cpu_frequency= self.uav_cpu_frequency, uav_velocity= self.uav_velocity
								, energy_level= self.energy_level, min_bits= self.min_bits, max_bits= self.max_bits, distance_min= self.distance_min, distance_max= self.distance_max, number_of_uavs= self.number_of_uavs)
		
	def run_single_submodular_game(self, solving_method: str, c: float, b: float, logger: logging.Logger = None)->None:
		"""
		Run a single submodular game
		
		Parameters:
		solving_method : str
			Solving method to be used for the submodular game (cvxpy or scipy)
		"""
		self.logger = logger
		self.logger.info("Running a Single Submodular Game")
		uav = self.get_uav()
		graph = self.get_graph()
		U = self.get_number_of_users()
		key = self.get_key()
		convergence_threshold = self.get_convergence_threshold()
		T = 2
		
		# Start playing the game inside the current node
		done = False
		temp_U = U[uav.get_current_node().get_node_id()]
		user_strategies = jnp.ones(temp_U) * 0.001  # Strategies for all users
		#user_strategies = random.uniform(key, shape=(temp_U,1), minval=0.001, maxval=1.0)
		uav_bandwidth = uav.get_uav_bandwidth()
		uav_cpu_frequency = uav.get_cpu_frequency()
		uav_total_data_processing_capacity = uav.get_total_data_processing_capacity()
		
		user_channel_gains = jnp.zeros(temp_U)
		user_transmit_powers = jnp.zeros(temp_U)
		user_data_in_bits = jnp.zeros(temp_U)
		
		convergence_history = []
		convergence_counter = 0
		convergence_counter += 5
		convergence_history.append(convergence_counter/1)
		
		for idx, user in enumerate(uav.get_current_node().get_user_list()):
			# Calculate channel gain
			user.calculate_channel_gain(uav.get_current_coordinates(), uav.get_height())
			# Assing the channel gain and transmit power to the user
			user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
			user_transmit_powers = user_transmit_powers.at[idx].set(user.get_transmit_power())
			user_data_in_bits = user_data_in_bits.at[idx].set(user.get_user_bits())
			user.set_user_strategy(user_strategies[idx])
		
		iteration_counter = 1
		
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
						
						self.logger.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
						self.logger.info("User %d has maximized its utility to %s", idx, maximized_utility)
						
						# Update the user's strategy
						user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
						
						# Update user's channel gain
						user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
						
				elif solving_method == "scipy":
					# Play the submodular game
					maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
																							uav_cpu_frequency, uav_total_data_processing_capacity, T, uav.get_current_coordinates(), uav.get_height())

					# logger.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
					# logger.info("User %d has maximized its utility to %s", idx, maximized_utility)
					
					# Update the user's strategy
					user_strategies = user_strategies.at[idx].set(percentage_offloaded[0])
					
					# Update user's channel gain
					user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())

			# Check how different the strategies are from the previous iteration    
			strategy_difference = jnp.linalg.norm(user_strategies - previous_strategies)
			
			convergence_counter += strategy_difference
			convergence_history.append(convergence_counter/iteration_counter)
				
			# Check if the strategies have converged
			if strategy_difference < convergence_threshold:
				# Calculate the consumed energy for all users based on the strategies they have chosen and adjust the energy
				# for idx, user in enumerate(uav.get_current_node().get_user_list()):
				#     user.calculate_consumed_energy()
				#     user.adjust_energy(user.get_current_consumed_energy())
				
				# Adjust UAV energy for processing the offloaded data
				uav.energy_to_process_data(energy_coefficient= 0.1)
				
				# Decreases the user data in bits based on the offloaded data
				for idx, user in enumerate(uav.get_current_node().get_user_list()):
					user.set_user_strategy(user_strategies[idx])
					user.calculate_remaining_data()
					user.calculate_consumed_energy()
				uav.get_current_node().calculate_total_bit_data()
				done = True
					
		# Calculate total data in bits processed by the UAV in this node
		total_data_processed = 0
		for user in self.uav.get_current_node().get_user_list():
			total_data_processed += user.get_current_strategy() * user.get_user_bits()
		
		self.uav.update_total_processed_data(total_data_processed)
		
		uav.set_finished_business_in_node(True)
		uav.hover_over_node(time_hover= T)
		self.logger.info("The UAV has finished its business in the current node")
		self.logger.info("The strategies at node %s have converged to: %s", uav.get_current_node().get_node_id(), user_strategies)
		# Log the task_intensity of the users
		task_intensities = []
		for user in uav.get_current_node().get_user_list():
			task_intensities.append(user.get_task_intensity())
		self.logger.info("The task intensities of the users at node %s are: %s", uav.get_current_node().get_node_id(), task_intensities)
		self.logger.info("Converged with strategy difference: %s in %d iterations", strategy_difference, iteration_counter)
		
		# Keep adding to the convergence history for 10 more iterations
		for i in range(100):
			convergence_counter += strategy_difference
			iteration_counter += 1
			convergence_history.append(convergence_counter/iteration_counter)
				
		return convergence_history
						

		
	def run_random_walk_algorithm(self, solving_method: str, max_iter: int, c: float, b: float, logger : logging.Logger = None  )->bool:
		"""
		Algorithm that makes the UAV navigate through the graph randomly
		Every time it needs to move from one node to another, it will randomly choose the next node from a set of unvisited nodes
			
		Parameters:
		solving_method : str
			Solving method to be used for the submodular game (cvxpy or scipy)
		max_iter : int
			Maximum number of iterations to run the algorithm
		
		Returns:
			bool
				True if the UAV has reached the final node, False otherwise
		"""
		self.logger = logger
		self.logger.info("Running the Random Walk Algorithm")
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
					c = c
					b = b
					
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
							
							logger.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
							logger.info("User %d has maximized its utility to %s", idx, maximized_utility)
							
							# Update the user's strategy
							user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
							
							# Update user's channel gain
							user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
							
						elif solving_method == "scipy":
							# Play the submodular game
							maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
																									uav_cpu_frequency, uav_total_data_processing_capacity, 2, uav.get_current_coordinates(), uav.get_height())
	
							# logger.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
							# logger.info("User %d has maximized its utility to %s", idx, maximized_utility)
							
							# Update the user's strategy
							user_strategies = user_strategies.at[idx].set(percentage_offloaded[0])
							
							# Update user's channel gain
							user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())

					# Check how different the strategies are from the previous iteration    
					strategy_difference = jnp.linalg.norm(user_strategies - previous_strategies)
					
					# Check if the strategies have converged
					if strategy_difference < convergence_threshold:
						# Calculate the consumed energy for all users based on the strategies they have chosen and adjust the energy
						# for idx, user in enumerate(uav.get_current_node().get_user_list()):
						#     user.calculate_consumed_energy()
						#     user.adjust_energy(user.get_current_consumed_energy())
						
						# Adjust UAV energy for processing the offloaded data
						uav.energy_to_process_data(energy_coefficient= 0.1)
						
						# Decreases the user data in bits based on the offloaded data
						for idx, user in enumerate(uav.get_current_node().get_user_list()):
							user.set_user_strategy(user_strategies[idx])
							user.calculate_remaining_data()
						uav.get_current_node().calculate_total_bit_data()
						done = True
						
				# Calculate total data in bits processed by the UAV in this node
				total_data_processed = 0
				for user in self.uav.get_current_node().get_user_list():
					total_data_processed += user.get_current_strategy() * user.get_user_bits()
					
				self.uav.update_total_processed_data(total_data_processed)
				
				uav.set_finished_business_in_node(True)
				uav.hover_over_node(time_hover= T)
				self.logger.info("The UAV has finished its business in the current node")
				self.logger.info("The strategies at node %s have converged to: %s", uav.get_current_node().get_node_id(), user_strategies)
				# Log the task_intensity of the users
				task_intensities = []
				for user in uav.get_current_node().get_user_list():
					task_intensities.append(user.get_task_intensity())
				self.logger.info("The task intensities of the users at node %s are: %s", uav.get_current_node().get_node_id(), task_intensities)
				self.logger.info("Converged with strategy difference: %s in %d iterations", strategy_difference, iteration_counter)
				
				if (uav_has_reached_final_node):
					self.logger.info("The UAV has reached the final node and has finished its business")
					break_flag = True
				#print(f"Final Strategies: {user_strategies}")
			else:
				# Decide to which node to move next randomly from the ones availalbe that are not visited
				next_node = uav.get_random_unvisited_next_node(nodes= graph.get_nodes(), key= key, max_iter= max_iter)
				if (next_node is not None):
					if (uav.travel_to_node(next_node)):
						self.logger.info("The UAV has reached the next node")
						self.logger.info("The UAV energy level is: %s after going to the next node", uav.get_energy_level())
						# Check if the UAV has reached the final node
						if uav.check_if_final_node(uav.get_current_node()):
							self.logger.info("The UAV has reached the final node")
							uav_has_reached_final_node = True
					else:
						self.logger.info("The UAV has not reached the next node because it has not enough energy")
						break_flag = True
				else:
					self.logger.info("The UAV has visited all the nodes")
					break_flag = True
		trajectory = uav.get_visited_nodes()
		trajectory_ids = []
		for node in trajectory:
			trajectory_ids.append(node.get_node_id())
			
		self.logger.info("The UAV trajectory is: %s", trajectory_ids)
		self.logger.info("The UAV has visited %d nodes", len(trajectory_ids))
		self.set_trajectory(trajectory_ids)
		
		if uav_has_reached_final_node:
			return True
		else:
			return False
		
	def run_random_proportional_fairness_algorithm(self, solving_method: str, max_iter: int, c: float, b: float, logger : logging.Logger = None)->bool:
		"""
		Algorithm that makes the UAV navigate through the graph randomly giving bigger chance to visit nodes that have more data to process
			
		Parameters:
		solving_method : str
			Solving method to be used for the submodular game (cvxpy or scipy)
		max_iter : int
			Maximum number of iterations to run the algorithm
		
		Returns:
			bool
				True if the UAV has reached the final node, False otherwise
		"""
		self.logger = logger
		self.logger.info("Running the Proportional Fairness Algorithm")
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
					c = c
					b = b
					
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
							
							self.logger.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
							self.logger.info("User %d has maximized its utility to %s", idx, maximized_utility)
							
							# Update the user's strategy
							user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
							
							# Update user's channel gain
							user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
							
						elif solving_method == "scipy":
							# Play the submodular game
							maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
																									uav_cpu_frequency, uav_total_data_processing_capacity, 2, uav.get_current_coordinates(), uav.get_height())
	
							# logger.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
							# logger.info("User %d has maximized its utility to %s", idx, maximized_utility)
							
							# Update the user's strategy
							user_strategies = user_strategies.at[idx].set(percentage_offloaded[0])
							
							# Update user's channel gain
							user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())

					# Check how different the strategies are from the previous iteration    
					strategy_difference = jnp.linalg.norm(user_strategies - previous_strategies)
					
					# Check if the strategies have converged
					if strategy_difference < convergence_threshold:
						# Calculate the consumed energy for all users based on the strategies they have chosen and adjust the energy
						# for idx, user in enumerate(uav.get_current_node().get_user_list()):
						#     user.calculate_consumed_energy()
						#     user.adjust_energy(user.get_current_consumed_energy())
						
						# Adjust UAV energy for processing the offloaded data
						uav.energy_to_process_data(energy_coefficient= 0.1)
						
						# Decreases the user data in bits based on the offloaded data
						for idx, user in enumerate(uav.get_current_node().get_user_list()):
							user.set_user_strategy(user_strategies[idx])
							user.calculate_remaining_data()
						uav.get_current_node().calculate_total_bit_data()
						done = True
						
				# Calculate total data in bits processed by the UAV in this node
				total_data_processed = 0
				for user in self.uav.get_current_node().get_user_list():
					total_data_processed += user.get_current_strategy() * user.get_user_bits()
					
				self.uav.update_total_processed_data(total_data_processed)
				
				uav.set_finished_business_in_node(True)
				uav.hover_over_node(time_hover= T)
				self.logger.info("The UAV has finished its business in the current node")
				self.logger.info("The strategies at node %s have converged to: %s", uav.get_current_node().get_node_id(), user_strategies)
				# Log the task_intensity of the users
				task_intensities = []
				for user in uav.get_current_node().get_user_list():
					task_intensities.append(user.get_task_intensity())
				self.logger.info("The task intensities of the users at node %s are: %s", uav.get_current_node().get_node_id(), task_intensities)
				self.logger.info("Converged with strategy difference: %s in %d iterations", strategy_difference, iteration_counter)
				
				if (uav_has_reached_final_node):
					self.logger.info("The UAV has reached the final node and has finished its business")
					break_flag = True
				#print(f"Final Strategies: {user_strategies}")
			else:
				# Decide to which node to move next randomly from the ones availalbe that are not visited
				next_node = uav.get_random_unvisited_next_node_with_proportion(nodes= graph.get_nodes(), key= key, max_iter= max_iter)
				if (next_node is not None):
					if (uav.travel_to_node(next_node)):
						self.logger.info("The UAV has reached the next node")
						self.logger.info("The UAV energy level is: %s after going to the next node", uav.get_energy_level())
						# Check if the UAV has reached the final node
						if uav.check_if_final_node(uav.get_current_node()):
							self.logger.info("The UAV has reached the final node")
							uav_has_reached_final_node = True
					else:
						self.logger.info("The UAV has not reached the next node because it has not enough energy")
						break_flag = True
				else:
					self.logger.info("The UAV has visited all the nodes")
					break_flag = True
		trajectory = uav.get_visited_nodes()
		trajectory_ids = []
		for node in trajectory:
			trajectory_ids.append(node.get_node_id())
			
		self.logger.info("The UAV trajectory is: %s", trajectory_ids)
		self.set_trajectory(trajectory_ids)
		
		if uav_has_reached_final_node:
			return True
		else:
			return False

	def run_max_logit_algorithm(self, solving_method: str, max_iter: int, c: float, b: float, logger : logging.Logger = None)->bool:
		"""
		Algorithm that makes the UAV navigate through the graph randomly giving bigger chance to visit nodes that have more data to process
			
		Parameters:
		solving_method : str
			Solving method to be used for the submodular game (cvxpy or scipy)
		max_iter : int
			Maximum number of iterations to run the algorithm
		
		Returns:
			bool
				True if the UAV has reached the final node, False otherwise
		"""
		self.logger = logger
		self.logger.info("Running the Max-Logit Algorithm")
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
					c = c
					b = b
					
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
							
							self.logger.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
							self.logger.info("User %d has maximized its utility to %s", idx, maximized_utility)
							
							# Update the user's strategy
							user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
							
							# Update user's channel gain
							user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
							
						elif solving_method == "scipy":
							# Play the submodular game
							maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
																									uav_cpu_frequency, uav_total_data_processing_capacity, 2, uav.get_current_coordinates(), uav.get_height())
	
							# logger.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
							# logger.info("User %d has maximized its utility to %s", idx, maximized_utility)
							
							# Update the user's strategy
							user_strategies = user_strategies.at[idx].set(percentage_offloaded[0])
							
							# Update user's channel gain
							user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())

					# Check how different the strategies are from the previous iteration    
					strategy_difference = jnp.linalg.norm(user_strategies - previous_strategies)
					
					# Check if the strategies have converged
					if strategy_difference < convergence_threshold:
						# Calculate the consumed energy for all users based on the strategies they have chosen and adjust the energy
						# for idx, user in enumerate(uav.get_current_node().get_user_list()):
						#     user.calculate_consumed_energy()
						#     user.adjust_energy(user.get_current_consumed_energy())
						
						# Adjust UAV energy for processing the offloaded data
						uav.energy_to_process_data(energy_coefficient= 0.1)
						
						# Decreases the user data in bits based on the offloaded data
						for idx, user in enumerate(uav.get_current_node().get_user_list()):
							user.set_user_strategy(user_strategies[idx])
							user.calculate_remaining_data()
						uav.get_current_node().calculate_total_bit_data()
						done = True
						
				# Calculate total data in bits processed by the UAV in this node
				total_data_processed = 0
				for user in self.uav.get_current_node().get_user_list():
					total_data_processed += user.get_current_strategy() * user.get_user_bits()
					
				self.uav.update_total_processed_data(total_data_processed)
				
				uav.set_finished_business_in_node(True)
				uav.hover_over_node(time_hover= T)
				self.logger.info("The UAV has finished its business in the current node")
				self.logger.info("The strategies at node %s have converged to: %s", uav.get_current_node().get_node_id(), user_strategies)
				# Log the task_intensity of the users
				task_intensities = []
				for user in uav.get_current_node().get_user_list():
					task_intensities.append(user.get_task_intensity())
				self.logger.info("The task intensities of the users at node %s are: %s", uav.get_current_node().get_node_id(), task_intensities)
				self.logger.info("Converged with strategy difference: %s in %d iterations", strategy_difference, iteration_counter)
				
				if (uav_has_reached_final_node):
					self.logger.info("The UAV has reached the final node and has finished its business")
					break_flag = True
				#print(f"Final Strategies: {user_strategies}")
			else:
				# Decide to which node to move next randomly from the ones availalbe that are not visited
				next_node = uav.get_random_unvisited_next_node_with_max_logit(nodes= graph.get_nodes(), key= key, max_iter= max_iter)
				if (next_node is not None):
					if (uav.travel_to_node(next_node)):
						self.logger.info("The UAV has reached the next node")
						self.logger.info("The UAV energy level is: %s after going to the next node", uav.get_energy_level())
						# Check if the UAV has reached the final node
						if uav.check_if_final_node(uav.get_current_node()):
							self.logger.info("The UAV has reached the final node")
							uav_has_reached_final_node = True
					else:
						self.logger.info("The UAV has not reached the next node because it has not enough energy")
						break_flag = True
				else:
					self.logger.info("The UAV has visited all the nodes")
					break_flag = True
		trajectory = uav.get_visited_nodes()
		trajectory_ids = []
		for node in trajectory:
			trajectory_ids.append(node.get_node_id())
			
		self.logger.info("The UAV trajectory is: %s", trajectory_ids)
		self.set_trajectory(trajectory_ids)
		
		if uav_has_reached_final_node:
			return True
		else:
			return False

	def run_b_logit_algorithm(self, solving_method: str, max_iter: int, c: float, b: float, beta: float= 1.0, logger : logging.Logger = None)->bool:
		"""
		Algorithm that makes the UAV navigate through the graph randomly giving bigger chance to visit nodes that have more data to process
			
		Parameters:
		solving_method : str
			Solving method to be used for the submodular game (cvxpy or scipy)
		max_iter : int
			Maximum number of iterations to run the algorithm
		
		Returns:
			bool
				True if the UAV has reached the final node, False otherwise
		"""
		self.logger = logger
		self.logger.info("Running the B-Logit Algorithm")
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
					c = c
					b = b
					
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
							
							self.logger.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
							self.logger.info("User %d has maximized its utility to %s", idx, maximized_utility)
							
							# Update the user's strategy
							user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
							
							# Update user's channel gain
							user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
							
						elif solving_method == "scipy":
							# Play the submodular game
							maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
																									uav_cpu_frequency, uav_total_data_processing_capacity, 2, uav.get_current_coordinates(), uav.get_height())
	
							# logger.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
							# logger.info("User %d has maximized its utility to %s", idx, maximized_utility)
							
							# Update the user's strategy
							user_strategies = user_strategies.at[idx].set(percentage_offloaded[0])
							
							# Update user's channel gain
							user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())

					# Check how different the strategies are from the previous iteration    
					strategy_difference = jnp.linalg.norm(user_strategies - previous_strategies)
					
					# Check if the strategies have converged
					if strategy_difference < convergence_threshold:
						# Calculate the consumed energy for all users based on the strategies they have chosen and adjust the energy
						# for idx, user in enumerate(uav.get_current_node().get_user_list()):
						#     user.calculate_consumed_energy()
						#     user.adjust_energy(user.get_current_consumed_energy())
						
						# Adjust UAV energy for processing the offloaded data
						uav.energy_to_process_data(energy_coefficient= 0.1)
						
						# Decreases the user data in bits based on the offloaded data
						for idx, user in enumerate(uav.get_current_node().get_user_list()):
							user.set_user_strategy(user_strategies[idx])
							user.calculate_remaining_data()
						uav.get_current_node().calculate_total_bit_data()
						done = True
						
				# Calculate total data in bits processed by the UAV in this node
				total_data_processed = 0
				for user in self.uav.get_current_node().get_user_list():
					total_data_processed += user.get_current_strategy() * user.get_user_bits()
					
				self.uav.update_total_processed_data(total_data_processed)
				
				uav.set_finished_business_in_node(True)
				uav.hover_over_node(time_hover= T)
				self.logger.info("The UAV has finished its business in the current node")
				self.logger.info("The strategies at node %s have converged to: %s", uav.get_current_node().get_node_id(), user_strategies)
				# Log the task_intensity of the users
				task_intensities = []
				for user in uav.get_current_node().get_user_list():
					task_intensities.append(user.get_task_intensity())
				self.logger.info("The task intensities of the users at node %s are: %s", uav.get_current_node().get_node_id(), task_intensities)
				self.logger.info("Converged with strategy difference: %s in %d iterations", strategy_difference, iteration_counter)
				
				if (uav_has_reached_final_node):
					self.logger.info("The UAV has reached the final node and has finished its business")
					break_flag = True
				#print(f"Final Strategies: {user_strategies}")
			else:
				# Decide to which node to move next randomly from the ones availalbe that are not visited
				next_node = uav.get_random_unvisited_next_node_with_b_logit(nodes= graph.get_nodes(), key= key, max_iter= max_iter, beta= beta)
				if (next_node is not None):
					if (uav.travel_to_node(next_node)):
						self.logger.info("The UAV has reached the next node")
						self.logger.info("The UAV energy level is: %s after going to the next node", uav.get_energy_level())
						# Check if the UAV has reached the final node
						if uav.check_if_final_node(uav.get_current_node()):
							self.logger.info("The UAV has reached the final node")
							uav_has_reached_final_node = True
					else:
						self.logger.info("The UAV has not reached the next node because it has not enough energy")
						break_flag = True
				else:
					self.logger.info("The UAV has visited all the nodes")
					break_flag = True
		trajectory = uav.get_visited_nodes()
		trajectory_ids = []
		for node in trajectory:
			trajectory_ids.append(node.get_node_id())
			
		self.logger.info("The UAV trajectory is: %s", trajectory_ids)
		self.set_trajectory(trajectory_ids)
		
		if uav_has_reached_final_node:
			return True
		else:
			return False
		
	def brave_greedy(self, solving_method:str, max_iter: int, c: float, b: float, logger : logging.Logger = None)->bool:
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
		self.logger = logger
		self.logger.info("Running the Brave Greedy Algorithm")
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
					
				c = c
				b = b
					
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
							
							self.logger.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
							self.logger.info("User %d has maximized its utility to %s", idx, maximized_utility)
							
							# Update the user's strategy
							user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
							
							# Update user's channel gain
							user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
							
						elif solving_method == "scipy":
							# Play the submodular game
							maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, c, b, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
																									uav_cpu_frequency, uav_total_data_processing_capacity, 2, uav.get_current_coordinates(), uav.get_height())
	
							# logger.info("User %d has offloaded %f of its data", idx, percentage_offloaded[0])
							# logger.info("User %d has maximized its utility to %s", idx, maximized_utility)
							
							# Update the user's strategy
							user_strategies = user_strategies.at[idx].set(percentage_offloaded[0])
							
							# Update user's channel gain
							user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())

					# Check how different the strategies are from the previous iteration    
					strategy_difference = jnp.linalg.norm(user_strategies - previous_strategies)
					
					# Check if the strategies have converged
					if strategy_difference < convergence_threshold:
						# Calculate the consumed energy for all users based on the strategies they have chosen and adjust the energy
						# for idx, user in enumerate(uav.get_current_node().get_user_list()):
						#     user.calculate_consumed_energy()
						#     user.adjust_energy(user.get_current_consumed_energy())
						
						# Adjust UAV energy for processing the offloaded data
						uav.energy_to_process_data(energy_coefficient= 0.1)
						
						# Decreases the user data in bits based on the offloaded data
						for idx, user in enumerate(uav.get_current_node().get_user_list()):
							user.set_user_strategy(user_strategies[idx])
							user.calculate_remaining_data()
						uav.get_current_node().calculate_total_bit_data()
						
						done = True
					
				# Calculate total data in bits processed by the UAV in this node
				total_data_processed = 0
				for user in self.uav.get_current_node().get_user_list():
					total_data_processed += user.get_current_strategy() * user.get_user_bits()
					
				self.uav.update_total_processed_data(total_data_processed)
				
				uav.set_finished_business_in_node(True)
				uav.hover_over_node(time_hover= T)
				self.logger.info("The UAV has finished its business in the current node")
				self.logger.info("The strategies at node %s have converged to: %s", uav.get_current_node().get_node_id(), user_strategies)
				# Log the task_intensity of the users
				task_intensities = []
				for user in uav.get_current_node().get_user_list():
					task_intensities.append(user.get_task_intensity())
				self.logger.info("The task intensities of the users at node %s are: %s", uav.get_current_node().get_node_id(), task_intensities)
				self.logger.info("Converged with strategy difference: %s in %d iterations", strategy_difference, iteration_counter)
				
				if (uav_has_reached_final_node):
					logger.info("The UAV has reached the final node and has finished its business")
					break_flag = True
				#print(f"Final Strategies: {user_strategies}")
			else:
				# Decide to which node to move next randomly from the ones availalbe that are not visited
				next_node = uav.get_brave_greedy_next_node(nodes= graph.get_nodes(), max_iter= max_iter)
				if (next_node is not None):
					if (uav.travel_to_node(next_node)):
						self.logger.info("The UAV has reached the next node")
						# Check if the UAV has reached the final node
						if uav.check_if_final_node(uav.get_current_node()):
							self.logger.info("The UAV has reached the final node")
							uav_has_reached_final_node = True
					else:
						self.logger.info("The UAV has not reached the next node because it has not enough energy")
						break_flag = True
				else:
					self.logger.info("The UAV has visited all the nodes")
					break_flag = True
		trajectory = uav.get_visited_nodes()
		trajectory_ids = []
		for node in trajectory:
			trajectory_ids.append(node.get_node_id())
			
		self.logger.info("The UAV trajectory is: %s", trajectory_ids)
		self.set_trajectory(trajectory_ids)
		
		if uav_has_reached_final_node:
			return True
		else:
			return False
		
	def q_brave(self, solving_method:str, number_of_episodes: int, max_travels_per_episode: int, c: float, b: float, logger : logging.Logger = None)->bool:
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
		self.logger = logger
		self.logger.info("Running the Q-Brave Algorithm")
		uav = self.get_uav()
		graph = self.get_graph()
		U = self.get_number_of_users()
		convergence_threshold = self.get_convergence_threshold()
		T = 2
		
		n_observations = len(graph.get_nodes()) # The State resembles a node in the graph. Therefore, the number of observations is equal to the number of nodes
		n_actions = n_observations # The Uav can travel to any node from any node (including itself)
		Q_table = jnp.zeros((n_observations,n_actions)) # Initialize the Q-table with zeros for all state-action pairs
		action_node_list = [node for node in graph.get_nodes()]

		#initialize the exploration probability to 1
		exploration_proba = 1

		#exploartion decreasing decay for exponential decreasing
		exploration_decreasing_decay = 0.001

		# minimum of exploration proba
		min_exploration_proba = 0.01

		#discounted factor
		gamma = 0.99

		#learning rate
		lr = 0.1
		
		env = Qenv(graph= graph, uav= uav, number_of_users= U, convergence_threshold= convergence_threshold,
				   n_actions= n_actions, n_observations= n_observations, solving_method= solving_method, T= T, c= c, b= b, max_iter= max_travels_per_episode)
		
		rewards_per_episode = []
		total_bits_processed_per_episode = []
		energy_expended_per_episode = []
		uav_visited_nodes_per_episode = []
		#we iterate over episodes
		for e in tqdm(range(number_of_episodes), desc= "Running Q-Brave Algorithm"):
			
			#we initialize the first state of the episode
			#print("\nEPISODE START")
			self.logger.info("EPISODE START")
			self.reset()
			current_state = env.reset(graph= self.get_graph(), uav= self.get_uav(),)
			current_state = current_state.get_node_id()
			done = False
			
			#sum the rewards that the agent gets from the environment
			total_episode_reward = 0
			
			for i in range(max_travels_per_episode):
				# we sample a float from a uniform distribution over 0 and 1
				# if the sampled flaot is less than the exploration probability
				#     then the agent selects a random action
				# else
				#     he exploits his knowledge using the bellman equation  
				
				if random.uniform(random.split(self.key)[0], (1,))[0] < exploration_proba:
					action = env.action_space.sample()
				else:
					action = jnp.argmax(Q_table[current_state,:])
				
				action_node = action_node_list[action]
				# The environment runs the chosen action and returns
				# the next state, a reward and true if the epiosed is ended.
				returns = env.step(action_node)
				next_state, reward, done, info = returns
				next_state = next_state.get_node_id()
				
				# We update our Q-table using the Q-learning iteration
				Q_table = Q_table.at[current_state, action].set((1-lr) * Q_table[current_state, action] +lr*(reward + gamma*max(Q_table[next_state,:])))
				total_episode_reward = total_episode_reward + reward
				# If the episode is finished, we leave the for loop
				if done:
					logger.info("The episode has finished")
					break
				current_state = next_state
			#We update the exploration proba using exponential decay formula
			exploration_proba = max(min_exploration_proba, jnp.exp(-exploration_decreasing_decay*e))
			rewards_per_episode.append(total_episode_reward)
			total_bits_processed_per_episode.append(self.get_uav().get_total_processed_data())
			energy_expended_per_episode.append(self.get_uav().get_total_energy_level() - self.get_uav().get_energy_level())
			trajectory = uav.get_visited_nodes()
			trajectory_ids = []
			for node in info['visited_nodes']:
				trajectory_ids.append(node.get_node_id())
			uav_visited_nodes_per_episode.append(trajectory_ids)
			self.logger.info("The reward for episode %d is: %s", e, total_episode_reward)
			self.logger.info("The UAV has visited %d nodes in episode %d", len(trajectory_ids), e)
			self.logger.info("The UAV has visited the nodes: %s", trajectory_ids)
			self.logger.info("EPISODE FINISHED")
			#print("EPISODE END")
			
		# print("The Q-table is: ", Q_table)
		# print("The rewards per episode are: ", rewards_per_episode)
		# print("Mean reward per episode: ", jnp.mean(jnp.array(rewards_per_episode)))
		# print("Max reward: ", jnp.max(jnp.array(rewards_per_episode)))
		self.logger.info("The rewards per episode are: %s", rewards_per_episode)
		self.logger.info("Q Table is: %s", Q_table)
		self.logger.info("Mean reward per episode: %s", jnp.mean(jnp.array(rewards_per_episode)))
		self.logger.info("Max reward: %s", jnp.max(jnp.array(rewards_per_episode)))
  
		# Find at which episode the Q-Learning algorithm had the best reward
		best_episode = jnp.argmax(jnp.array(rewards_per_episode))
  
		# Set the Information used for the plots by getting the information on the best episode
		self.set_most_processed_bits(total_bits_processed_per_episode[best_episode])
		self.set_most_energy_expended(energy_expended_per_episode[best_episode])
		self.set_most_visited_nodes(uav_visited_nodes_per_episode[best_episode])
		self.set_best_trajectory(uav_visited_nodes_per_episode[best_episode])
			
		return True

	def multi_agent_q_learning_coop(self, solving_method:str, number_of_episodes: int, max_travels_per_episode: int, c: float, b: float, logger : logging.Logger = None)->bool:
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
		self.logger = logger
		self.logger.info("Running the Multi-Agent Coop Q-Brave Algorithm")
		uavs = self.get_uavs()
		graph = self.get_graph()
		U = self.get_number_of_users()
		convergence_threshold = self.get_convergence_threshold()
		T = 2
		
		n_observations = len(graph.get_nodes()) # The State resembles a node in the graph. Therefore, the number of observations is equal to the number of nodes
		n_actions = n_observations # The Uav can travel to any node from any node (including itself)
		Q_table = jnp.zeros((n_observations,n_actions)) # Initialize the Q-table with zeros for all state-action pairs
		action_node_list = [node for node in graph.get_nodes()]

		#initialize the exploration probability to 1
		exploration_proba = 1

		#exploartion decreasing decay for exponential decreasing
		exploration_decreasing_decay = 0.001

		# minimum of exploration proba
		min_exploration_proba = 0.01

		#discounted factor
		gamma = 0.99

		#learning rate
		lr = 0.1
		
		env = Multiagent_Qenv(graph= graph, uavs= uavs, number_of_users= U, convergence_threshold= convergence_threshold,
				   n_actions= n_actions, n_observations= n_observations, solving_method= solving_method, T= T, c= c, b= b, max_iter= max_travels_per_episode)
		
		rewards_per_episode = []
		total_bits_processed_per_episode = []
		energy_expended_per_episode = []
		uav_visited_nodes_per_episode = []
		uav_trajectory_per_episode = []
    
		#we iterate over episodes
		for e in tqdm(range(number_of_episodes), desc= "Running Q-Brave Algorithm"):
			
			#we initialize the first state of the episode
			#print("\nEPISODE START")
			self.logger.info("EPISODE START")
			self.reset()
			current_state = env.reset(graph= self.get_graph(), uavs= self.get_uavs(),)
			current_state_temp = []
			for state in current_state:
				current_state_temp.append(state.get_node_id())
			current_state = current_state_temp
			done = False
			
			#sum the rewards that the agent gets from the environment
			total_episode_reward = 0
			
			for i in range(max_travels_per_episode):
				# we sample a float from a uniform distribution over 0 and 1
				# if the sampled flaot is less than the exploration probability
				#     then the agent selects a random action
				# else
				#     he exploits his knowledge using the bellman equation
				#sum the rewards that the agent gets from the environment

				
				uav_actions = []
				for i in range(len(uavs)):
					if random.uniform(random.split(self.key)[0], (1,))[0] < exploration_proba:
						action = env.action_space.sample()[0]
					else:
						action = jnp.argmax(Q_table[current_state[i],:])
					uav_actions.append(action)
				
				action_nodes = []
				for i in range(len(uav_actions)):
					action_nodes.append(action_node_list[i])
     
				# The environment runs the chosen action and returns
				# the next state, a reward and true
				# The environment runs the chosen action and returns
				# the next state, a reward and true if the epiosed is ended.
				returns = env.step(action_nodes)
				next_state, reward, done, info = returns
				next_state_temp = []
				for state in next_state:
					next_state_temp.append(state.get_node_id())
				next_state = next_state_temp
				
				for j in range(len(uav_actions)):
					# We update the shared Q-table using the Q-learning iteration
					Q_table = Q_table.at[current_state[j], uav_actions[j]].set((1-lr) * Q_table[current_state[j], uav_actions[j]] +lr*(reward[j] + gamma*max(Q_table[next_state[j],:])))
					total_episode_reward = total_episode_reward + reward[j]
				# If the episode is finished, we leave the for loop
				if done:
					logger.info("The episode has finished")
					break
				current_state = next_state
    
			#We update the exploration proba using exponential decay formula
			exploration_proba = max(min_exploration_proba, jnp.exp(-exploration_decreasing_decay*e))
			rewards_per_episode.append(total_episode_reward)
   
			total_bits_processed = 0
			for uav in uavs:
				total_bits_processed += uav.get_total_processed_data()
			total_bits_processed_per_episode.append(total_bits_processed)
			
			energy_expended = 0
			for uav in uavs:
				energy_expended += uav.get_total_energy_level() - uav.get_energy_level()
			energy_expended_per_episode.append(energy_expended)
   
			trajectory = []
			for uav in uavs:
				trajectory.append(uav.get_visited_nodes())
	
			trajectories = []		
			total_visited_nodes = 0
			for uav in info['visited_nodes']:
				trajectory_ids = []
				for node in uav:
					trajectory_ids.append(node.get_node_id())
				trajectories.append(trajectory_ids)
				total_visited_nodes += len(trajectory_ids)
     
			uav_visited_nodes_per_episode.append(total_visited_nodes)
			uav_trajectory_per_episode.append(trajectories)
   
			self.logger.info("The total reward for episode %d is: %s", e, total_episode_reward)
			self.logger.info("The UAVs have visited %d nodes in episode %d", total_visited_nodes, e)
			self.logger.info("The UAVs have visited the nodes: %s", trajectories)
			self.logger.info("The UAVs have expended %s energy in episode %d", energy_expended, e)
			self.logger.info("EPISODE FINISHED")
			#print("EPISODE END")
			
		# print("The Q-table is: ", Q_table)
		# print("The rewards per episode are: ", rewards_per_episode)
		# print("Mean reward per episode: ", jnp.mean(jnp.array(rewards_per_episode)))
		# print("Max reward: ", jnp.max(jnp.array(rewards_per_episode)))
		self.logger.info("The rewards per episode are: %s", rewards_per_episode)
		self.logger.info("Q Table is: %s", Q_table)
		self.logger.info("Mean reward per episode: %s", jnp.mean(jnp.array(rewards_per_episode)))
		self.logger.info("Max reward: %s", jnp.max(jnp.array(rewards_per_episode)))
  
		# Find at which episode the Q-Learning algorithm had the best reward
		best_episode = jnp.argmax(jnp.array(rewards_per_episode))
  
		# Set the Information used for the plots by getting the information on the best episode
		self.set_most_processed_bits(total_bits_processed_per_episode[best_episode])
		self.set_most_energy_expended(energy_expended_per_episode[best_episode])
		self.set_most_visited_nodes(uav_visited_nodes_per_episode[best_episode])
		self.set_best_trajectories(uav_trajectory_per_episode[best_episode])
			
		return True

	def multi_agent_q_learning_indi(self, solving_method:str, number_of_episodes: int, max_travels_per_episode: int, c: float, b: float, logger : logging.Logger = None)->bool:
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
		self.logger = logger
		self.logger.info("Running the Multi-Agent Coop Q-Brave Algorithm")
		uavs = self.get_uavs()
		graph = self.get_graph()
		U = self.get_number_of_users()
		convergence_threshold = self.get_convergence_threshold()
		T = 2
		
		n_observations = len(graph.get_nodes()) # The State resembles a node in the graph. Therefore, the number of observations is equal to the number of nodes
		n_actions = n_observations # The Uav can travel to any node from any node (including itself)
  
		Q_tables = []
		for i in range(len(uavs)):
			Q_tables.append(jnp.zeros((n_observations,n_actions)))
   
   		# Initialize the Q-table with zeros for all state-action pairs
		action_node_list = [node for node in graph.get_nodes()]

		#initialize the exploration probability to 1
		exploration_proba = 1

		#exploartion decreasing decay for exponential decreasing
		exploration_decreasing_decay = 0.001

		# minimum of exploration proba
		min_exploration_proba = 0.01

		#discounted factor
		gamma = 0.99

		#learning rate
		lr = 0.1
		
		env = Multiagent_Qenv(graph= graph, uavs= uavs, number_of_users= U, convergence_threshold= convergence_threshold,
				   n_actions= n_actions, n_observations= n_observations, solving_method= solving_method, T= T, c= c, b= b, max_iter= max_travels_per_episode)
		
		rewards_per_episode = []
		total_bits_processed_per_episode = []
		energy_expended_per_episode = []
		uav_visited_nodes_per_episode = []
		uav_trajectory_per_episode = []
    
		#we iterate over episodes
		for e in tqdm(range(number_of_episodes), desc= "Running Q-Brave Algorithm"):
			
			#we initialize the first state of the episode
			#print("\nEPISODE START")
			self.logger.info("EPISODE START")
			self.reset()
			current_state = env.reset(graph= self.get_graph(), uavs= self.get_uavs(),)
			current_state_temp = []
			for state in current_state:
				current_state_temp.append(state.get_node_id())
			current_state = current_state_temp
			done = False
			
			#sum the rewards that the agent gets from the environment
			total_episode_reward = 0
			
			for i in range(max_travels_per_episode):
				# we sample a float from a uniform distribution over 0 and 1
				# if the sampled flaot is less than the exploration probability
				#     then the agent selects a random action
				# else
				#     he exploits his knowledge using the bellman equation
				#sum the rewards that the agent gets from the environment

				
				uav_actions = []
				for i in range(len(uavs)):
					if random.uniform(random.split(self.key)[0], (1,))[0] < exploration_proba:
						action = env.action_space.sample()[0]
					else:
						action = jnp.argmax(Q_tables[i][current_state[i],:])
					uav_actions.append(action)
				
				action_nodes = []
				for i in range(len(uav_actions)):
					action_nodes.append(action_node_list[i])
     
				# The environment runs the chosen action and returns
				# the next state, a reward and true
				# The environment runs the chosen action and returns
				# the next state, a reward and true if the epiosed is ended.
				returns = env.step(action_nodes)
				next_state, reward, done, info = returns
				next_state_temp = []
				for state in next_state:
					next_state_temp.append(state.get_node_id())
				next_state = next_state_temp
				
				for j in range(len(uav_actions)):
					# We update the shared Q-table using the Q-learning iteration
					Q_tables[j] = Q_tables[j].at[current_state[j], uav_actions[j]].set((1-lr) * Q_tables[j][current_state[i], uav_actions[j]] +lr*(reward[j] + gamma*max(Q_tables[j][next_state[j],:])))
					total_episode_reward = total_episode_reward + reward[j]
				# If the episode is finished, we leave the for loop
				if done:
					logger.info("The episode has finished")
					break
				current_state = next_state
    
			#We update the exploration proba using exponential decay formula
			exploration_proba = max(min_exploration_proba, jnp.exp(-exploration_decreasing_decay*e))
			rewards_per_episode.append(total_episode_reward)
   
			total_bits_processed = 0
			for uav in uavs:
				total_bits_processed += uav.get_total_processed_data()
			total_bits_processed_per_episode.append(total_bits_processed)
			
			energy_expended = 0
			for uav in uavs:
				energy_expended += uav.get_total_energy_level() - uav.get_energy_level()
			energy_expended_per_episode.append(energy_expended)
   
			trajectory = []
			for uav in uavs:
				trajectory.append(uav.get_visited_nodes())
	
			trajectories = []		
			total_visited_nodes = 0
			for uav in info['visited_nodes']:
				trajectory_ids = []
				for node in uav:
					trajectory_ids.append(node.get_node_id())
				trajectories.append(trajectory_ids)
				total_visited_nodes += len(trajectory_ids)
     
			uav_visited_nodes_per_episode.append(total_visited_nodes)
			uav_trajectory_per_episode.append(trajectories)
   
			self.logger.info("The total reward for episode %d is: %s", e, total_episode_reward)
			self.logger.info("The UAVs have visited %d nodes in episode %d", total_visited_nodes, e)
			self.logger.info("The UAVs have visited the nodes: %s", trajectories)
			self.logger.info("The UAVs have expended %s energy in episode %d", energy_expended, e)
			self.logger.info("EPISODE FINISHED")
			#print("EPISODE END")
			
		# print("The Q-table is: ", Q_table)
		# print("The rewards per episode are: ", rewards_per_episode)
		# print("Mean reward per episode: ", jnp.mean(jnp.array(rewards_per_episode)))
		# print("Max reward: ", jnp.max(jnp.array(rewards_per_episode)))
		self.logger.info("The rewards per episode are: %s", rewards_per_episode)
		self.logger.info("Q Tables are: %s", Q_tables)
		self.logger.info("Mean reward per episode: %s", jnp.mean(jnp.array(rewards_per_episode)))
		self.logger.info("Max reward: %s", jnp.max(jnp.array(rewards_per_episode)))
  
		# Find at which episode the Q-Learning algorithm had the best reward
		best_episode = jnp.argmax(jnp.array(rewards_per_episode))
  
		# Set the Information used for the plots by getting the information on the best episode
		self.set_most_processed_bits(total_bits_processed_per_episode[best_episode])
		self.set_most_energy_expended(energy_expended_per_episode[best_episode])
		self.set_most_visited_nodes(uav_visited_nodes_per_episode[best_episode])
		self.set_best_trajectories(uav_trajectory_per_episode[best_episode])
			
		return True