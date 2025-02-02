from jax import random
from Algorithms import Algorithms
import pickle
import jax.numpy as jnp
import logging
from plot_graphs import plot_graphs
from Utility_functions import setup_logger
import time

system_logger = setup_logger('single_game', 'single_game.log')

# Create a random key
key = random.PRNGKey(20)

# Create N nodes with U users in them
N = 1
#U = 100
# Generate random user number for each node
U = random.randint(key, (N,), 10, 10)
NODE_RADIUS = 2
MIN_DISTANCE_BETWEEN_NODES = 100  # Minimum distance to maintain between nodes
UAV_HEIGHT = 30
CONVERGENCE_THRESHOLD = 1e-15
UAV_ENERGY_CAPACITY = 19800000
UAV_BANDWIDTH = 5*(10**6)
UAV_PROCESSING_CAPACITY = 1 * (10**9)
UAV_CPU_FREQUENCY = 2 * (10**9)
UAV_VELOCITY = 1
MAX_ITER = 50
DISTANCE_MIN = 10
DISTANCE_MAX = 67
MAX_BITS = 2 * 10**6
MIN_BITS = 3 * 10**5
ENERGY_LEVEL = 29000
B = 0.74
C = 0.00043

data_dict = {}

# Start the timer
start_time = time.time()

# Create the algorithm object
algorithm = Algorithms(convergence_threshold= CONVERGENCE_THRESHOLD)

algorithm.setup_singular_experiment(number_of_nodes= N, number_of_users= U, node_radius= NODE_RADIUS, key= key, min_distance_between_nodes= MIN_DISTANCE_BETWEEN_NODES, uav_height= UAV_HEIGHT, 
						   uav_energy_capacity=UAV_ENERGY_CAPACITY, uav_bandwidth= UAV_BANDWIDTH, uav_processing_capacity= UAV_PROCESSING_CAPACITY, uav_cpu_frequency= UAV_CPU_FREQUENCY, uav_velocity= UAV_VELOCITY, 
							min_bits= MIN_BITS, max_bits= MAX_BITS, distance_min= DISTANCE_MIN, distance_max= DISTANCE_MAX, energy_level= ENERGY_LEVEL, logger= system_logger)

# Get User IDs
data_dict["User IDs"] = [user.get_user_id() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]

# Get Total Bits of each user
data_dict["User Total Bits"] = [user.get_user_bits() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]

# Run the Submodular Game Algorithm
#convergence_history = algorithm.run_single_submodular_game(solving_method= "scipy", b=B, c=C, logger= system_logger)
convergence_history = algorithm.run_single_sla_game(b= B, c= C, learning_rate= 0.7, logger= system_logger)
#print(convergence_history)

# Get time overhead of each user
data_dict["User Time Overhead"] = [user.get_current_time_overhead() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]
#print(data_dict["User Time Overhead"])

# Get total overhead of each user
data_dict["User Total Overhead"] = [user.get_current_total_overhead() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]
#print(data_dict["User Total Overhead"])

system_logger.info("Consumed Energy of each user  : {}".format([user.get_current_consumed_energy() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]))

# Get consumed Energy of each user
data_dict["User Consumed Energy"] = [(user.get_current_consumed_energy()) for user in algorithm.get_graph().get_nodes()[0].get_user_list()]

# Get utility of each user 
data_dict["User Utility"] = [user.get_user_utility() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]

# Get data rate of each user
data_dict["User Data Rate"] = [user.get_current_data_rate() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]

# Get channel gain of each user
data_dict["User Channel Gain"] = [user.get_channel_gain() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]

# Data offloaded by each user
data_dict["User Data Offloaded"] = [user.get_data_offloaded() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]

# Get the distance of each user from the Node
distance_from_node = []
d = []
node = algorithm.get_graph().get_nodes()[0]
for user in node.get_user_list():
	dist = jnp.sqrt( (node.get_coordinates()[0] - user.get_coordinates()[0])**2 + (node.get_coordinates()[1] - user.get_coordinates()[1])**2 + 
						(node.get_coordinates()[2] - user.get_coordinates()[2])**2)
	distance_from_node.append(dist)
	d.append(user.get_distance())
data_dict["User Distance from Node"] = d

# Print the minimum and maximum distance of the users from the node
system_logger.info("Minimum distance of the users from the node : {}".format(min(d)))
system_logger.info("Maximum distance of the users from the node : {}".format(max(d)))

# End the timer
end_time = time.time()

# Print the time taken for the algorithm to run
system_logger.info("Time taken for the algorithm to run : {}".format(end_time - start_time))
print("Time taken for the algorithm to run: {}".format(end_time - start_time))

# Dump the data dictionary to a pickle file
with open('user_data_dict_sla.pkl', 'wb') as handle:
	pickle.dump(data_dict, handle)

# Dump the convergence history to a pickle file
with open('convergence_history_sla.pkl', 'wb') as handle:
	pickle.dump(convergence_history, handle)
 
# Plot the graphs
#plot_graphs()