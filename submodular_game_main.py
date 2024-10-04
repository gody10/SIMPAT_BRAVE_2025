from jax import random
from Algorithms import Algorithms
import pickle

# Create a random key
key = random.PRNGKey(20)

# Create N nodes with U users in them
N = 1
#U = 100
# Generate random user number for each node
U = random.randint(key, (N,), 50, 250)
NODE_RADIUS = 2
MIN_DISTANCE_BETWEEN_NODES = 20  # Minimum distance to maintain between nodes
UAV_HEIGHT = 100
CONVERGENCE_THRESHOLD = 1e-15
UAV_ENERGY_CAPACITY = 19800000
UAV_BANDWIDTH = 15
UAV_PROCESSING_CAPACITY = 1000
UAV_CPU_FREQUENCY = 2
UAV_VELOCITY = 1
MAX_ITER = 50

data_dict = {}

# Create the algorithm object
algorithm = Algorithms(convergence_threshold= CONVERGENCE_THRESHOLD)

algorithm.setup_experiment(number_of_nodes= N, number_of_users= U, node_radius= NODE_RADIUS, key= key, min_distance_between_nodes= MIN_DISTANCE_BETWEEN_NODES, uav_height= UAV_HEIGHT, 
						   uav_energy_capacity=UAV_ENERGY_CAPACITY, uav_bandwidth= UAV_BANDWIDTH, uav_processing_capacity= UAV_PROCESSING_CAPACITY, uav_cpu_frequency= UAV_CPU_FREQUENCY, uav_velocity= UAV_VELOCITY)

# Get User IDs
data_dict["User IDs"] = [user.get_user_id() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]

# Get Total Bits of each user
data_dict["User Total Bits"] = [user.get_user_bits() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]

# Run the Submodular Game Algorithm
convergence_history = algorithm.run_single_submodular_game(solving_method= "scipy")
#print(convergence_history)

# Get time overhead of each user
data_dict["User Time Overhead"] = [user.get_current_time_overhead() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]

# Get total overhead of each user
data_dict["User Total Overhead"] = [user.get_current_total_overhead() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]

# Get consumed Energy of each user
data_dict["User Consumed Energy"] = [user.get_current_consumed_energy() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]

# Get utility of each user 
data_dict["User Utility"] = [user.get_user_utility() for user in algorithm.get_graph().get_nodes()[0].get_user_list()]

# Dump the data dictionary to a pickle file
with open('user_data_dict.pkl', 'wb') as handle:
	pickle.dump(data_dict, handle)

# Dump the convergence history to a pickle file
with open('convergence_history.pkl', 'wb') as handle:
	pickle.dump(convergence_history, handle)