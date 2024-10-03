import logging
from jax import random
from Algorithms import Algorithms
import pickle

# Create a random key
key = random.PRNGKey(20)

# Create N nodes with U users in them
N = 15
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

# Run the Submodular Game Algorithm
convergence_history = algorithm.run_single_submodular_game(solving_method= "scipy")
print(convergence_history)

# Dump the convergence history to a pickle file
with open('convergence_history.pkl', 'wb') as handle:
    pickle.dump(convergence_history, handle)