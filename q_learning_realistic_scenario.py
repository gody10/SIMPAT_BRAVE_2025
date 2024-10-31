import logging
from jax import random
from Algorithms import Algorithms
import pickle
import time
import matplotlib.pyplot as plt
from plot_graphs import plot_graphs
from Utility_functions import setup_logger, compute_average
from tqdm import tqdm
import os
import jax.numpy as jnp

# Initialize timers for each method
q_brave_time = 0

# Create N nodes with U users in them
N = 6
NODE_RADIUS = 2
MIN_DISTANCE_BETWEEN_NODES = 10  # Minimum distance to maintain between nodes
UAV_HEIGHT = 30
CONVERGENCE_THRESHOLD = 1e-15
UAV_ENERGY_CAPACITY = 1065600
UAV_BANDWIDTH = 5*(10**6)
UAV_PROCESSING_CAPACITY = 1 * (10**9)
UAV_CPU_FREQUENCY = 2 * (10**9)
UAV_VELOCITY = 1
DISTANCE_MIN = 10
DISTANCE_MAX = 67
MAX_BITS = 2 * 10**6
MIN_BITS = 3 * 10**5
ENERGY_LEVEL = 29000
B = 0.74
C = 0.00043
MAX_ITER = 30
NUMBER_OF_EPISODES = 50

# Initialize the main random key
main_key = random.PRNGKey(10)

# Initialize the logger
q_brave_logger = setup_logger('q_learning_realistic_scenario', 'q_learning_realistic_scenario.log')

# Initialize the algorithm
algorithm = Algorithms(convergence_threshold= CONVERGENCE_THRESHOLD)

# Setup subkeys
subkeys = random.split(main_key, N)  # Split into as many subkeys as there are nodes

# Generate a unique number of users for each node using the subkey
U = [random.randint(subkey, (1,), 15, 15) for subkey in subkeys]

# Setup the algorithm 
algorithm.setup_realistic_scenario(
        number_of_nodes=N,
        number_of_users=U,
        node_radius=NODE_RADIUS,
        key=main_key,
        min_distance_between_nodes=MIN_DISTANCE_BETWEEN_NODES,
        uav_height=UAV_HEIGHT,
        uav_energy_capacity=UAV_ENERGY_CAPACITY,
        uav_bandwidth=UAV_BANDWIDTH,
        uav_processing_capacity=UAV_PROCESSING_CAPACITY,
        uav_cpu_frequency=UAV_CPU_FREQUENCY,
        uav_velocity=UAV_VELOCITY,
        min_bits=MIN_BITS,
        max_bits=MAX_BITS,
        distance_min=DISTANCE_MIN,
        distance_max=DISTANCE_MAX,
        energy_level=ENERGY_LEVEL
    )

# -------------------- Q-Brave Algorithm --------------------
start_time = time.time()

# Run the Q-Brave Algorithm
success_q_brave_ = algorithm.q_brave(
    solving_method="scipy",
    number_of_episodes=NUMBER_OF_EPISODES,
    max_travels_per_episode=MAX_ITER,
    b=B,
    c=C,
    logger=q_brave_logger
)

q_brave_logger.info("The UAV energy level is: %s", algorithm.get_uav().get_energy_level())
q_brave_logger.info("The UAV processed in total: %s bits", algorithm.most_processed_bits)

# Get UAV trajectory and the number of bits processed at each node
processed_bits = algorithm.get_most_processed_bits()
energy_expended = algorithm.get_most_expended_energy()
total_visited_nodes = algorithm.get_most_visited_nodes()
trajectory = algorithm.get_best_trajectory()
trajectory.append(5)

# End the timer for Q-Brave Algorithm
q_brave_time = time.time() - start_time
q_brave_logger.info("Q-Brave Algorithm took: %s seconds", q_brave_time)

if success_q_brave_:
    q_brave_logger.info("Q-Brave Algorithm has successfully reached the final node!")
else:
    q_brave_logger.info("Q-Brave Algorithm failed to reach the final node!")
    
# Count how many times the UAV visited each node based on the trajectory
count_arrays = jnp.zeros(N)

# Count visits based on trajectory data
for node in trajectory:
    count_arrays = count_arrays.at[node].add(1)
    
# Log the UAV trajectory
q_brave_logger.info("The UAV trajectory is: %s", trajectory)

# Plot as a histogram the number of times the UAV visited each node
plt.figure()

plt.bar(jnp.arange(0, N), count_arrays, color='blue')

plt.xlabel('Node ID')
plt.ylabel('Number of Visits')

plt.title('Number of Visits to Each Node')

plt.savefig('q_brave_visits.png')


# Reset the Algorithm object for the next run
algorithm.reset()