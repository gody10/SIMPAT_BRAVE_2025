import logging
from jax import random
from Algorithms import Algorithms
import pickle
import time
import matplotlib.pyplot as plt
from Utility_functions import setup_logger
from tqdm import tqdm
import numpy as np
import jax.numpy as jnp

# -------------------- Configuration --------------------

# Number of runs
NUM_RUNS = 500

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
NUMBER_OF_EPISODES = 15

# Initialize the main random key
main_key = random.PRNGKey(10)

# Initialize the logger
q_brave_logger = setup_logger('q_learning_realistic_scenario', 'q_learning_realistic_scenario.log')

# Initialize accumulators for metrics
all_count_arrays = []

# Initialize the algorithm once outside the loop
algorithm = Algorithms(convergence_threshold=CONVERGENCE_THRESHOLD)

# -------------------- Multiple Runs --------------------

for run in tqdm(range(NUM_RUNS), desc="Running Q-Brave Algorithm"):
    # Split the main key to get a new key for this run
    main_key, run_key = random.split(main_key)
    
    # Setup subkeys for nodes
    subkeys = random.split(run_key, N)  # Split into as many subkeys as there are nodes
    
    # Generate a unique number of users for each node using the subkey
    U = [int(random.randint(main_key, (1,), 15, 15)) for subkey in subkeys]
    
    # Setup the algorithm for this run
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
    # Run the Q-Brave Algorithm
    success_q_brave = algorithm.q_brave(
        solving_method="scipy",
        number_of_episodes=NUMBER_OF_EPISODES,
        max_travels_per_episode=MAX_ITER,
        b=B,
        c=C,
        logger=q_brave_logger  # Disable logging for faster runs
    )
    
    # Get UAV trajectory
    trajectory = algorithm.get_best_trajectory()
    trajectory.append(5)  # Assuming '5' is a special node or end point
    
    # Count how many times the UAV visited each node based on the trajectory
    count_arrays = jnp.zeros(N)
    for node in trajectory:
        if 0 <= node < N:
            count_arrays = count_arrays.at[node].add(1)
        else:
            pass  # Handle nodes outside the expected range if necessary
    
    # Append metrics to accumulators
    all_count_arrays.append(np.array(count_arrays))
    
    # Reset the Algorithm object for the next run
    algorithm.reset()

# -------------------- Compute Averages --------------------

# Convert list to numpy array for easier computation
all_count_arrays = np.array(all_count_arrays)  # Shape: (NUM_RUNS, N)

# Compute averages
average_count_arrays = np.mean(all_count_arrays, axis=0)

# Save to pickle file
with open('q_brave_visits_average.pkl', 'wb') as f:
    pickle.dump(average_count_arrays, f)

# -------------------- Logging Averages --------------------

q_brave_logger.info("After %d runs:", NUM_RUNS)
q_brave_logger.info("Average number of visits to each node: %s", average_count_arrays)

# -------------------- Plotting Averages --------------------

# Plot the average number of times the UAV visited each node
plt.figure(figsize=(10, 6))
plt.bar(np.arange(N), average_count_arrays, color='blue')
plt.xlabel('Node ID')
plt.ylabel('Average Number of Visits')
plt.title('Average Number of Visits to Each Node Over 500 Runs')
plt.xticks(np.arange(N))
plt.savefig('q_brave_visits_average.png')
plt.close()

# -------------------- Summary --------------------
print(f"Completed {NUM_RUNS} runs.")
print(f"Average Number of Visits to Each Node: {average_count_arrays}")