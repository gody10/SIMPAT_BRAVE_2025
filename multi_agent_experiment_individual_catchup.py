import logging
from jax import random
from Algorithms import Algorithms
import pickle
import time
from plot_graphs_multi_agent_vs import plot_graphs_multi_agent_vs
from Utility_functions import setup_logger, compute_average
from tqdm import tqdm
import os

# Initialize timers for each method
q_brave_time = 0

# Create N nodes with U users in them
N = 10
NODE_RADIUS = 2
MIN_DISTANCE_BETWEEN_NODES = 40  # Minimum distance to maintain between nodes
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
NUMBER_OF_EPISODES = 20
NUMBER_OF_UAVS = 3

# Initialize the main random key
main_key = random.PRNGKey(10)

# Initialize accumulation dictionaries
algorithms_total_bits_acc = {}
algorithms_expended_energy_acc = {}
algorithms_total_visited_nodes_acc = {}
timer_dict_acc = {}

# List of algorithms
algorithm_names = ["Q-Learning Common Table", "Q-Learning Individual Tables", "Q-Learning Individual Tables Double Episodes", 
                   "Q-Learning Individual Tables Triple Episodes"]

# Initialize accumulation dictionaries with empty lists for each algorithm
for name in algorithm_names:
    algorithms_total_bits_acc[f"{name} Total Bits"] = []
    algorithms_expended_energy_acc[f"{name} Energy Level"] = []
    algorithms_total_visited_nodes_acc[f"{name} Total Visited Nodes"] = []
    timer_dict_acc[f"{name} Time"] = [] 

# Initialize the logger
multi_q_learning_logger = setup_logger('multi_q_learning', 'multi_q_learning.log')

# Initialize the algorithm
algorithm = Algorithms(convergence_threshold= CONVERGENCE_THRESHOLD)

# Setup subkeys
subkey = random.split(main_key)[0]

# Generate a unique number of users for each node using the subkey
U = [random.randint(subkey, (N,), 15, 15)]  # Adjust min and max values as needed

# Setup the algorithm 
algorithm.setup_multiagent_scenario(
        number_of_nodes=N,
        number_of_users=U,
        node_radius=NODE_RADIUS,
        key=subkey,
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
        energy_level=ENERGY_LEVEL,
        number_of_uavs=NUMBER_OF_UAVS
    )

# -------------------- Multi Q-Brave Solution with Common Q-table --------------------
start_time = time.time()

# Run the Algorithm
success_common = algorithm.multi_agent_q_learning_coop(
    solving_method="scipy",
    number_of_episodes=NUMBER_OF_EPISODES,
    max_travels_per_episode=MAX_ITER,
    b=B,
    c=C,
    logger= multi_q_learning_logger,
)

# Get UAV trajectory and the number of bits processed at each node
processed_bits = algorithm.get_most_processed_bits()
energy_expended = algorithm.get_most_expended_energy()
total_visited_nodes = algorithm.get_most_visited_nodes()
trajectory = algorithm.get_best_trajectories()

# End the timer for Q-Brave Algorithm
q_common_time = time.time() - start_time
multi_q_learning_logger.info("Q-Brave Algorithm took: %s seconds", q_common_time)
    
# Add the results to the accumulation dictionaries
algorithms_total_bits_acc["Q-Learning Common Table Total Bits"].append(processed_bits)
algorithms_expended_energy_acc["Q-Learning Common Table Energy Level"].append(energy_expended)
algorithms_total_visited_nodes_acc["Q-Learning Common Table Total Visited Nodes"].append(total_visited_nodes)
timer_dict_acc["Q-Learning Common Table Time"].append(q_common_time)

# Reset the Algorithm object for the next run
algorithm.reset()

# -------------------- Multi Q-Brave Solution with Individual Q-tables --------------------
start_time = time.time()

# Run the Algorithm
success_individual = algorithm.multi_agent_q_learning_indi(
    solving_method="scipy",
    number_of_episodes=NUMBER_OF_EPISODES,
    max_travels_per_episode=MAX_ITER,
    b=B,
    c=C,
    logger= multi_q_learning_logger,
)

# Get UAV trajectory and the number of bits processed at each node
processed_bits = algorithm.get_most_processed_bits()
energy_expended = algorithm.get_most_expended_energy()
total_visited_nodes = algorithm.get_most_visited_nodes()
trajectory = algorithm.get_best_trajectory()

# End the timer for Q-Brave Algorithm
q_individual_time = time.time() - start_time
multi_q_learning_logger.info("Q-Brave Algorithm took: %s seconds", q_individual_time)

# Add the results to the accumulation dictionaries
algorithms_total_bits_acc["Q-Learning Individual Tables Total Bits"].append(processed_bits)
algorithms_expended_energy_acc["Q-Learning Individual Tables Energy Level"].append(energy_expended)
algorithms_total_visited_nodes_acc["Q-Learning Individual Tables Total Visited Nodes"].append(total_visited_nodes)
timer_dict_acc["Q-Learning Individual Tables Time"].append(q_individual_time)

# Reset the Algorithm object for the next run
algorithm.reset()

# -------------------- Multi Q-Brave Solution with Individual Q-tables *2 EPISODES --------------------
start_time = time.time()

# Run the Algorithm
success_individual = algorithm.multi_agent_q_learning_indi(
    solving_method="scipy",
    number_of_episodes=NUMBER_OF_EPISODES *2,
    max_travels_per_episode=MAX_ITER,
    b=B,
    c=C,
    logger= multi_q_learning_logger,
)

# Get UAV trajectory and the number of bits processed at each node
processed_bits = algorithm.get_most_processed_bits()
energy_expended = algorithm.get_most_expended_energy()
total_visited_nodes = algorithm.get_most_visited_nodes()
trajectory = algorithm.get_best_trajectory()

# End the timer for Q-Brave Algorithm
q_individual_time = time.time() - start_time
multi_q_learning_logger.info("Q-Brave Algorithm took: %s seconds", q_individual_time)

# Add the results to the accumulation dictionaries
algorithms_total_bits_acc["Q-Learning Individual Tables Double Episodes Total Bits"].append(processed_bits)
algorithms_expended_energy_acc["Q-Learning Individual Tables Double Episodes Energy Level"].append(energy_expended)
algorithms_total_visited_nodes_acc["Q-Learning Individual Tables Double Episodes Total Visited Nodes"].append(total_visited_nodes)
timer_dict_acc["Q-Learning Individual Tables Double Episodes Time"].append(q_individual_time)

# Reset the Algorithm object for the next run
algorithm.reset()

# -------------------- Multi Q-Brave Solution with Individual Q-tables *3 EPISODES --------------------
start_time = time.time()

# Run the Algorithm
success_individual = algorithm.multi_agent_q_learning_indi(
    solving_method="scipy",
    number_of_episodes=NUMBER_OF_EPISODES *3,
    max_travels_per_episode=MAX_ITER,
    b=B,
    c=C,
    logger= multi_q_learning_logger,
)

# Get UAV trajectory and the number of bits processed at each node
processed_bits = algorithm.get_most_processed_bits()
energy_expended = algorithm.get_most_expended_energy()
total_visited_nodes = algorithm.get_most_visited_nodes()
trajectory = algorithm.get_best_trajectory()

# End the timer for Q-Brave Algorithm
q_individual_time = time.time() - start_time
multi_q_learning_logger.info("Q-Brave Algorithm took: %s seconds", q_individual_time)

# Add the results to the accumulation dictionaries
algorithms_total_bits_acc["Q-Learning Individual Tables Triple Episodes Total Bits"].append(processed_bits)
algorithms_expended_energy_acc["Q-Learning Individual Tables Triple Episodes Energy Level"].append(energy_expended)
algorithms_total_visited_nodes_acc["Q-Learning Individual Tables Triple Episodes Total Visited Nodes"].append(total_visited_nodes)
timer_dict_acc["Q-Learning Individual Tables Triple Episodes Time"].append(q_individual_time)

# Reset the Algorithm object for the next run
algorithm.reset()

# Save the dictionaries to different pickle files
with open('multi_q_learning_total_bits.pkl', 'wb') as f:
    pickle.dump(algorithms_total_bits_acc, f)
    
with open('multi_q_learning_expended_energy.pkl', 'wb') as f:
    pickle.dump(algorithms_expended_energy_acc, f)  
    
with open('multi_q_learning_total_visited_nodes.pkl', 'wb') as f:
    pickle.dump(algorithms_total_visited_nodes_acc, f)
    
with open('multi_q_learning_timers.pkl', 'wb') as f:
    pickle.dump(timer_dict_acc, f)
    
# Plot the graphs
plot_graphs_multi_agent_vs(folder_path= f'marl_coop_vs_individual_{NUMBER_OF_EPISODES}_episodes/')
