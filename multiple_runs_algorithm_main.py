import logging
from jax import random
from Algorithms import Algorithms
import pickle
import time
from plot_graphs import plot_graphs
from Utility_functions import setup_logger, compute_average
from tqdm import tqdm

# Initialize timers for each method
random_walk_time = 0
proportional_fairness_time = 0
brave_greedy_time = 0
q_brave_time = 0

# Create a random key
key = random.PRNGKey(2062702539)

# Create N nodes with U users in them
N = 10
#U = 100
# Generate random user number for each node
U = random.randint(key, (N,), 15, 15)
NODE_RADIUS = 2
MIN_DISTANCE_BETWEEN_NODES = 20  # Minimum distance to maintain between nodes
UAV_HEIGHT = 100
CONVERGENCE_THRESHOLD = 1e-15
UAV_ENERGY_CAPACITY = 19800000
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
NUMBER_OF_EPISODES = 100

# Define the number of runs
NUM_RUNS = 1000

# Initialize accumulation dictionaries
algorithms_total_bits_acc = {}
algorithms_expended_energy_acc = {}
algorithms_total_visited_nodes_acc = {}
timer_dict_acc = {}

# List of algorithms
algorithm_names = ["Random Walk", "Proportional Fairness", "Brave Greedy", "Q-Brave"]

# Initialize accumulation dictionaries with empty lists for each algorithm
for name in algorithm_names:
    algorithms_total_bits_acc[f"{name} Total Bits"] = []
    algorithms_expended_energy_acc[f"{name} Energy Level"] = []
    algorithms_total_visited_nodes_acc[f"{name} Total Visited Nodes"] = []
    timer_dict_acc[f"{name} Time"] = []

# Set up the system logger
system_logger = setup_logger('system_logger', 'system.log')

# Set up separate loggers for each algorithm
random_walk_logger = setup_logger('random_walk_logger', 'random_walk_algorithm.log')
proportional_fairness_logger = setup_logger('proportional_fairness_logger', 'proportional_algorithm.log')
brave_greedy_logger = setup_logger('brave_greedy_logger', 'brave_algorithm.log')
q_brave_logger = setup_logger('q_brave_logger', 'q_brave_algorithm.log')

# Create the algorithm object once outside the loop if applicable
# If each run requires a fresh object, move this inside the loop
algorithm = Algorithms(convergence_threshold=CONVERGENCE_THRESHOLD)

# Set up the algorithm experiment for each run
algorithm.setup_algorithm_experiment(
    number_of_nodes=N,
    number_of_users=U,
    node_radius=NODE_RADIUS,
    key=key,
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

for run in tqdm(range(1, NUM_RUNS + 1)):
    system_logger.info(f"Starting run {run} of {NUM_RUNS}")
    
    algorithm.reset()
    
    # -------------------- Random Walk Algorithm --------------------
    start_time = time.time()
    
    # Run the Random Walk Algorithm
    success_random_walk = algorithm.run_random_walk_algorithm(
        solving_method="scipy",
        max_iter=MAX_ITER / 2,
        b=B,
        c=C,
        logger=random_walk_logger
    )
    
    random_walk_logger.info("The UAV energy level is: %s at the end of the algorithm", algorithm.get_uav().get_energy_level())
    
    if success_random_walk:
        random_walk_logger.info("Random Walk Algorithm has successfully reached the final node!")
    else:
        random_walk_logger.info("Random Walk Algorithm failed to reach the final node!")
        
    random_walk_logger.info("The UAV processed in total: %s bits", algorithm.get_uav().get_total_processed_data())
    
    # Accumulate metrics
    algorithms_total_bits_acc["Random Walk Total Bits"].append(algorithm.get_uav().get_total_processed_data())
    algorithms_expended_energy_acc["Random Walk Energy Level"].append(
        algorithm.get_uav().get_total_energy_level() - algorithm.get_uav().get_energy_level()
    )
    algorithms_total_visited_nodes_acc["Random Walk Total Visited Nodes"].append(len(algorithm.get_trajectory()))
    
    # End the timer for Random Walk Algorithm
    random_walk_time = time.time() - start_time
    random_walk_logger.info("Random Walk Algorithm took: %s seconds", random_walk_time)
    timer_dict_acc["Random Walk Time"].append(random_walk_time)
    
    # Reset the Algorithm object
    algorithm.reset()
    
    # -------------------- Proportional Fairness Algorithm --------------------
    start_time = time.time()
    
    # Run the Proportional Fairness Algorithm
    success_proportional_fairness = algorithm.run_random_proportional_fairness_algorithm(
        solving_method="scipy",
        max_iter=MAX_ITER,
        b=B,
        c=C,
        logger=proportional_fairness_logger
    )
    
    proportional_fairness_logger.info("The UAV energy level is: %s at the end of the algorithm", algorithm.get_uav().get_energy_level())
    
    if success_proportional_fairness:
        proportional_fairness_logger.info("Proportional Fairness Algorithm has successfully reached the final node!")
    else:
        proportional_fairness_logger.info("Proportional Fairness Algorithm failed to reach the final node!")
        
    proportional_fairness_logger.info("The UAV processed in total: %s bits", algorithm.get_uav().get_total_processed_data())
    
    # Accumulate metrics
    algorithms_total_bits_acc["Proportional Fairness Total Bits"].append(algorithm.get_uav().get_total_processed_data())
    algorithms_expended_energy_acc["Proportional Fairness Energy Level"].append(
        algorithm.get_uav().get_total_energy_level() - algorithm.get_uav().get_energy_level()
    )
    algorithms_total_visited_nodes_acc["Proportional Fairness Total Visited Nodes"].append(len(algorithm.get_trajectory()))
    
    # End the timer for Proportional Fairness Algorithm
    proportional_fairness_time = time.time() - start_time
    proportional_fairness_logger.info("Proportional Fairness Algorithm took: %s seconds", proportional_fairness_time)
    timer_dict_acc["Proportional Fairness Time"].append(proportional_fairness_time)
    
    # Reset the Algorithm object
    algorithm.reset()
    
    # -------------------- Brave Greedy Algorithm --------------------
    start_time = time.time()
    
    # Run the Brave Greedy Algorithm
    success_brave_greedy = algorithm.brave_greedy(
        solving_method="scipy",
        max_iter=MAX_ITER,
        b=B,
        c=C,
        logger=brave_greedy_logger
    )
    
    brave_greedy_logger.info("The UAV energy level is: %s at the end of the algorithm", algorithm.get_uav().get_energy_level())
    
    if success_brave_greedy:
        brave_greedy_logger.info("Brave Greedy Algorithm has successfully reached the final node!")
    else:
        brave_greedy_logger.info("Brave Greedy Algorithm failed to reach the final node!")
        
    brave_greedy_logger.info("The UAV processed in total: %s bits", algorithm.get_uav().get_total_processed_data())
    
    # Accumulate metrics
    algorithms_total_bits_acc["Brave Greedy Total Bits"].append(algorithm.get_uav().get_total_processed_data())
    algorithms_expended_energy_acc["Brave Greedy Energy Level"].append(
        algorithm.get_uav().get_total_energy_level() - algorithm.get_uav().get_energy_level()
    )
    algorithms_total_visited_nodes_acc["Brave Greedy Total Visited Nodes"].append(len(algorithm.get_trajectory()))
        
    # End the timer for Brave Greedy Algorithm
    brave_greedy_time = time.time() - start_time
    brave_greedy_logger.info("Brave Greedy Algorithm took: %s seconds", brave_greedy_time)
    timer_dict_acc["Brave Greedy Time"].append(brave_greedy_time)
     
    # Reset the Algorithm object
    algorithm.reset()
    
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
    
    # Accumulate metrics
    algorithms_total_bits_acc["Q-Brave Total Bits"].append(algorithm.get_most_processed_bits())
    algorithms_expended_energy_acc["Q-Brave Energy Level"].append(algorithm.get_most_expended_energy())
    algorithms_total_visited_nodes_acc["Q-Brave Total Visited Nodes"].append(algorithm.get_most_visited_nodes())
    
    # End the timer for Q-Brave Algorithm
    q_brave_time = time.time() - start_time
    q_brave_logger.info("Q-Brave Algorithm took: %s seconds", q_brave_time)
    timer_dict_acc["Q-Brave Time"].append(q_brave_time)
    
    if success_q_brave_:
        q_brave_logger.info("Q-Brave Algorithm has successfully reached the final node!")
    else:
        q_brave_logger.info("Q-Brave Algorithm failed to reach the final node!")
    
    # Reset the Algorithm object for the next run
    algorithm.reset()

# -------------------- Compute Averages --------------------

algorithms_total_bits_avg = compute_average(algorithms_total_bits_acc)
algorithms_expended_energy_avg = compute_average(algorithms_expended_energy_acc)
algorithms_total_visited_nodes_avg = compute_average(algorithms_total_visited_nodes_acc)
timer_dict_avg = compute_average(timer_dict_acc)

# -------------------- Save the Averaged Data as Pickle Files --------------------
with open("algorithms_total_bits_avg.pkl", "wb") as file:
    pickle.dump(algorithms_total_bits_avg, file)

with open("algorithms_expended_energy_avg.pkl", "wb") as file:
    pickle.dump(algorithms_expended_energy_avg, file)
     
with open("algorithms_total_visited_nodes_avg.pkl", "wb") as file:
    pickle.dump(algorithms_total_visited_nodes_avg, file)
     
with open("timer_dict_avg.pkl", "wb") as file:
    pickle.dump(timer_dict_avg, file)
    
system_logger.info("Averaged data dictionaries have been saved as pickle files!")

# -------------------- Plot the Graphs --------------------
plot_graphs(folder_for_pure_learning= 'plots/multiple_runs_{}'.format(NUM_RUNS))