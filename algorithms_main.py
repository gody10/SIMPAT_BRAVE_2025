import logging
from jax import random
from Algorithms import Algorithms
import pickle
import time
from plot_graphs import plot_graphs
from Utility_functions import setup_logger

# Initialize timers for each method
random_walk_time = 0
proportional_fairness_time = 0
brave_greedy_time = 0
q_brave_time = 0

# Create a random key
key = random.PRNGKey(10)

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
DISTANCE_MAX = 40
MAX_BITS = 2 * 10**6
MIN_BITS = 3 * 10**5
ENERGY_LEVEL = 29000
B = 0.74
C = 0.00043
MAX_ITER = 30
NUMBER_OF_EPISODES = 100

algorithms_total_bits = {}
algorithms_expended_energy = {}
algorithms_total_visited_nodes = {}
timer_dict = {}

system_logger = setup_logger('system_logger', 'system.log')

# Create the algorithm object
algorithm = Algorithms(convergence_threshold= CONVERGENCE_THRESHOLD)

algorithm.setup_algorithm_experiment(number_of_nodes= N, number_of_users= U, node_radius= NODE_RADIUS, key= key, min_distance_between_nodes= MIN_DISTANCE_BETWEEN_NODES, uav_height= UAV_HEIGHT, 
						   uav_energy_capacity=UAV_ENERGY_CAPACITY, uav_bandwidth= UAV_BANDWIDTH, uav_processing_capacity= UAV_PROCESSING_CAPACITY, uav_cpu_frequency= UAV_CPU_FREQUENCY, uav_velocity= UAV_VELOCITY, 
							min_bits= MIN_BITS, max_bits= MAX_BITS, distance_min= DISTANCE_MIN, distance_max= DISTANCE_MAX, energy_level= ENERGY_LEVEL)

# Set up separate loggers for each algorithm
random_walk_logger = setup_logger('random_walk_logger', 'random_walk_algorithm.log')
proportional_fairness_logger = setup_logger('proportional_fairness_logger', 'proportional_algorithm.log')
brave_greedy_logger = setup_logger('brave_greedy_logger', 'brave_algorithm.log')
q_brave_logger = setup_logger('q_brave_logger', 'q_brave_algorithm.log')

# Start the timer for Random Walk Algorithm
start_time = time.time()

# Run the Random Walk Algorithm
success_random_walk = algorithm.run_random_walk_algorithm(solving_method= "scipy", max_iter= MAX_ITER, b= B, c= C, logger= random_walk_logger)

random_walk_logger.info("The UAV energy level is: %s at the end of the algorithm", algorithm.get_uav().get_energy_level())

if success_random_walk:
	random_walk_logger.info("Random Walk Algorithm has successfully reached the final node!")
else:
	random_walk_logger.info("Random Walk Algorithm failed to reach the final node!")
	
random_walk_logger.info("The UAV processed in total: %s bits", algorithm.get_uav().get_total_processed_data())
algorithms_total_bits["Random Walk Total Bits"] = algorithm.get_uav().get_total_processed_data()
algorithms_expended_energy["Random Walk Energy Level"] = algorithm.get_uav().get_total_energy_level() - algorithm.get_uav().get_energy_level()
algorithms_total_visited_nodes["Random Walk Total Visited Nodes"] = len(algorithm.get_trajectory())

# End the timer for Random Walk Algorithm
random_walk_time = time.time() - start_time
random_walk_logger.info("Random Walk Algorithm took: %s seconds", random_walk_time)
timer_dict["Random Walk Time"] = random_walk_time

# Reset the Algorithm object
algorithm.reset()

# Begin the timer for Proportional Fairness Algorithm
start_time = time.time()

# Run the Proportional Fairness Algorithm
success_proportional_fairness = algorithm.run_random_proportional_fairness_algorithm(solving_method= "scipy", max_iter= MAX_ITER, b= B, c= C, logger= proportional_fairness_logger)

proportional_fairness_logger.info("The UAV energy level is: %s at the end of the algorithm", algorithm.get_uav().get_energy_level())

if success_proportional_fairness:
	proportional_fairness_logger.info("Proportional Fairness Algorithm has successfully reached the final node!")
else:
	proportional_fairness_logger.info("Proportional Fairness Algorithm failed to reach the final node!")
	
proportional_fairness_logger.info("The UAV processed in total: %s bits", algorithm.get_uav().get_total_processed_data())
algorithms_total_bits["Proportional Fairness Total Bits"] = algorithm.get_uav().get_total_processed_data()
algorithms_expended_energy["Proportional Fairness Energy Level"] = algorithm.get_uav().get_total_energy_level() - algorithm.get_uav().get_energy_level()
algorithms_total_visited_nodes["Proportional Fairness Total Visited Nodes"] = len(algorithm.get_trajectory())

# End the timer for Proportional Fairness Algorithm
proportional_fairness_time = time.time() - start_time
proportional_fairness_logger.info("Proportional Fairness Algorithm took: %s seconds", proportional_fairness_time)
timer_dict["Proportional Fairness Time"] = proportional_fairness_time

# Reset the Algorithm object
algorithm.reset()

# Start the timer for Brave Greedy Algorithm
start_time = time.time()

# Run the Brave Greedy Algorithm
success_brave_greedy = algorithm.brave_greedy(solving_method= "scipy", max_iter= MAX_ITER, b= B, c= C, logger= brave_greedy_logger)

brave_greedy_logger.info("The UAV energy level is: %s at the end of the algorithm", algorithm.get_uav().get_energy_level())

if success_brave_greedy:
	brave_greedy_logger.info("Brave Greedy Algorithm has successfully reached the final node!")
else:
	brave_greedy_logger.info("Brave Greedy Algorithm failed to reach the final node!")
	
brave_greedy_logger.info("The UAV processed in total: %s bits", algorithm.get_uav().get_total_processed_data())
algorithms_total_bits["Brave Greedy Total Bits"] = algorithm.get_uav().get_total_processed_data()
algorithms_expended_energy["Brave Greedy Energy Level"] = algorithm.get_uav().get_total_energy_level() - algorithm.get_uav().get_energy_level()
algorithms_total_visited_nodes["Brave Greedy Total Visited Nodes"] = len(algorithm.get_trajectory())
	
# End the timer for Brave Greedy Algorithm
brave_greedy_time = time.time() - start_time
brave_greedy_logger.info("Brave Greedy Algorithm took: %s seconds", brave_greedy_time)
timer_dict["Brave Greedy Time"] = brave_greedy_time
 
# Reset the Algorithm object
algorithm.reset()

# Start the timer for Q-Brave Algorithm
start_time = time.time()

# Run the Q-Brave Algorithm
success_q_brave_ = algorithm.q_brave(solving_method= "scipy", number_of_episodes= NUMBER_OF_EPISODES, max_travels_per_episode= MAX_ITER, b= B, c= C, logger= q_brave_logger)

q_brave_logger.info("The UAV energy level is: %s", algorithm.get_uav().get_energy_level())
q_brave_logger.info("The UAV processed in total: %s bits", algorithm.most_processed_bits)
algorithms_total_bits["Q-Brave Total Bits"] = algorithm.get_most_processed_bits()
algorithms_expended_energy["Q-Brave Energy Level"] = algorithm.get_most_expended_energy()
algorithms_total_visited_nodes["Q-Brave Total Visited Nodes"] = algorithm.get_most_visited_nodes()

# End the timer for Q-Brave Algorithm
q_brave_time = time.time() - start_time
q_brave_logger.info("Q-Brave Algorithm took: %s seconds", q_brave_time)
timer_dict["Q-Brave Time"] = q_brave_time

if success_q_brave_:
	q_brave_logger.info("Q-Brave Algorithm has successfully reached the final node!")
else:
	q_brave_logger.info("Q-Brave Algorithm failed to reach the final node!")


# Save the data dictionaries as a pickle files
with open("algorithms_total_bits.pkl", "wb") as file:
	pickle.dump(algorithms_total_bits, file)

with open("algorithms_expended_energy.pkl", "wb") as file:
	pickle.dump(algorithms_expended_energy, file)
 
with open("algorithms_total_visited_nodes.pkl", "wb") as file:
	pickle.dump(algorithms_total_visited_nodes, file)
 
with open("timer_dict.pkl", "wb") as file:
	pickle.dump(timer_dict, file)
	
system_logger.info("Data dictionaries has been saved as a pickle file!")

# Plot the graphs
plot_graphs()