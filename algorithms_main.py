import logging
from jax import random
from Algorithms import Algorithms
import pickle
import time
from plot_graphs import plot_graphs

# Initialize timers for each method
random_walk_time = 0
proportional_fairness_time = 0
brave_greedy_time = 0
q_brave_time = 0

# Create a random key
key = random.PRNGKey(20)

# Setup logging
logging.basicConfig(
	filename='algorithm_logs.log',  # Log file name
	filemode='w',            # Mode: 'w' for overwrite, 'a' for append
	level=logging.INFO,      # Logging level
	format='%(asctime)s - %(levelname)s - %(message)s'  # Format of the log messages
)

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
UAV_BANDWIDTH = 5*(10**6)
UAV_PROCESSING_CAPACITY = 1 * (10**9)
UAV_CPU_FREQUENCY = 2 * (10**9)
UAV_VELOCITY = 1
MAX_ITER = 50
DISTANCE_MIN = 10
DISTANCE_MAX = 40
MAX_BITS = 2 * 10**6
MIN_BITS = 3 * 10**5
ENERGY_LEVEL = 29000
B = 0.74
C = 0.00043
MAX_ITER = 100
NUMBER_OF_EPISODES = 50

algorithms_total_bits = {}
algorithms_expended_energy = {}
algorithms_total_visited_nodes = {}
timer_dict = {}

# Create the algorithm object
algorithm = Algorithms(convergence_threshold= CONVERGENCE_THRESHOLD)

algorithm.setup_algorithm_experiment(number_of_nodes= N, number_of_users= U, node_radius= NODE_RADIUS, key= key, min_distance_between_nodes= MIN_DISTANCE_BETWEEN_NODES, uav_height= UAV_HEIGHT, 
						   uav_energy_capacity=UAV_ENERGY_CAPACITY, uav_bandwidth= UAV_BANDWIDTH, uav_processing_capacity= UAV_PROCESSING_CAPACITY, uav_cpu_frequency= UAV_CPU_FREQUENCY, uav_velocity= UAV_VELOCITY, 
							min_bits= MIN_BITS, max_bits= MAX_BITS, distance_min= DISTANCE_MIN, distance_max= DISTANCE_MAX, energy_level= ENERGY_LEVEL)

# Start the timer for Random Walk Algorithm
start_time = time.time()

# Run the Random Walk Algorithm
success_random_walk = algorithm.run_random_walk_algorithm(solving_method= "scipy", max_iter= MAX_ITER, b= B, c= C)

logging.info("The UAV energy level is: %s at the end of the algorithm", algorithm.get_uav().get_energy_level())

if success_random_walk:
	logging.info("Random Walk Algorithm has successfully reached the final node!")
else:
	logging.info("Random Walk Algorithm failed to reach the final node!")
	
logging.info("The UAV processed in total: %s bits", algorithm.get_uav().get_total_processed_data())
algorithms_total_bits["Random Walk Total Bits"] = algorithm.get_uav().get_total_processed_data()
algorithms_expended_energy["Random Walk Energy Level"] = algorithm.get_uav().get_total_energy_level() - algorithm.get_uav().get_energy_level()
algorithms_total_visited_nodes["Random Walk Total Visited Nodes"] = len(algorithm.get_trajectory())

# End the timer for Random Walk Algorithm
random_walk_time = time.time() - start_time
logging.info("Random Walk Algorithm took: %s seconds", random_walk_time)
timer_dict["Random Walk Time"] = random_walk_time

# Reset the Algorithm object
algorithm.reset()

# Begin the timer for Proportional Fairness Algorithm
start_time = time.time()

# Run the Proportional Fairness Algorithm
success_proportional_fairness = algorithm.run_random_proportional_fairness_algorithm(solving_method= "scipy", max_iter= MAX_ITER, b= B, c= C)

logging.info("The UAV energy level is: %s at the end of the algorithm", algorithm.get_uav().get_energy_level())

if success_proportional_fairness:
	logging.info("Proportional Fairness Algorithm has successfully reached the final node!")
else:
	logging.info("Proportional Fairness Algorithm failed to reach the final node!")
	
logging.info("The UAV processed in total: %s bits", algorithm.get_uav().get_total_processed_data())
algorithms_total_bits["Proportional Fairness Total Bits"] = algorithm.get_uav().get_total_processed_data()
algorithms_expended_energy["Proportional Fairness Energy Level"] = algorithm.get_uav().get_total_energy_level() - algorithm.get_uav().get_energy_level()
algorithms_total_visited_nodes["Proportional Fairness Total Visited Nodes"] = len(algorithm.get_trajectory())

# End the timer for Proportional Fairness Algorithm
proportional_fairness_time = time.time() - start_time
logging.info("Proportional Fairness Algorithm took: %s seconds", proportional_fairness_time)
timer_dict["Proportional Fairness Time"] = proportional_fairness_time

# Reset the Algorithm object
algorithm.reset()

# Start the timer for Brave Greedy Algorithm
start_time = time.time()

# Run the Brave Greedy Algorithm
success_brave_greedy = algorithm.brave_greedy(solving_method= "scipy", max_iter= MAX_ITER, b= B, c= C)

logging.info("The UAV energy level is: %s at the end of the algorithm", algorithm.get_uav().get_energy_level())

if success_brave_greedy:
	logging.info("Brave Greedy Algorithm has successfully reached the final node!")
else:
	logging.info("Brave Greedy Algorithm failed to reach the final node!")
	
logging.info("The UAV processed in total: %s bits", algorithm.get_uav().get_total_processed_data())
algorithms_total_bits["Brave Greedy Total Bits"] = algorithm.get_uav().get_total_processed_data()
algorithms_expended_energy["Brave Greedy Energy Level"] = algorithm.get_uav().get_total_energy_level() - algorithm.get_uav().get_energy_level()
algorithms_total_visited_nodes["Brave Greedy Total Visited Nodes"] = len(algorithm.get_trajectory())
	
# End the timer for Brave Greedy Algorithm
brave_greedy_time = time.time() - start_time
logging.info("Brave Greedy Algorithm took: %s seconds", brave_greedy_time)
timer_dict["Brave Greedy Time"] = brave_greedy_time
 
# Reset the Algorithm object
algorithm.reset()

# Start the timer for Q-Brave Algorithm
start_time = time.time()

# Run the Q-Brave Algorithm
success_q_brave_ = algorithm.q_brave(solving_method= "scipy", number_of_episodes= NUMBER_OF_EPISODES, max_travels_per_episode= MAX_ITER, b= B, c= C)

logging.info("The UAV energy level is: %s", algorithm.get_uav().get_energy_level())
logging.info("The UAV processed in total: %s bits", algorithm.most_processed_bits)
algorithms_total_bits["Q-Brave Total Bits"] = algorithm.get_most_processed_bits()
algorithms_expended_energy["Q-Brave Energy Level"] = algorithm.get_most_expended_energy()
algorithms_total_visited_nodes["Q-Brave Total Visited Nodes"] = algorithm.get_most_visited_nodes()

# End the timer for Q-Brave Algorithm
q_brave_time = time.time() - start_time
logging.info("Q-Brave Algorithm took: %s seconds", q_brave_time)
timer_dict["Q-Brave Time"] = q_brave_time

if success_q_brave_:
	logging.info("Q-Brave Algorithm has successfully reached the final node!")
else:
	logging.info("Q-Brave Algorithm failed to reach the final node!")


# Save the data dictionaries as a pickle files
with open("algorithms_total_bits.pkl", "wb") as file:
	pickle.dump(algorithms_total_bits, file)

with open("algorithms_expended_energy.pkl", "wb") as file:
	pickle.dump(algorithms_expended_energy, file)
 
with open("algorithms_total_visited_nodes.pkl", "wb") as file:
	pickle.dump(algorithms_total_visited_nodes, file)
 
with open("timer_dict.pkl", "wb") as file:
	pickle.dump(timer_dict, file)
	
logging.info("Data dictionaries has been saved as a pickle file!")

# Sort the Nodes based on total bits and log the data
sorted_nodes = algorithm.sort_nodes_based_on_total_bits()
logging.info("The sorted nodes based on total bits are: %s", sorted_nodes)

# Plot the graphs
plot_graphs()