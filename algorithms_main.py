import logging
from jax import random
from Algorithms import Algorithms
import pickle

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
UAV_ENERGY_CAPACITY = 198000
UAV_BANDWIDTH = 15
UAV_PROCESSING_CAPACITY = 1000
UAV_CPU_FREQUENCY = 2
UAV_VELOCITY = 1
MAX_ITER = 100
NUMBER_OF_EPISODES = 100

algorithms_total_bits = {}
algorithms_expended_energy = {}
algorithms_total_visited_nodes = {}

# Create the algorithm object
algorithm = Algorithms(convergence_threshold= CONVERGENCE_THRESHOLD)

algorithm.setup_experiment(number_of_nodes= N, number_of_users= U, node_radius= NODE_RADIUS, key= key, min_distance_between_nodes= MIN_DISTANCE_BETWEEN_NODES, uav_height= UAV_HEIGHT, 
						   uav_energy_capacity=UAV_ENERGY_CAPACITY, uav_bandwidth= UAV_BANDWIDTH, uav_processing_capacity= UAV_PROCESSING_CAPACITY, uav_cpu_frequency= UAV_CPU_FREQUENCY, uav_velocity= UAV_VELOCITY)

# Run the Random Walk Algorithm
success_random_walk = algorithm.run_random_walk_algorithm(solving_method= "scipy", max_iter= MAX_ITER)

logging.info("The UAV energy level is: %s at the end of the algorithm", algorithm.get_uav().get_energy_level())

if success_random_walk:
	logging.info("Random Walk Algorithm has successfully reached the final node!")
else:
	logging.info("Random Walk Algorithm failed to reach the final node!")
	
logging.info("The UAV processed in total: %s bits", algorithm.get_uav().get_total_processed_data())
algorithms_total_bits["Random Walk Total Bits"] = algorithm.get_uav().get_total_processed_data()
algorithms_expended_energy["Random Walk Energy Level"] = algorithm.get_uav().get_total_energy_level() - algorithm.get_uav().get_energy_level()
algorithms_total_visited_nodes["Random Walk Total Visited Nodes"] = len(algorithm.get_trajectory())

# Reset the Algorithm object
algorithm.reset()

# Run the Proportional Fairness Algorithm
success_random_walk = algorithm.run_random_proportional_fairness_algorithm(solving_method= "scipy", max_iter= MAX_ITER)

logging.info("The UAV energy level is: %s at the end of the algorithm", algorithm.get_uav().get_energy_level())

if success_random_walk:
	logging.info("Random Walk Algorithm has successfully reached the final node!")
else:
	logging.info("Random Walk Algorithm failed to reach the final node!")
	
logging.info("The UAV processed in total: %s bits", algorithm.get_uav().get_total_processed_data())
algorithms_total_bits["Proportional Fairness Total Bits"] = algorithm.get_uav().get_total_processed_data()
algorithms_expended_energy["Proportional Fairness Energy Level"] = algorithm.get_uav().get_total_energy_level() - algorithm.get_uav().get_energy_level()
algorithms_total_visited_nodes["Proportional Fairness Total Visited Nodes"] = len(algorithm.get_trajectory())

# Reset the Algorithm object
algorithm.reset()

# Run the Brave Greedy Algorithm
success_brave_greedy = algorithm.brave_greedy(solving_method= "scipy", max_iter= MAX_ITER)

logging.info("The UAV energy level is: %s at the end of the algorithm", algorithm.get_uav().get_energy_level())

if success_brave_greedy:
	logging.info("Brave Greedy Algorithm has successfully reached the final node!")
else:
	logging.info("Brave Greedy Algorithm failed to reach the final node!")
	
logging.info("The UAV processed in total: %s bits", algorithm.get_uav().get_total_processed_data())
algorithms_total_bits["Brave Greedy Total Bits"] = algorithm.get_uav().get_total_processed_data()
algorithms_expended_energy["Brave Greedy Energy Level"] = algorithm.get_uav().get_total_energy_level() - algorithm.get_uav().get_energy_level()
algorithms_total_visited_nodes["Brave Greedy Total Visited Nodes"] = len(algorithm.get_trajectory())
	
# Reset the Algorithm object
algorithm.reset()

# Run the Q-Brave Algorithm
success_q_brave_ = algorithm.q_brave(solving_method= "scipy", number_of_episodes= NUMBER_OF_EPISODES, max_travels_per_episode= MAX_ITER)

logging.info("The UAV energy level is: %s", algorithm.get_uav().get_energy_level())
logging.info("The UAV processed in total: %s bits", algorithm.most_processed_bits)
algorithms_total_bits["Q-Brave Total Bits"] = algorithm.get_most_processed_bits()
algorithms_expended_energy["Q-Brave Energy Level"] = algorithm.get_most_expended_energy()
algorithms_total_visited_nodes["Q-Brave Total Visited Nodes"] = algorithm.get_most_visited_nodes()

if success_q_brave_:
	logging.info("Q-Brave Algorithm has successfully reached the final node!")
else:
	logging.info("Q-Brave Algorithm failed to reach the final node!")


# Save the data dictionary as a pickle file
with open("algorithms_total_bits.pkl", "wb") as file:
	pickle.dump(algorithms_total_bits, file)

with open("algorithms_expended_energy.pkl", "wb") as file:
	pickle.dump(algorithms_expended_energy, file)
 
with open("algorithms_total_visited_nodes.pkl", "wb") as file:
	pickle.dump(algorithms_total_visited_nodes, file)
	
logging.info("Data dictionaries has been saved as a pickle file!")

# Sort the Nodes based on total bits and log the data
sorted_nodes = algorithm.sort_nodes_based_on_total_bits()
logging.info("The sorted nodes based on total bits are: %s", sorted_nodes)