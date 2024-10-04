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
MAX_ITER = 10

data_dict = {}

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
data_dict["Random Walk Total Bits"] = algorithm.get_uav().get_total_processed_data()


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
data_dict["Proportional Fairness Total Bits"] = algorithm.get_uav().get_total_processed_data()

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
data_dict["Brave Greedy Total Bits"] = algorithm.get_uav().get_total_processed_data()
	
# Reset the Algorithm object
algorithm.reset()

# Run the Q-Brave Algorithm
success_q_brave_ = algorithm.q_brave(solving_method= "scipy", number_of_episodes= 5, max_travels_per_episode= MAX_ITER)

logging.info("The UAV energy level is: %s", algorithm.get_uav().get_energy_level())
logging.info("The UAV processed in total: %s bits", algorithm.most_processed_bits)
data_dict["Q-Brave Total Bits"] = algorithm.most_processed_bits

if success_q_brave_:
	logging.info("Q-Brave Algorithm has successfully reached the final node!")
else:
	logging.info("Q-Brave Algorithm failed to reach the final node!")


# Save the data dictionary as a pickle file
with open("data_dict.pkl", "wb") as file:
	pickle.dump(data_dict, file)
	
logging.info("Data dictionary has been saved as a pickle file!")

# Sort the Nodes based on total bits and log the data
sorted_nodes = algorithm.sort_nodes_based_on_total_bits()
logging.info("The sorted nodes based on total bits are: %s", sorted_nodes)