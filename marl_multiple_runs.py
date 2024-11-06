import logging
from jax import random
from Algorithms import Algorithms
import pickle
import time
from plot_graphs_multi_agent_vs import plot_graphs_multi_agent_vs
from Utility_functions import setup_logger, compute_average
from tqdm import tqdm
import os

# ------------------------------ Configuration Parameters ------------------------------ #

# Number of runs
TOTAL_RUNS = 500
CHECKPOINT_INTERVAL = 100  # Compute average every 100 runs

# Experiment Parameters
N = 10
NODE_RADIUS = 2
MIN_DISTANCE_BETWEEN_NODES = 40  # Minimum distance to maintain between nodes
UAV_HEIGHT = 30
CONVERGENCE_THRESHOLD = 1e-15
UAV_ENERGY_CAPACITY = 1065600
UAV_BANDWIDTH = 5 * (10**6)
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
MAX_ITER =20
NUMBER_OF_EPISODES = 15
NUMBER_OF_UAVS = 3

# Algorithms
algorithm_names = [
    "Q-Learning Common Table",
    "Q-Learning Individual Tables",
    "Q-Learning Individual Tables Double Episodes",
]

# ------------------------------ Initialization ------------------------------ #

# Initialize the logger
multi_q_learning_logger = setup_logger('multi_q_learning', 'multi_q_learning.log')

# Initialize accumulation dictionaries with empty lists for each algorithm
algorithms_total_bits_acc = {f"{name} Total Bits": [] for name in algorithm_names}
algorithms_expended_energy_acc = {f"{name} Energy Level": [] for name in algorithm_names}
algorithms_total_visited_nodes_acc = {f"{name} Total Visited Nodes": [] for name in algorithm_names}
timer_dict_acc = {f"{name} Time": [] for name in algorithm_names}

# Initialize the main random key
main_key = random.PRNGKey(10)

# ------------------------------ Helper Functions ------------------------------ #

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# ------------------------------ Experiment Loop ------------------------------ #

# Initialize progress bar
for run in tqdm(range(1, TOTAL_RUNS + 1), desc="Running Experiments"):

    # Generate a unique subkey for this run
    main_key, subkey = random.split(main_key)

    # Generate a unique number of users for each node using the subkey
    U = [random.randint(main_key, (N,), 15, 15)]  # Adjust min and max values as needed

    # Initialize the algorithm
    algorithm = Algorithms(convergence_threshold=CONVERGENCE_THRESHOLD)

    # Setup the algorithm with the current run's parameters
    algorithm.setup_multiagent_scenario(
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
        energy_level=ENERGY_LEVEL,
        number_of_uavs=NUMBER_OF_UAVS
    )

    # -------------------- Q-Learning Common Table --------------------
    start_time = time.time()

    success_common = algorithm.multi_agent_q_learning_coop(
        solving_method="scipy",
        number_of_episodes=NUMBER_OF_EPISODES,
        max_travels_per_episode=MAX_ITER,
        b=B,
        c=C,
        logger=multi_q_learning_logger,
    )

    processed_bits = algorithm.get_most_processed_bits()
    energy_expended = algorithm.get_most_expended_energy()
    total_visited_nodes = algorithm.get_most_visited_nodes()
    trajectory = algorithm.get_best_trajectories()

    q_common_time = time.time() - start_time
    multi_q_learning_logger.info("Run %d - Q-Learning Common Table took: %s seconds", run, q_common_time)

    # Accumulate results
    algorithms_total_bits_acc["Q-Learning Common Table Total Bits"].append(processed_bits)
    algorithms_expended_energy_acc["Q-Learning Common Table Energy Level"].append(energy_expended)
    algorithms_total_visited_nodes_acc["Q-Learning Common Table Total Visited Nodes"].append(total_visited_nodes)
    timer_dict_acc["Q-Learning Common Table Time"].append(q_common_time)

    # Reset the Algorithm object for the next run
    algorithm.reset()

    # -------------------- Q-Learning Individual Tables --------------------
    start_time = time.time()

    success_individual = algorithm.multi_agent_q_learning_indi(
        solving_method="scipy",
        number_of_episodes=NUMBER_OF_EPISODES,
        max_travels_per_episode=MAX_ITER,
        b=B,
        c=C,
        logger=multi_q_learning_logger,
    )

    processed_bits = algorithm.get_most_processed_bits()
    energy_expended = algorithm.get_most_expended_energy()
    total_visited_nodes = algorithm.get_most_visited_nodes()
    trajectory = algorithm.get_best_trajectory()

    q_individual_time = time.time() - start_time
    multi_q_learning_logger.info("Run %d - Q-Learning Individual Tables took: %s seconds", run, q_individual_time)

    # Accumulate results
    algorithms_total_bits_acc["Q-Learning Individual Tables Total Bits"].append(processed_bits)
    algorithms_expended_energy_acc["Q-Learning Individual Tables Energy Level"].append(energy_expended)
    algorithms_total_visited_nodes_acc["Q-Learning Individual Tables Total Visited Nodes"].append(total_visited_nodes)
    timer_dict_acc["Q-Learning Individual Tables Time"].append(q_individual_time)

    # Reset the Algorithm object for the next run
    algorithm.reset()

    # -------------------- Q-Learning Individual Tables Double Episodes --------------------
    start_time = time.time()

    success_individual_double = algorithm.multi_agent_q_learning_indi(
        solving_method="scipy",
        number_of_episodes=NUMBER_OF_EPISODES * 2,
        max_travels_per_episode=MAX_ITER,
        b=B,
        c=C,
        logger=multi_q_learning_logger,
    )

    processed_bits = algorithm.get_most_processed_bits()
    energy_expended = algorithm.get_most_expended_energy()
    total_visited_nodes = algorithm.get_most_visited_nodes()
    trajectory = algorithm.get_best_trajectory()

    q_individual_double_time = time.time() - start_time
    multi_q_learning_logger.info("Run %d - Q-Learning Individual Tables Double Episodes took: %s seconds", run, q_individual_double_time)

    # Accumulate results
    algorithms_total_bits_acc["Q-Learning Individual Tables Double Episodes Total Bits"].append(processed_bits)
    algorithms_expended_energy_acc["Q-Learning Individual Tables Double Episodes Energy Level"].append(energy_expended)
    algorithms_total_visited_nodes_acc["Q-Learning Individual Tables Double Episodes Total Visited Nodes"].append(total_visited_nodes)
    timer_dict_acc["Q-Learning Individual Tables Double Episodes Time"].append(q_individual_double_time)

    # Reset the Algorithm object for the next run
    algorithm.reset()

    # -------------------- Checkpointing --------------------
    if run % CHECKPOINT_INTERVAL == 0:
        # Compute averages up to the current run
        avg_total_bits = compute_average(algorithms_total_bits_acc)
        avg_expended_energy = compute_average(algorithms_expended_energy_acc)
        avg_total_visited_nodes = compute_average(algorithms_total_visited_nodes_acc)
        avg_timers = compute_average(timer_dict_acc)

        # Define checkpoint directory
        checkpoint_dir = f'checkpoints/run_{run}/'
        create_directory(checkpoint_dir)

        # Save the averaged data to pickle files
        save_pickle(avg_total_bits, 'multi_q_learning_total_bits.pkl')
        save_pickle(avg_expended_energy, 'multi_q_learning_expended_energy.pkl')
        save_pickle(avg_total_visited_nodes, 'multi_q_learning_total_visited_nodes.pkl')
        save_pickle(avg_timers, 'multi_q_learning_timers.pkl')

        # Log checkpoint
        multi_q_learning_logger.info("Checkpoint at run %d saved.", run)

        # Plot the graphs using the averaged data
        plot_graphs_multi_agent_vs(folder_path=checkpoint_dir)

# ------------------------------ Final Averaging and Saving ------------------------------ #

# After all runs are completed, compute final averages
final_avg_total_bits = compute_average(algorithms_total_bits_acc)
final_avg_expended_energy = compute_average(algorithms_expended_energy_acc)
final_avg_total_visited_nodes = compute_average(algorithms_total_visited_nodes_acc)
final_avg_timers = compute_average(timer_dict_acc)

# Define final output directory
final_output_dir = f'final_results_{TOTAL_RUNS}/'
create_directory(final_output_dir)

# Save the final averaged data to pickle files
save_pickle(final_avg_total_bits,'multi_q_learning_total_bits.pkl')
save_pickle(final_avg_expended_energy, 'multi_q_learning_expended_energy.pkl')
save_pickle(final_avg_total_visited_nodes, 'multi_q_learning_total_visited_nodes.pkl')
save_pickle(final_avg_timers, 'multi_q_learning_timers.pkl')

# Log final results
multi_q_learning_logger.info("Final averages after %d runs saved.", TOTAL_RUNS)

# Plot the final graphs
plot_graphs_multi_agent_vs(folder_path=final_output_dir)
