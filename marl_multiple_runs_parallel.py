import logging
from jax import random
from Algorithms import Algorithms
import pickle
import time
from plot_graphs_multi_agent_vs import plot_graphs_multi_agent_vs
from Utility_functions import setup_logger, compute_average
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

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
MAX_ITER = 20
NUMBER_OF_EPISODES = 15
NUMBER_OF_UAVS = 3

# Algorithms
algorithm_names = [
    "Q-Learning Common Table",
    "Q-Learning Individual Tables",
    "Q-Learning Individual Tables Double Episodes",
]

# ------------------------------ Helper Functions ------------------------------ #

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# ------------------------------ Single Experiment Function ------------------------------ #

def run_single_experiment(run_number, key):
    """
    Executes a single experiment run with the given run number and random key.

    Returns:
        A dictionary containing results for each algorithm.
    """
    # Initialize the algorithm
    algorithm = Algorithms(convergence_threshold=CONVERGENCE_THRESHOLD)
    
    # Generate a unique number of users for each node using the subkey
    U = [random.randint(key, (N,), 15, 15)]  # Adjust min and max values as needed
    
    # Setup the algorithm with the current run's parameters
    algorithm.setup_multiagent_scenario(
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
        energy_level=ENERGY_LEVEL,
        number_of_uavs=NUMBER_OF_UAVS
    )
    
    results = {}
    
    # -------------------- Q-Learning Common Table --------------------
    start_time = time.time()
    
    success_common = algorithm.multi_agent_q_learning_coop(
        solving_method="scipy",
        number_of_episodes=NUMBER_OF_EPISODES,
        max_travels_per_episode=MAX_ITER,
        b=B,
        c=C,
        logger=None,  # Logging handled in main process
    )
    
    processed_bits = algorithm.get_most_processed_bits()
    energy_expended = algorithm.get_most_expended_energy()
    total_visited_nodes = algorithm.get_most_visited_nodes()
    # trajectory = algorithm.get_best_trajectories()  # Not used in accumulation
    
    q_common_time = time.time() - start_time
    
    results["Q-Learning Common Table"] = {
        "Total Bits": processed_bits,
        "Energy Level": energy_expended,
        "Total Visited Nodes": total_visited_nodes,
        "Time": q_common_time,
    }
    
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
        logger=None,
    )
    
    processed_bits = algorithm.get_most_processed_bits()
    energy_expended = algorithm.get_most_expended_energy()
    total_visited_nodes = algorithm.get_most_visited_nodes()
    # trajectory = algorithm.get_best_trajectory()
    
    q_individual_time = time.time() - start_time
    
    results["Q-Learning Individual Tables"] = {
        "Total Bits": processed_bits,
        "Energy Level": energy_expended,
        "Total Visited Nodes": total_visited_nodes,
        "Time": q_individual_time,
    }
    
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
        logger=None,
    )
    
    processed_bits = algorithm.get_most_processed_bits()
    energy_expended = algorithm.get_most_expended_energy()
    total_visited_nodes = algorithm.get_most_visited_nodes()
    # trajectory = algorithm.get_best_trajectory()
    
    q_individual_double_time = time.time() - start_time
    
    results["Q-Learning Individual Tables Double Episodes"] = {
        "Total Bits": processed_bits,
        "Energy Level": energy_expended,
        "Total Visited Nodes": total_visited_nodes,
        "Time": q_individual_double_time,
    }
    
    # Reset the Algorithm object for the next run
    algorithm.reset()
    
    return results

# ------------------------------ Main Execution ------------------------------ #

def main():
    # Initialize the logger
    multi_q_learning_logger = setup_logger('multi_q_learning', 'multi_q_learning.log')
    
    # Initialize accumulation dictionaries with empty lists for each algorithm
    algorithms_total_bits_acc = {f"{name} Total Bits": [] for name in algorithm_names}
    algorithms_expended_energy_acc = {f"{name} Energy Level": [] for name in algorithm_names}
    algorithms_total_visited_nodes_acc = {f"{name} Total Visited Nodes": [] for name in algorithm_names}
    timer_dict_acc = {f"{name} Time": [] for name in algorithm_names}
    
    # Initialize the main random key and split into individual keys
    main_key = random.PRNGKey(10)
    keys = random.split(main_key, TOTAL_RUNS)
    
    # Define a partial function to pass only run_number and key
    # This is useful if additional arguments are needed
    # run_func = partial(run_single_experiment)
    
    # Start the parallel execution
    with ProcessPoolExecutor() as executor:
        # Submit all runs
        futures = {
            executor.submit(run_single_experiment, run, keys[run]): run 
            for run in range(1, TOTAL_RUNS + 1)
        }
        
        # Initialize a counter for checkpointing
        completed_runs = 0
        
        # Use tqdm for progress bar
        for future in tqdm(as_completed(futures), total=TOTAL_RUNS, desc="Running Experiments"):
            run = futures[future]
            try:
                result = future.result()
            except Exception as e:
                multi_q_learning_logger.error("Run %d generated an exception: %s", run, e)
                continue
            
            # Accumulate results for each algorithm
            for algo_name, metrics in result.items():
                algorithms_total_bits_acc[f"{algo_name} Total Bits"].append(metrics["Total Bits"])
                algorithms_expended_energy_acc[f"{algo_name} Energy Level"].append(metrics["Energy Level"])
                algorithms_total_visited_nodes_acc[f"{algo_name} Total Visited Nodes"].append(metrics["Total Visited Nodes"])
                timer_dict_acc[f"{algo_name} Time"].append(metrics["Time"])
                
                # Log the timing information
                multi_q_learning_logger.info(
                    "Run %d - %s took: %.4f seconds",
                    run,
                    algo_name,
                    metrics["Time"]
                )
            
            completed_runs += 1
            
            # Handle checkpointing
            if completed_runs % CHECKPOINT_INTERVAL == 0:
                # Compute averages up to the current run
                avg_total_bits = compute_average(algorithms_total_bits_acc)
                avg_expended_energy = compute_average(algorithms_expended_energy_acc)
                avg_total_visited_nodes = compute_average(algorithms_total_visited_nodes_acc)
                avg_timers = compute_average(timer_dict_acc)
                
                # Define checkpoint directory
                checkpoint_dir = f'checkpoints/run_{completed_runs}/'
                create_directory(checkpoint_dir)
                
                # Save the averaged data to pickle files
                save_pickle(avg_total_bits, os.path.join(checkpoint_dir, 'multi_q_learning_total_bits.pkl'))
                save_pickle(avg_expended_energy, os.path.join(checkpoint_dir, 'multi_q_learning_expended_energy.pkl'))
                save_pickle(avg_total_visited_nodes, os.path.join(checkpoint_dir, 'multi_q_learning_total_visited_nodes.pkl'))
                save_pickle(avg_timers, os.path.join(checkpoint_dir, 'multi_q_learning_timers.pkl'))
                
                # Log checkpoint
                multi_q_learning_logger.info("Checkpoint at run %d saved.", completed_runs)
                
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
    save_pickle(final_avg_total_bits, os.path.join(final_output_dir, 'multi_q_learning_total_bits.pkl'))
    save_pickle(final_avg_expended_energy, os.path.join(final_output_dir, 'multi_q_learning_expended_energy.pkl'))
    save_pickle(final_avg_total_visited_nodes, os.path.join(final_output_dir, 'multi_q_learning_total_visited_nodes.pkl'))
    save_pickle(final_avg_timers, os.path.join(final_output_dir, 'multi_q_learning_timers.pkl'))
    
    # Log final results
    multi_q_learning_logger.info("Final averages after %d runs saved.", TOTAL_RUNS)
    
    # Plot the final graphs
    plot_graphs_multi_agent_vs(folder_path=final_output_dir)

if __name__ == "__main__":
    main()
