import os
import pickle
import matplotlib.pyplot as plt

def plot_graphs_multi_agent_vs(folder_for_comparison='plots/multi_agent_comparison'):
    """
    Plot comparison graphs for multiple Q-learning algorithms in a refined style.
    
    Parameters:
    - folder_for_comparison (str): Directory where the plots will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(folder_for_comparison, exist_ok=True)
    
    # Define the folder to load data from
    folder_to_load = "final_results_500_5_ep"
    
    # Load data from pickle files
    with open(os.path.join(folder_to_load, 'multi_q_learning_total_bits.pkl'), 'rb') as f:
        algorithms_total_bits_acc = pickle.load(f)
    
    with open(os.path.join(folder_to_load, 'multi_q_learning_expended_energy.pkl'), 'rb') as f:
        algorithms_expended_energy_acc = pickle.load(f)
    
    with open(os.path.join(folder_to_load, 'multi_q_learning_total_visited_nodes.pkl'), 'rb') as f:
        algorithms_total_visited_nodes_acc = pickle.load(f)
    
    with open(os.path.join(folder_to_load, 'multi_q_learning_timers.pkl'), 'rb') as f:
        timer_dict_acc = pickle.load(f)
    
    # Print the loaded data for verification
    print("Total Bits Processed by Each Algorithm")
    print(algorithms_total_bits_acc)
    
    print("\nExpended Energy by Each Algorithm")
    print(algorithms_expended_energy_acc)
    
    print("\nTotal Visited Nodes by Each Algorithm")
    print(algorithms_total_visited_nodes_acc)
    
    print("\nTime Taken by Each Algorithm")
    print(timer_dict_acc)
    
    # Define algorithm display names
    algo_names = ['BRAVE-MARL', 'IL', 'DET', 'TET']
    
    # Define colors for each algorithm
    colors = ['blue', 'green', 'red', 'grey']
    
    ##################### TOTAL BITS PROCESSED PLOT #####################
    # Extract and process data
    bits_keys = list(algorithms_total_bits_acc.keys())
    bits_processed = [float(algorithms_total_bits_acc[key]) for key in bits_keys]
    
    # Scale bits to MBits for better visualization
    bits_processed_mbits = [bits / 1e6 for bits in bits_processed]
    
    # Create the figure
    plt.figure(figsize=(24, 16))
    
    # Create a bar plot
    bars = plt.bar(algo_names, bits_processed_mbits, color=colors)
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Customize y-ticks
    plt.yticks(fontsize=52, fontweight='bold')
    
    # Remove x-ticks for a clean look
    plt.xticks([])
    
    # Define ylim
    plt.ylim(50, 58)
    
    # Set y-axis label
    plt.ylabel(r"UAV's Collected" '\n' r"Data $\boldsymbol{B}_\boldsymbol{P}$[MBits]", fontsize=58, fontweight='bold')
    
    # Add annotation "(a)" at the top-right corner
    plt.text(0.95, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=68, 
             fontweight='bold', ha='right', va='top')
    
    # Add legend
    plt.legend(bars, algo_names, fontsize=55, loc='upper center')
    
    # Save the figure
    plt.savefig(os.path.join(folder_for_comparison, "total_bits.png"))
    plt.close()
    
    ##################### EXPENDED ENERGY PLOT #####################
    # Extract and process data
    energy_keys = list(algorithms_expended_energy_acc.keys())
    energy_expended = [float(algorithms_expended_energy_acc[key]) for key in energy_keys]
    
    # Scale energy to kW for better visualization (assuming original is in W)
    energy_expended_kw = [e / 1e6 for e in energy_expended]
    
    # Create the figure
    plt.figure(figsize=(24, 16))
    
    # Create a bar plot
    bars = plt.bar(algo_names, energy_expended_kw, color=colors)
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Customize y-ticks
    plt.yticks(fontsize=40, fontweight='bold')
    
    # Remove x-ticks for a clean look
    plt.xticks([])
    
    # Set y-axis label
    plt.ylabel(r"UAV's Consumed" '\n' r"Energy $\boldsymbol{E}_\boldsymbol{P}$ [MJoules]", fontsize=50, fontweight='bold')
    
    # Set y-axis limits
    plt.ylim(2.650, 2.700)
    
    # Add annotation "(b)" at the top-right corner
    plt.text(0.95, 0.95, '(a)', transform=plt.gca().transAxes, fontsize=68, 
             fontweight='bold', ha='right', va='top')
    
    # Add legend
    plt.legend(bars, algo_names, fontsize=55, loc='upper center')
    
    # Save the figure
    plt.savefig(os.path.join(folder_for_comparison, "expended_energy.png"))
    plt.close()
    
    ##################### TOTAL VISITED NODES PLOT #####################
    # Extract and process data
    nodes_keys = list(algorithms_total_visited_nodes_acc.keys())
    total_visited_nodes = [float(algorithms_total_visited_nodes_acc[key]) for key in nodes_keys]
    
    # Create the figure
    plt.figure(figsize=(24, 16))

    # Create a bar plot
    bars = plt.bar(algo_names, total_visited_nodes, color=colors)
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Customize y-ticks
    plt.yticks(fontsize=50, fontweight='bold')
    
    # Remove x-ticks for a clean look
    plt.xticks([])
    
    # Set y-lim
    plt.ylim(60,80)
    
    # Set y-axis label
    plt.ylabel(r"Number of Visited AoIs" + r" $|\boldsymbol{P}|$", fontsize=58, fontweight='bold')
    
    # Add annotation "(c)" at the top-right corner
    plt.text(0.95, 0.95, '(c)', transform=plt.gca().transAxes, fontsize=68, 
             fontweight='bold', ha='right', va='top')
    
    # Add legend
    plt.legend(bars, algo_names, fontsize=50, loc='upper left')
    
    # Save the figure
    plt.savefig(os.path.join(folder_for_comparison, "total_visited_nodes.png"))
    plt.close()

    values = [(total_bits_processed * total_nodes_visited) / energy_expended for total_bits_processed, total_nodes_visited, energy_expended in zip(bits_processed, total_visited_nodes, energy_expended)]

    # Define colors
    colors = ['blue', 'green', 'red', 'grey']

    # Create the figure
    plt.figure(figsize=(24, 16))

    # Create a bar plot
    bars = plt.bar(algo_names, values, color=colors)

    # Add a grid behind the bars
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Set the y-tick parameters
    plt.yticks(fontsize=40, fontweight='bold')

    # Remove the x-axis ticks and labels
    plt.xticks([])

    # Add x-grid
    plt.grid(axis='x', linestyle='--', alpha=1)

    # Add y-grid
    plt.grid(axis='y', linestyle='--', alpha=1)

    # Set the y-axis label
    plt.ylabel(r'Efficiency $\frac{\boldsymbol{B}_\boldsymbol{P} \cdot |\boldsymbol{P}|}{\boldsymbol{E}_\boldsymbol{P}}$', 
               fontsize=55, fontweight='bold')
    
        # Add annotation "(c)" at the top-right corner
    plt.text(0.95, 0.95, '(d)', transform=plt.gca().transAxes, fontsize=68, 
             fontweight='bold', ha='right', va='top')

    # Add a legend
    plt.legend(bars, algo_names, fontsize=60, loc='upper center')
    
    # Set y-limit
    plt.ylim(1350,1450)

    # Save the figure
    plt.savefig(os.path.join(folder_for_comparison, "custom_metric.png"))
    plt.close()
    
    ##################### TIME TAKEN PLOT #####################
    # Extract and process data
    time_keys = list(timer_dict_acc.keys())
    time_taken = [float(timer_dict_acc[key]) for key in time_keys]
    
    # Divide by 60 to convert seconds to minutes
    time_taken = [t / 60 for t in time_taken]
    
    # Create the figure
    plt.figure(figsize=(24, 16))
    
    # Create a bar plot
    bars = plt.bar(algo_names, time_taken, color=colors)
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Customize y-ticks
    plt.yticks(fontsize=50, fontweight='bold')
    
    # Remove x-ticks for a clean look
    plt.xticks([])
    
    # Set y-axis label
    plt.ylabel('Execution Time [min]', fontsize=58, fontweight='bold')
    

    plt.ylim(25, 80)
    
    # Add annotation "(e)" at the top-right corner
    plt.text(0.95, 0.95, '(e)', transform=plt.gca().transAxes, fontsize=68, 
             fontweight='bold', ha='right', va='top')
    
    # Add legend without frame
    plt.legend(bars, algo_names, fontsize=60, loc='upper left', frameon=True)
    
    # Save the figure
    plt.savefig(os.path.join(folder_for_comparison, 'time_taken.png'))
    plt.close()

if __name__ == "__main__":
    plot_graphs_multi_agent_vs(folder_for_comparison='multi_q_learning_results_5_ep')
