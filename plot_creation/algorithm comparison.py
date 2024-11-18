import pickle
import matplotlib.pyplot as plt
import os
import jax.numpy as jnp

def plot_graphs(folder_for_pure_learning= 'plots/pure_learning'):
	"""
	Plot the graphs for the algorithms
	"""
	# Specify the folder to save the plot
	basic_folder = "plots"
	folder_for_pure_learning = folder_for_pure_learning
 
	os.makedirs(folder_for_pure_learning, exist_ok=True)

	# Read the data dictionary from pickle
	with open('big_run_plots_equal/multiple_runs_500/algorithms_total_bits_avg.pkl', 'rb') as handle:
		data_dict = pickle.load(handle)

	# Extract the keys and values
	algorithms = list(data_dict.keys())

	# Remove total bits from the name of algorithms
	algorithms = [algorithm.replace("Total Bits", "") for algorithm in algorithms]
 
	# Rename the second algorithm to "BRAVE-EXPO"
	algorithms[1] = "BRAVE-EXPO"
	
	# Rename the first algorithm to "BRAVE-PRO"
	algorithms[0] = "BRAVE-PRO"
 
	# Rename the third algorith to "BRAVE-GREEDY"
	algorithms[2] = "BRAVE-GREEDY"
 
	# Rename the fourth algorithm to "Q-BRAVE"
	algorithms[3] = "Q-BRAVE"

	bits_processed = list(data_dict.values())

	# Define colors for each bar (for visualization purposes)
	colors = ['blue', 'green', 'red', 'grey']  # Customize colors as needed

	# Create the figure
	plt.figure(figsize=(12, 8))
 
	# Divide by 1e6 for better visualization
	bits_processed_temp = [bits / 1e6 for bits in bits_processed]

	# Create a bar plot with colored bars
	bars = plt.bar(algorithms, bits_processed_temp, color=colors)

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the y-tick parameters
	plt.yticks(fontsize=30, fontweight='bold')

	# Remove the x-axis ticks and labels for a clean look
	plt.xticks([])
 
	# Add x-grid
	plt.grid(axis='x', linestyle='--', alpha=1)
 
	# Add y-grid
	plt.grid(axis='y', linestyle='--', alpha=1)

	# Set the y-axis label with appropriate formatting
	plt.ylabel(r"UAV's Collected Data $\boldsymbol{B}_\boldsymbol{P}$ [MBits]", fontsize=30, fontweight='bold')
 
	# Add annotation "(a)" on the top-right of the plot
	plt.text(0.65, 0.95, '(b)', transform=plt.gca().transAxes, fontsize=38, fontweight='bold', ha='right', va='top')

	# Add a legend
	plt.legend(bars, algorithms, fontsize=30, loc='upper left')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_learning, "total_bits_processed.png"))

	##################### COMPARATIVE PLOT FOR ENERGY #####################

	# Read the data dictionary from pickle
	with open('big_run_plots_equal/multiple_runs_500/algorithms_expended_energy_avg.pkl', 'rb') as handle:
		data_dict = pickle.load(handle)

	# Extract the keys and values
	algorithms = list(data_dict.keys())

	# Remove energy level from the name of algorithms
	algorithms = [algorithm.replace("Energy Level", "") for algorithm in algorithms]

	# Rename the second algorithm to "BRAVE-EXPO"
	algorithms[1] = "BRAVE-EXPO"
	
	# Rename the first algorithm to "BRAVE-PRO"
	algorithms[0] = "BRAVE-PRO"
 
	# Rename the third algorith to "BRAVE-GREEDY"
	algorithms[2] = "BRAVE-GREEDY"
 
	# Rename the fourth algorithm to "Q-BRAVE"
	algorithms[3] = "Q-BRAVE"

	energy = list(data_dict.values())

	# Convert to floats if necessary
	energy = [float(e) for e in energy]

	# Define colors for each bar
	colors = ['blue', 'green', 'red', 'grey']

	# Create the figure
	plt.figure(figsize=(12, 8))
 
	# Divide by 1e6 for better visualization
	energy_temp = [e / 1e6 for e in energy]

	# Create a bar plot
	bars = plt.bar(algorithms, energy_temp, color=colors)

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the y-tick parameters
	plt.yticks(fontsize=30, fontweight='bold')

	# Assign y_lim
	plt.ylim(0.6, 1.1)
 
	# Add x-grid
	plt.grid(axis='x', linestyle='--', alpha=1)
 
	# Add y-grid
	plt.grid(axis='y', linestyle='--', alpha=1)

	# Remove the x-axis ticks and labels
	plt.xticks([])

	# Set the y-axis label
	plt.ylabel(r"UAV's Consumed Energy $\boldsymbol{E}_\boldsymbol{P}$ [MJoules]", fontsize=27, fontweight='bold')

	# Add a legend
	plt.legend(bars, algorithms, fontsize=30, loc='upper left')

	# Add annotation "(a)" on the top-right of the plot
	plt.text(0.65, 0.95, '(a)', transform=plt.gca().transAxes, fontsize=38, fontweight='bold', ha='right', va='top')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_learning, "energy_expended.png"))

	##################### COMPARATIVE PLOT FOR TOTAL VISITED NODES #####################

	# Read the data dictionary from pickle
	with open('big_run_plots_equal/multiple_runs_500/algorithms_total_visited_nodes_avg.pkl', 'rb') as handle:
		data_dict = pickle.load(handle)

	# Extract the keys and values
	algorithms = list(data_dict.keys())

	# Remove total visited nodes from the name of algorithms
	algorithms = [algorithm.replace("Total Visited Nodes", "") for algorithm in algorithms]
 
	# Rename the second algorithm to "BRAVE-EXPO"
	algorithms[1] = "BRAVE-EXPO"
	
	# Rename the first algorithm to "BRAVE-PRO"
	algorithms[0] = "BRAVE-PRO"
 
	# Rename the third algorith to "BRAVE-GREEDY"
	algorithms[2] = "BRAVE-GREEDY"
 
	# Rename the fourth algorithm to "Q-BRAVE"
	algorithms[3] = "Q-BRAVE"

	visited_nodes = list(data_dict.values())

	# Define colors for each bar
	colors = ['blue', 'green', 'red', 'grey']

	# Create the figure
	plt.figure(figsize=(12, 8))

	# Create a bar plot
	bars = plt.bar(algorithms, visited_nodes, color=colors)

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the y-tick parameters
	plt.yticks(fontsize=30, fontweight='bold')
 
	# Add annotation "(a)" on the top-right of the plot
	plt.text(0.65, 0.95, '(c)', transform=plt.gca().transAxes, fontsize=38, fontweight='bold', ha='right', va='top')

	# Remove the x-axis ticks and labels
	plt.xticks([])
 
 	# Add x-grid
	plt.grid(axis='x', linestyle='--', alpha=1)
 
	# Add y-grid
	plt.grid(axis='y', linestyle='--', alpha=1)

	# Set the y-axis label
	plt.ylabel(r'Number of Visited AoIs $|\boldsymbol{P}|$', fontsize=38, fontweight='bold')

	# Add a legend
	plt.legend(bars, algorithms, fontsize=25, loc='upper left')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_learning, "total_visited_nodes.png"))

	##################### PURE LEARNING PROPORTIONAL PLOT #####################

	# Define algorithms and values for proportional metric
	algorithms = ["BRAVE-PRO", "BRAVE-EXPO", "BRAVE GREEDY", "Q-BRAVE"]

	values = [(total_bits_processed * total_nodes_visited) / energy_expended for total_bits_processed, total_nodes_visited, energy_expended in zip(bits_processed, visited_nodes, energy)]

	# Define colors
	colors = ['blue', 'green', 'red', 'grey']

	# Create the figure
	plt.figure(figsize=(12, 8))
 
	# Create a bar plot
	bars = plt.bar(algorithms, values, color=colors)

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the y-tick parameters
	plt.yticks(fontsize=15, fontweight='bold')

	# Remove the x-axis ticks and labels
	plt.xticks([])

	# Add x-grid
	plt.grid(axis='x', linestyle='--', alpha=1)
 
	# Add y-grid
	plt.grid(axis='y', linestyle='--', alpha=1)

	# Set the y-axis label
	plt.ylabel(
    r'Efficiency $\frac{\boldsymbol{B}_\boldsymbol{P} \cdot |\boldsymbol{P}|}{\boldsymbol{E}_\boldsymbol{P}}$',
    fontsize=38,
    fontweight='bold'
)

	# Add a legend
	plt.legend(bars, algorithms, fontsize=30, loc='upper left')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_learning, "custom_metric.png"))

if __name__ == "__main__":
	plot_graphs(folder_for_pure_learning= 'plots_algorithm_comparison')