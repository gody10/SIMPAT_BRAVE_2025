import pickle
import matplotlib.pyplot as plt
import os
import jax.numpy as jnp

def plot_graphs():
	"""
	Plot the graphs for the algorithms
	"""
	# Specify the folder to save the plot
	basic_folder = "plots"
	folder_for_pure_learning = "plots/pure_learning"
	folder_for_pure_game = "plots/pure_game"

	os.makedirs(basic_folder, exist_ok=True)
	os.makedirs(folder_for_pure_learning, exist_ok=True)
	os.makedirs(folder_for_pure_game, exist_ok=True)

	##################### COMPARATIVE PLOT FOR BITS #####################

	# Read the data dictionary from pickle
	with open('algorithms_total_bits.pkl', 'rb') as handle:
		data_dict = pickle.load(handle)

	# Extract the keys and values
	algorithms = list(data_dict.keys())

	# Remove total bits from the name of algorithms
	algorithms = [algorithm.replace("Total Bits", "") for algorithm in algorithms]

	bits_processed = list(data_dict.values())

	# Define colors for each bar (for visualization purposes)
	colors = ['blue', 'green', 'orange', 'grey']  # Customize colors as needed

	# Create the figure
	plt.figure(figsize=(12, 8))

	# Create a bar plot with colored bars
	bars = plt.bar(algorithms, bits_processed, color=colors)

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the y-tick parameters
	plt.yticks(fontsize=14, fontweight='bold')

	# Loop over the bars to place text within each bar
	for bar, algorithm in zip(bars, algorithms):
		height = bar.get_height()
		plt.text(
			bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
			height / 2,                        # Y position (half the height of the bar)
			algorithm,                   # Display the name of the algorithm
			ha='center', va='center',          # Centered horizontally and vertically
			fontsize=22, fontweight='bold',    # Customize font size and weight
			color='black',                     # Set text color
			rotation=90                        # Rotate text 90 degrees
		)

	# Remove the x-axis ticks and labels for a clean look
	plt.xticks([])

	# Y-ticks
	plt.yticks(fontsize=20, fontweight='bold')

	# Set the y-axis label with appropriate formatting
	plt.ylabel('Total Bits Processed', fontsize=25, fontweight='bold')

	# Set the plot title
	plt.title('Total Bits Processed by each Algorithm', fontsize=32, fontweight='bold')

	# Tight layout for better spacing
	plt.tight_layout()

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_learning, "total_bits_processed.png"), bbox_inches='tight')

	# Show the plot
	# plt.show()

	##################### COMPARATIVE PLOT FOR ENERGY #####################
	# Read the data dictionary from pickle
	with open('algorithms_expended_energy.pkl', 'rb') as handle:
		data_dict = pickle.load(handle)
	
	# Extract the keys and values
	algorithms = list(data_dict.keys())

	# Remove total bits from the name of algorithms
	algorithms = [algorithm.replace("Total Bits", "") for algorithm in algorithms]

	energy = list(data_dict.values())

	energy = [float(e[0]) for e in energy]

	# Define colors for each bar (for visualization purposes)
	colors = ['blue', 'green', 'orange', 'grey']  # Customize colors as needed

	# Create the figure
	plt.figure(figsize=(12, 8))

	# Create a bar plot with colored bars
	bars = plt.bar(algorithms, energy, color=colors)

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the y-tick parameters
	plt.yticks(fontsize=14, fontweight='bold')

	# Loop over the bars to place text within each bar
	for bar, algorithm in zip(bars, algorithms):
		height = bar.get_height()
		plt.text(
			bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
			height / 2,                        # Y position (half the height of the bar)
			algorithm,                   # Display the name of the algorithm
			ha='center', va='center',          # Centered horizontally and vertically
			fontsize=22, fontweight='bold',    # Customize font size and weight
			color='black',                     # Set text color
			rotation=90                        # Rotate text 90 degrees
		)

	# Remove the x-axis ticks and labels for a clean look
	plt.xticks([])

	# Y-ticks
	plt.yticks(fontsize=20, fontweight='bold')

	# Set the y-axis label with appropriate formatting
	plt.ylabel('Energy Expended', fontsize=25, fontweight='bold')

	# Set the plot title
	plt.title('Energy Expended by each Algorithm', fontsize=32, fontweight='bold')

	# Tight layout for better spacing
	plt.tight_layout()

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_learning, "energy_expended.png"), bbox_inches='tight')

	# Show the plot
	# plt.show()

	##################### COMPARATIVE PLOT FOR TOTAL VISITED NODES #####################
	# Read the data dictionary from pickle
	with open('algorithms_total_visited_nodes.pkl', 'rb') as handle:
		data_dict = pickle.load(handle)
	
	# Extract the keys and values
	algorithms = list(data_dict.keys())

	# Remove total bits from the name of algorithms
	algorithms = [algorithm.replace("Total Bits", "") for algorithm in algorithms]

	visited_nodes = list(data_dict.values())

	# Define colors for each bar (for visualization purposes)
	colors = ['blue', 'green', 'orange', 'grey']  # Customize colors as needed

	# Create the figure
	plt.figure(figsize=(12, 8))

	# Create a bar plot with colored bars
	bars = plt.bar(algorithms, visited_nodes, color=colors)

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the y-tick parameters
	plt.yticks(fontsize=14, fontweight='bold')

	# Loop over the bars to place text within each bar
	for bar, algorithm in zip(bars, algorithms):
		height = bar.get_height()
		plt.text(
			bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
			height / 2,                        # Y position (half the height of the bar)
			algorithm,                   # Display the name of the algorithm
			ha='center', va='center',          # Centered horizontally and vertically
			fontsize=22, fontweight='bold',    # Customize font size and weight
			color='black',                     # Set text color
			rotation=90                        # Rotate text 90 degrees
		)

	# Remove the x-axis ticks and labels for a clean look
	plt.xticks([])

	# Y-ticks
	plt.yticks(fontsize=20, fontweight='bold')

	# Set the y-axis label with appropriate formatting
	plt.ylabel('Total Visited Nodes', fontsize=25, fontweight='bold')

	# Set the plot title
	plt.title('Total Visited Nodes by each Algorithm', fontsize=32, fontweight='bold')

	# Tight layout for better spacing
	plt.tight_layout()

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_learning, "total_visited_nodes.png"), bbox_inches='tight')

	# Show the plot
	# plt.show()

	##################### PURE LEARNING PROPORTIONAL PLOT #####################
	# Read the data dictionary from pickle
	algorithms = ["Random Walk", "Proportional Fairness", "Brave Greedy", "Q-Brave"]

	values = [(total_bits_processed * total_nodes_visited)/energy_expended for total_bits_processed, total_nodes_visited, energy_expended in zip(bits_processed, visited_nodes, energy)]

	# Define colors for each bar (for visualization purposes)
	colors = ['blue', 'green', 'orange', 'grey']  # Customize colors as needed

	# Create the figure
	plt.figure(figsize=(12, 8))

	# Create a bar plot with colored bars
	bars = plt.bar(algorithms, visited_nodes, color=colors)

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the y-tick parameters
	plt.yticks(fontsize=14, fontweight='bold')

	#Loop over the bars to place text within each bar
	for bar, algorithm in zip(bars, algorithms):
		height = bar.get_height()
		plt.text(
			bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
			height / 2,                        # Y position (half the height of the bar)
			algorithm,                   # Display the name of the algorithm
			ha='center', va='center',          # Centered horizontally and vertically
			fontsize=22, fontweight='bold',    # Customize font size and weight
			color='black',                     # Set text color
			rotation=90                        # Rotate text 90 degrees
		)

	# Remove the x-axis ticks and labels for a clean look
	plt.xticks([])

	# Y-ticks
	plt.yticks(fontsize=20, fontweight='bold')

	# Set the y-axis label with appropriate formatting
	plt.ylabel('(B*AoI)/E', fontsize=25, fontweight='bold')

	# Set the plot title
	plt.title('Custom Metric', fontsize=32, fontweight='bold')

	# Tight layout for better spacing
	plt.tight_layout()

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_learning, "custom_metric.png"), bbox_inches='tight')

	# Show the plot
	# plt.show()

	##################### GAME CONVERGENCE PLOT #####################
	with open('convergence_history.pkl', 'rb') as handle:
		convergence_history = pickle.load(handle)
		
	# Create the figure
	plt.figure(figsize=(12, 8))

	# Plot the convergence history
	plt.plot(convergence_history, linewidth=10)

	# Add a grid behind the plot
	plt.grid(axis='both', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('Iterations', fontsize=38, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Error', fontsize=38, fontweight='bold')

	# Set the plot title
	plt.title('Convergence History of the Submodular Game Algorithm', fontsize=25, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=30, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=30, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "convergence_history.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()


	##################### USER DATA PLOT #####################
	with open('user_data_dict.pkl', 'rb') as handle:
		user_data_dict = pickle.load(handle)
	
	# Extract the keys and values
	user_ids = user_data_dict["User IDs"]
	user_bits = user_data_dict["User Total Bits"]
	user_time_overhead = user_data_dict["User Time Overhead"]
	user_total_overhead = user_data_dict["User Total Overhead"]
	user_consumed_energy = user_data_dict["User Consumed Energy"]
	user_utility = user_data_dict["User Utility"]
	user_distance_from_node = user_data_dict["User Distance from Node"]
	user_channel_gain = user_data_dict["User Channel Gain"]
	user_data_rate = user_data_dict["User Data Rate"]
	user_data_offloaded = user_data_dict["User Data Offloaded"]

	# user_time_overhead_temp = []
	# for time_overhead in user_time_overhead:
	#     # Check for infinity
	#     if time_overhead == float('inf'):
	#         user_time_overhead_temp.append(float(1000000000))
	#     else:
	#         # Check if it's an array and if so, convert to a scalar
	#         if isinstance(time_overhead, jnp.ndarray):
	#             # Flatten to make sure it's 1D and extract the first element as a scalar
	#             time_overhead_scalar = time_overhead.flatten()[0]
	#             user_time_overhead_temp.append(float(time_overhead_scalar))
	#         else:
	#             # If it's already a scalar, append directly
	#             user_time_overhead_temp.append(float(time_overhead))
				
	# user_total_overhead_temp = []
	# for time_overhead in user_total_overhead:
	#     # Check for infinity
	#     if time_overhead == float('inf'):
	#         user_total_overhead_temp.append(float(1000000000))
	#     else:
	#         # Check if it's an array and if so, convert to a scalar
	#         if isinstance(time_overhead, jnp.ndarray):
	#             # Flatten to make sure it's 1D and extract the first element as a scalar
	#             total_overhead_scalar = time_overhead.flatten()[0]
	#             user_total_overhead_temp.append(float(time_overhead_scalar))
	#         else:
	#             # If it's already a scalar, append directly
	#             user_total_overhead_temp.append(float(time_overhead))
				
	# user_consumed_energy_temp = []
	# for time_overhead in user_consumed_energy:
	#     # Check for infinity
	#     if time_overhead == float('inf'):
	#         user_consumed_energy_temp.append(float(1000000000))
	#     else:
	#         # Check if it's an array and if so, convert to a scalar
	#         if isinstance(time_overhead, jnp.ndarray):
	#             # Flatten to make sure it's 1D and extract the first element as a scalar
	#             consumed_energy_scalar = time_overhead.flatten()[0]
	#             user_consumed_energy_temp.append(float(time_overhead_scalar))
	#         else:
	#             # If it's already a scalar, append directly
	#             user_consumed_energy_temp.append(float(time_overhead))
				
	# Create the figure for user total bits
	plt.figure(figsize=(12, 8))

	# Plot the total bits of each user
	plt.bar(user_ids, user_bits, color='blue', label='Total Bits')

	# Add a legend to the plot
	plt.legend(fontsize=20, loc='upper left')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=25, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Total Bits', fontsize=25, fontweight='bold')

	# Set the plot title
	plt.title('User Total Bits', fontsize=32, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=20, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=20, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_total_bits.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user data offloaded
	plt.figure(figsize=(12, 8))

	# Plot the data offloaded by each user
	plt.bar(user_ids, user_data_offloaded, color='red', label='Data Offloaded')

	# Add a legend to the plot
	plt.legend(fontsize=20, loc='upper left')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=25, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Data Offloaded', fontsize=25, fontweight='bold')

	# Set the plot title
	plt.title('User Data Offloaded', fontsize=32, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=20, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=20, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_data_offloaded.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user channel gain
	plt.figure(figsize=(12, 8))

	# Plot the channel gain of each user
	plt.bar(user_ids, user_channel_gain, color='green', label='Channel Gain')

	# Add a legend to the plot
	plt.legend(fontsize=20, loc='upper left')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=25, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Channel Gain', fontsize=25, fontweight='bold')

	# Set the plot title
	plt.title('User Channel Gain', fontsize=32, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=20, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=20, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_channel_gain.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user data rate
	plt.figure(figsize=(12, 8))

	# Plot the data rate of each user
	plt.bar(user_ids, user_data_rate, color='orange', label='Data Rate')

	# Add a legend to the plot
	plt.legend(fontsize=20, loc='upper left')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=25, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Data Rate', fontsize=25, fontweight='bold')

	# Set the plot title
	plt.title('User Data Rate', fontsize=32, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=20, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=20, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_data_rate.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user total bits
	plt.figure(figsize=(12, 8))

	# Plot the distance from the center of each user
	plt.bar(user_ids, user_distance_from_node, color='blue', label='Distance from Center')

	# Add a legend to the plot
	plt.legend(fontsize=20, loc='upper right')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=25, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Distance', fontsize=25, fontweight='bold')

	# Set the plot title
	plt.title('User Distance from Node Center', fontsize=32, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=20, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=20, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_distance_from_center.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user time overhead
	plt.figure(figsize=(12, 8))

	# Replace inf values in user_time_overhead with 1000000000 for visualization purposes
	#user_time_overhead = [jnp.array([1000000000]) if time_overhead == float('inf') else time_overhead for time_overhead in user_time_overhead]

	# Log scale for better visualization
	plt.yscale('log')

	# Plot the time overhead of each user
	plt.bar(user_ids, user_time_overhead, color='green', label='Time Overhead')

	# Add a legend to the plot
	plt.legend(fontsize=20, loc='upper left')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=25, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Time Overhead', fontsize=25, fontweight='bold')

	# Set the plot title
	plt.title('User Time Overhead', fontsize=32, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=20, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=20, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_time_overhead.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user total overhead
	plt.figure(figsize=(12, 8))

	# Apply log scale for better visualization
	plt.yscale('log')

	# Plot the total overhead of each user
	plt.bar(user_ids, user_total_overhead, color='orange', label='Total Overhead')

	# Add a legend to the plot
	plt.legend(fontsize=20, loc='upper left')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=25, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Total Overhead', fontsize=25, fontweight='bold')

	# Set the plot title
	plt.title('User Total Overhead', fontsize=32, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=20, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=20, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_total_overhead.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user consumed energy
	plt.figure(figsize=(12, 8))

	# Apply log scale for better visualization
	plt.yscale('log')

	# Plot the consumed energy of each user
	plt.bar(user_ids, user_consumed_energy, color='grey', label='Consumed Energy')

	# Add a legend to the plot
	plt.legend(fontsize=20, loc='upper left')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=25, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Consumed Energy', fontsize=25, fontweight='bold')

	# Set the plot title
	plt.title('User Consumed Energy', fontsize=32, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=20, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=20, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_consumed_energy.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user utility
	plt.figure(figsize=(12, 8))

	# Plot the utility of each user
	plt.bar(user_ids, user_utility, color='purple', label='Utility')

	# Add a legend to the plot
	plt.legend(fontsize=20, loc='upper left')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=25, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Utility', fontsize=25, fontweight='bold')

	# Set the plot title
	plt.title('User Utility', fontsize=32, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=20, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=20, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_utility.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()
