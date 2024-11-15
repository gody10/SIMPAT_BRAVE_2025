import pickle
import os
import matplotlib.pyplot as plt


def plot_results(folder_for_pure_game = 'pure_game'): 
    ##################### USER DATA PLOT #####################
	with open('pure_game/user_data_dict.pkl', 'rb') as handle:
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
				
	# Create the figure for user total bits
	plt.figure(figsize=(12, 8))

	# Plot the total bits of each user
	plt.bar(user_ids, user_bits, color='blue', label='Total Bits')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=38, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Total Bits', fontsize=38, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=30, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=30, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_total_bits.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user data offloaded
	plt.figure(figsize=(12, 8))

	# Apply log scale for better visualization
	#plt.yscale('log')

	# Plot the data offloaded by each user
	plt.bar(user_ids, user_data_offloaded, color='red', label='Data Offloaded')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=38, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Data Offloaded', fontsize=38, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=30, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=30, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_data_offloaded.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user channel gain
	plt.figure(figsize=(12, 8))

	# Plot the channel gain of each user
	plt.bar(user_ids, user_channel_gain, color='green', label='Channel Gain')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=38, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Channel Gain', fontsize=38, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=30, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=30, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_channel_gain.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user data rate
	plt.figure(figsize=(12, 8))

	# Plot the data rate of each user
	plt.bar(user_ids, user_data_rate, color='orange', label='Data Rate')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=38, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Data Rate', fontsize=38, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=30, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=30, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_data_rate.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user total bits
	plt.figure(figsize=(12, 8))

	# Plot the distance from the center of each user
	plt.bar(user_ids, user_distance_from_node, color='blue', label='Distance from Center')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=38, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Distance', fontsize=38, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=30, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=30, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_distance_from_center.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user time overhead
	plt.figure(figsize=(12, 8))

	# Replace inf values in user_time_overhead with 1000000000 for visualization purposes
	#user_time_overhead = [jnp.array([1000000000]) if time_overhead == float('inf') else time_overhead for time_overhead in user_time_overhead]

	# Log scale for better visualization
	#plt.yscale('log')

	# Plot the time overhead of each user
	plt.bar(user_ids, user_time_overhead, color='green', label='Time Overhead')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=38, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Time Overhead', fontsize=38, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=30, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=30, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_time_overhead.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user total overhead
	plt.figure(figsize=(12, 8))

	# Apply log scale for better visualization
	#plt.yscale('log')

	# Plot the total overhead of each user
	plt.bar(user_ids, user_total_overhead, color='orange', label='Total Overhead')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=38, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Total Overhead', fontsize=38, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=30, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=30, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_total_overhead.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user consumed energy
	plt.figure(figsize=(12, 8))

	# Apply log scale for better visualization
	#plt.yscale('log')

	# Plot the consumed energy of each user
	plt.bar(user_ids, user_consumed_energy, color='grey', label='Consumed Energy')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=38, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Consumed Energy', fontsize=38, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=30, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=30, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_consumed_energy.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Create the figure for user utility
	plt.figure(figsize=(12, 8))

	# Apply log scale for better visualization
	#plt.yscale('log')

	# Plot the utility of each user
	plt.bar(user_ids, user_utility, color='purple', label='Utility')

	# Add a grid behind the bars
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Set the x-axis label
	plt.xlabel('User IDs', fontsize=38, fontweight='bold')

	# Set the y-axis label
	plt.ylabel('Utility', fontsize=38, fontweight='bold')

	# Set the x-ticks
	plt.xticks(fontsize=30, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=30, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "user_utility.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()
 
 	##################### GAME CONVERGENCE PLOT #####################
	with open('pure_game/convergence_history.pkl', 'rb') as handle:
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

	# Set the x-ticks
	plt.xticks(fontsize=30, fontweight='bold')

	# Set the y-ticks
	plt.yticks(fontsize=30, fontweight='bold')

	# Save the figure
	plt.savefig(os.path.join(folder_for_pure_game, "convergence_history.png"), bbox_inches='tight')

	# Show the plot
	#plt.show()
 
if __name__ == '__main__':
    plot_results()