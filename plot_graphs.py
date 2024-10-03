import pickle
import matplotlib.pyplot as plt
import os

# Specify the folder to save the plot
folder_to_save_plots = "plots"
os.makedirs(folder_to_save_plots, exist_ok=True)

##################### COMPARATIVE PLOT #####################

# Read the data dictionary from pickle
with open('data_dict.pkl', 'rb') as handle:
    data_dict = pickle.load(handle)

# Extract the keys and values
algorithms = list(data_dict.keys())

# Remove total bits from the name of algorithms
algorithms = [algorithm.replace("Total Bits", "") for algorithm in algorithms]

bits_processed = list(data_dict.values())

# Define colors for each bar (for visualization purposes)
colors = ['blue', 'green', 'orange']  # Customize colors as needed

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
plt.savefig(os.path.join(folder_to_save_plots, "total_bits_processed.png"), bbox_inches='tight')

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
plt.savefig(os.path.join(folder_to_save_plots, "convergence_history.png"), bbox_inches='tight')

# Show the plot
#plt.show()