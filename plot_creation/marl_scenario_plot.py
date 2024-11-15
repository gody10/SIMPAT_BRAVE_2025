import matplotlib.pyplot as plt
import pickle
import os

def plot_graphs_multi_agent_vs(folder_path: str = "multi_q_learning_results", folder_to_load = "final_results_500_5_ep") -> None:
    
    # Import the data
    with open(folder_to_load + '/multi_q_learning_total_bits.pkl', 'rb') as f:
        algorithms_total_bits_acc = pickle.load(f)
        
    with open(folder_to_load + '/multi_q_learning_expended_energy.pkl', 'rb') as f:
        algorithms_expended_energy_acc = pickle.load(f)
        
    with open(folder_to_load + '/multi_q_learning_total_visited_nodes.pkl', 'rb') as f:
        algorithms_total_visited_nodes_acc = pickle.load(f)
        
    # Print all the values
    print("Total Bits Processed by Each Algorithm")
    print(algorithms_total_bits_acc)
    
    print("Expended Energy by Each Algorithm")
    print(algorithms_expended_energy_acc)
    
    print("Total Visited Nodes by Each Algorithm")
    print(algorithms_total_visited_nodes_acc)
        
    # Check if folder path exists otherwise create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
               
    algo_names = ['Coop Learning', 'Indi Learning', 'Indi Learning Double Ep', 'Indi Learning Triple Ep']

    # Get algorithm names
    algorithm_names = list(algorithms_total_bits_acc.keys())

    # Replace the Total Bits in the title with nothing
    algorithm_names_plain = [name.replace("Total Bits", "") for name in algorithm_names]

    # Define the colors used
    if len(algorithm_names) == 4:
        colors = ['blue', 'red', 'orange', 'green']
    else:
        colors = ['blue', 'red', 'orange']

    # Set the width of the bars
    bar_width = 0.6

    # Function to customize and save plots
    def create_bar_plot(data_dict, ylabel, save_name, legend_loc='upper left', legend_frame=True):
        plt.figure()

        # Get algorithm names
        algorithm_names = list(data_dict.keys())

        # Calculate the minimum and maximum bar values
        min_value = min(float(data_dict[algo_name]) for algo_name in algorithm_names)
        max_value = max(float(data_dict[algo_name]) for algo_name in algorithm_names)

        # Define a buffer around the max value to zoom in on differences
        buffer = (max_value - min_value) * 0.1  # 10% of the range
        zoomed_min = max(min_value - buffer, 0)  # Ensure y-axis doesn't go below zero
        zoomed_max = max_value + buffer

        for i, (algorithm_name, color) in enumerate(zip(algorithm_names, colors)):
            # Plot the bar
            bar_value = float(data_dict[algorithm_name])
            plt.bar(i, bar_value, color=color, width=bar_width, label=algo_names[i])

            # Add text within the bar
            plt.text(i, bar_value / 2, algo_names[i], ha='center', va='center', color='black', 
                    fontweight='bold', fontsize=15, rotation=90)

        # Adjust y-axis to zoom in
        plt.ylim(zoomed_min, zoomed_max)

        # Set x-axis labels and other configurations with increased font size and bold
        plt.xlabel('Algorithm', fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')

        # Remove x-ticks
        plt.xticks([])
        
        # Add bold to y_ticks
        plt.yticks(fontweight='bold')

        # Add a legend
        plt.legend(title="Algorithms", loc=legend_loc, frameon=legend_frame)

        # Add grids
        plt.grid(axis='x')
        
        # Add grid lines only for y-axis
        plt.grid(axis='y')

        # Save the plot
        plt.savefig(os.path.join(folder_path, save_name))
        plt.close()

    # Make a bar plot for the total bits processed
    create_bar_plot(algorithms_total_bits_acc, 'Total Bits Processed', 'total_bits.png', legend_loc= 'upper right', legend_frame=False)

    # Make a bar plot for the expended energy
    create_bar_plot(algorithms_expended_energy_acc, 'Expended Energy', 'expended_energy.png', 
                legend_loc='upper left', legend_frame=False)

    # Make a bar plot for the total visited nodes
    create_bar_plot(algorithms_total_visited_nodes_acc, 'Total Visited Nodes', 'total_visited_nodes.png', 
                legend_loc='upper left', legend_frame=False)

    # Plot Custom Metric
    custom_metric = {}

    algorithm_names = list(algorithms_total_bits_acc.keys())
    algorithm_names_plain = [name.replace("Total Bits", "") for name in algorithm_names]

    for algorithm_name in algorithm_names_plain:
        custom_metric[algorithm_name + "Custom Metric"] = (
            float(algorithms_total_bits_acc[algorithm_name + "Total Bits"]) *
            float(algorithms_total_visited_nodes_acc[algorithm_name + "Total Visited Nodes"])
        ) / float(algorithms_expended_energy_acc[algorithm_name + "Energy Level"])

    print("Custom Metric for Each Algorithm: ", custom_metric)

    plt.figure()

    for i, (algorithm_name, color) in enumerate(zip(algorithm_names_plain, colors)):
        bar_value = float(custom_metric[algorithm_name + "Custom Metric"])
        plt.bar(i, bar_value, color=color, width=bar_width, label=algo_names[i])

        # Add text within the bar
        # Uncomment the next line if you want to add text inside the bars
        # plt.text(i, bar_value / 2, algo_names[i], ha='center', va='center', color='black', 
        #          fontweight='bold', fontsize=15, rotation=90)

    # Set x-axis labels and other configurations with increased font size and bold
    plt.xlabel('Algorithm', fontsize=14, fontweight='bold')
    plt.ylabel('Custom Metric', fontsize=14, fontweight='bold')

    # Remove x-ticks
    plt.xticks([])
    
    # Add bold to y_ticks
    plt.yticks(fontweight='bold')

    # Add grids
    plt.grid(axis='x')
    plt.grid(axis='y')

    # Set y-axis limit
    plt.ylim(min(custom_metric.values()) - 50, max(custom_metric.values()) + 50)

    # Add a legend
    plt.legend(title="Algorithms", loc='upper left')

    # Save the plot
    plt.savefig(os.path.join(folder_path, 'custom_metric.png'))
    plt.close()

    # Import time dictionary
    with open(os.path.join(folder_to_load, 'multi_q_learning_timers.pkl'), 'rb') as f:
        timer_dict_acc = pickle.load(f)

    # Get algorithm names
    algorithm_names = list(timer_dict_acc.keys())
    algorithm_names_plain = [name.replace("Time", "") for name in algorithm_names]

    # Calculate the minimum and maximum bar values
    min_value = min(float(timer_dict_acc[algo_name]) for algo_name in algorithm_names)
    max_value = max(float(timer_dict_acc[algo_name]) for algo_name in algorithm_names)

    # Define a buffer around the max value to zoom in on differences
    buffer = (max_value - min_value) * 0.1  # 10% of the range
    zoomed_min = max(min_value - buffer, 0)  # Ensure y-axis doesn't go below zero
    zoomed_max = max_value + buffer

    # Plot the time taken by each algorithm
    plt.figure()

    for i, (algorithm_name, color) in enumerate(zip(algorithm_names, colors)):
        bar_value = float(timer_dict_acc[algorithm_name])
        plt.bar(i, bar_value, color=color, width=bar_width, label=algo_names[i])

        # Add text within the bar
        # Uncomment the next line if you want to add text inside the bars
        # plt.text(i, bar_value / 2, algo_names[i], ha='center', va='center', color='black', 
        #          fontweight='bold', fontsize=15, rotation=90)

    # Adjust y-axis to zoom in
    plt.ylim(zoomed_min, zoomed_max)

    # Set x-axis labels and other configurations with increased font size and bold
    plt.xlabel('Algorithm', fontsize=14, fontweight='bold')
    plt.ylabel('Time Taken (s)', fontsize=14, fontweight='bold')

    # Remove x-ticks
    plt.xticks([])
    
    # Add bold to y_ticks
    plt.yticks(fontweight='bold')

    # Add grids
    plt.grid(axis='x')
    plt.grid(axis='y')

    # Add a legend
    plt.legend(title="Algorithms", loc='upper left', frameon=False)

    # Save the plot
    plt.savefig(os.path.join(folder_path, 'time_taken.png'))
    plt.close()
    
if __name__ == '__main__':
    plot_graphs_multi_agent_vs(folder_path= 'multi_q_learning_results_8_ep', folder_to_load='final_results_500_8_ep')