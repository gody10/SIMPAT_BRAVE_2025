import matplotlib.pyplot as plt
import pickle
import os

def plot_graphs_multi_agent_vs(folder_path: str = "multi_q_learning_results") -> None:
    
    # Import the data
    with open('multi_q_learning_total_bits.pkl', 'rb') as f:
        algorithms_total_bits_acc = pickle.load(f)
        
    with open('multi_q_learning_expended_energy.pkl', 'rb') as f:
        algorithms_expended_energy_acc = pickle.load(f)
        
    with open('multi_q_learning_total_visited_nodes.pkl', 'rb') as f:
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
    colors = ['blue', 'red', 'orange', 'green']
    
    # Set the width of the bars
    bar_width = 0.6
    
    # Make a bar plot for the total bits processed
    plt.figure()
    
    for i, (algorithm_name, color) in enumerate(zip(algorithm_names, colors)):
        # Plot the bar
        bar_value = algorithms_total_bits_acc[algorithm_name]
        bar_value = float(bar_value)

        plt.bar(i, bar_value, color=color, width=bar_width)
        
        # Add text within the bar
        plt.text(i, bar_value / 2, algo_names[i], ha='center', va='center', color='black', fontweight='bold', fontsize= 15, rotation=90)

    # Set x-axis labels and other configurations
    #plt.xticks(range(len(algorithm_names)), algorithm_names_plain)
    plt.xlabel('Algorithm')
    plt.ylabel('Total Bits Processed')
    plt.title('Total Bits Processed by Each Algorithm')

    # Save the plot
    plt.savefig(os.path.join(folder_path, 'total_bits.png'))
    
    # Make a bar plot for the expended energy
    plt.figure()
    
        # Get algorithm names
    algorithm_names = list(algorithms_expended_energy_acc.keys())
    
    # Replace the Total Bits in the title with nothing
    algorithm_names_plain = [name.replace("Energy Level", "") for name in algorithm_names]
    
    
    for i, (algorithm_name, color) in enumerate(zip(algorithm_names, colors)):
        bar_value = algorithms_expended_energy_acc[algorithm_name]
        bar_value = float(bar_value)

        plt.bar(i, bar_value, color=color, width= bar_width)

        # Add text within the bar
        plt.text(i, bar_value / 2, algo_names[i], ha='center', va='center', color='black', fontweight='bold', fontsize= 15,rotation=90)
        
    #plt.xticks(range(len(algorithm_names)), algorithm_names_plain)
    
    plt.xlabel('Algorithm')
    
    plt.ylabel('Expended Energy')
    
    plt.title('Expended Energy by Each Algorithm')
    
    plt.savefig(os.path.join(folder_path, 'expended_energy.png'))
    
    # Make a bar plot for the total visited nodes
    plt.figure()
    
    # Get algorithm names
    algorithm_names = list(algorithms_total_visited_nodes_acc.keys())
    
    # Replace the Total Bits in the title with nothing
    algorithm_names_plain = [name.replace("Total Visited Nodes", "") for name in algorithm_names]
    
    for i, (algorithm_name, color) in enumerate(zip(algorithm_names, colors)):
        bar_value = algorithms_total_visited_nodes_acc[algorithm_name]
        bar_value = float(bar_value)

        plt.bar(i, bar_value, color=color, width= bar_width)

        # Add text within the bar
        plt.text(i, bar_value / 2, algo_names[i], ha='center', va='center', color='black', fontweight='bold',fontsize= 15,rotation=90)
        
    #plt.xticks(range(len(algorithm_names)), algorithm_names_plain)
    
    plt.xlabel('Algorithm')
    
    plt.ylabel('Total Visited Nodes')
    
    plt.title('Total Visited Nodes by Each Algorithm')
    
    plt.savefig(os.path.join(folder_path, 'total_visited_nodes.png'))

    # Plot Custom Metric
    custom_metric = {}

    algorithm_names = list(algorithms_total_bits_acc.keys())
    algorithm_names_plain = [name.replace("Total Bits", "") for name in algorithm_names]

    for algorithm_name in algorithm_names_plain:
        custom_metric[algorithm_name + "Custom Metric"] = ( float(algorithms_total_bits_acc[algorithm_name + "Total Bits"]) \
        * float(algorithms_total_visited_nodes_acc[algorithm_name + "Total Visited Nodes"])) \
        / float(algorithms_expended_energy_acc[algorithm_name + "Energy Level"])
        
    print("Custom Metric for Each Algorithm: ", custom_metric)
    
    plt.figure()

    for i, (algorithm_name, color) in enumerate(zip(algorithm_names_plain, colors)):
        bar_value = custom_metric[algorithm_name + "Custom Metric"]
        bar_value = float(bar_value)

        plt.bar(i, bar_value, color=color, width= bar_width)

        # Add text within the bar
        plt.text(i, bar_value / 2, algo_names[i], ha='center', va='center', color='black', fontweight='bold',fontsize= 15,rotation=90)

    #plt.xticks(range(len(algorithm_names)), algorithm_names_plain)

    plt.xlabel('Algorithm')

    plt.ylabel('Custom Metric')

    plt.title('Custom Metric by Each Algorithm')

    plt.savefig(os.path.join(folder_path, 'custom_metric.png'))

    # Import time dictionary
    with open('multi_q_learning_timers.pkl', 'rb') as f:
        timer_dict_acc = pickle.load(f)

    algorithm_names = list(timer_dict_acc.keys())

    algorithm_names_plain = [name.replace("Time", "") for name in algorithm_names]

    # Plot the time taken by each algorithm
    plt.figure()

    for i, (algorithm_name, color) in enumerate(zip(algorithm_names, colors)):
        bar_value = timer_dict_acc[algorithm_name]
        bar_value = float(bar_value)

        plt.bar(i, bar_value, color=color, width= bar_width)

        # Add text within the bar
        plt.text(i, bar_value / 2, algo_names[i], ha='center', va='center', color='black', fontweight='bold',fontsize= 15,rotation=90)

    #plt.xticks(range(len(algorithm_names)), algorithm_names_plain)

    plt.xlabel('Algorithm')

    plt.ylabel('Time Taken (s)')

    plt.title('Time Taken by Each Algorithm')

    plt.savefig(os.path.join(folder_path, 'time_taken.png'))
    
if __name__ == '__main__':
    plot_graphs_multi_agent_vs(folder_path=  'test')