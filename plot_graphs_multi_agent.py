import matplotlib.pyplot as plt
import pickle
import os

def plot_graphs_multi_agent(folder_path: str = "multi_q_learning_results") -> None:
    
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
               
    # Get algorithm names
    algorithm_names = list(algorithms_total_bits_acc.keys())
    
    # Replace the Total Bits in the title with nothing
    algorithm_names_plain = [name.replace("Total Bits", "") for name in algorithm_names]
    
    # Define the colors used
    colors = ['blue', 'red']
    
    # Set the width of the bars
    bar_width = 0.6
    
    # Make a bar plot for the total bits processed
    plt.figure()
    
    for i, (algorithm_name, color) in enumerate(zip(algorithm_names, colors)):
        plt.bar(i, algorithms_total_bits_acc[algorithm_name], color=color, label=algorithm_names_plain[i], width= bar_width)
        
    plt.xticks(range(len(algorithm_names)), algorithm_names_plain)
    
    plt.xlabel('Algorithm')
    
    plt.ylabel('Total Bits Processed')
    
    plt.title('Total Bits Processed by Each Algorithm')
    
    plt.legend()
    
    plt.savefig(os.path.join(folder_path, 'total_bits.png'))
    
    # Make a bar plot for the expended energy
    plt.figure()
    
        # Get algorithm names
    algorithm_names = list(algorithms_expended_energy_acc.keys())
    
    # Replace the Total Bits in the title with nothing
    algorithm_names_plain = [name.replace("Energy Level", "") for name in algorithm_names]
    
    
    for i, (algorithm_name, color) in enumerate(zip(algorithm_names, colors)):
        plt.bar(i, algorithms_expended_energy_acc[algorithm_name], color=color, label=algorithm_names_plain[i], width= bar_width)
        
    plt.xticks(range(len(algorithm_names)), algorithm_names_plain)
    
    plt.xlabel('Algorithm')
    
    plt.ylabel('Expended Energy')
    
    plt.title('Expended Energy by Each Algorithm')
    
    plt.legend()
    
    plt.savefig(os.path.join(folder_path, 'expended_energy.png'))
    
    # Make a bar plot for the total visited nodes
    plt.figure()
    
    # Get algorithm names
    algorithm_names = list(algorithms_total_visited_nodes_acc.keys())
    
    # Replace the Total Bits in the title with nothing
    algorithm_names_plain = [name.replace("Total Visited Nodes", "") for name in algorithm_names]
    
    for i, (algorithm_name, color) in enumerate(zip(algorithm_names, colors)):
        plt.bar(i, algorithms_total_visited_nodes_acc[algorithm_name], color=color, label=algorithm_names_plain[i], width= bar_width)
        
    plt.xticks(range(len(algorithm_names)), algorithm_names_plain)
    
    plt.xlabel('Algorithm')
    
    plt.ylabel('Total Visited Nodes')
    
    plt.title('Total Visited Nodes by Each Algorithm')
    
    plt.legend()
    
    plt.savefig(os.path.join(folder_path, 'total_visited_nodes.png'))
    
    
if __name__ == '__main__':
    plot_graphs_multi_agent()