import jax.numpy as jnp
import jax
from jax import random
from Utility_functions import generate_node_coordinates
from Graph import Graph
from Edge import Edge
from Node import Node
from AoiUser import AoiUser
from Uav import Uav

# Create a random key
key = random.PRNGKey(10)

# Create N nodes with U users in them
N = 2
U = 20
NODE_RADIUS = 2
MIN_DISTANCE_BETWEEN_NODES = 10  # Minimum distance to maintain between nodes
UAV_HEIGHT = 100
CONVERGENCE_THRESHOLD = 0.01

nodes = []
for i in range(N):
    # Generate random center coordinates for the node
    node_coords = generate_node_coordinates(key, nodes, MIN_DISTANCE_BETWEEN_NODES)
    
    users = []
    for j in range(U):
        # Generate random polar coordinates (r, theta, phi) within the radius of the node
        r = NODE_RADIUS * random.uniform(random.split(key)[0], (1,))[0]  # distance from the center within radius
        theta = random.uniform(random.split(key)[0], (1,))[0] * 2 * jnp.pi  # azimuthal angle (0 to 2*pi)
        phi = random.uniform(random.split(key)[0], (1,))[0] * jnp.pi  # polar angle (0 to pi)
        
        # Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z)
        x = r * jnp.sin(phi) * jnp.cos(theta)
        y = r * jnp.sin(phi) * jnp.sin(theta)
        z = r * jnp.cos(phi)
        
        # User coordinates relative to the node center
        user_coords = (node_coords[0] + x, node_coords[1] + y, node_coords[2] + z)
        
        users.append(AoiUser(
            user_id=j,
            data_in_bits=random.uniform(random.split(key + j)[0], (1,))[0] * 10,
            transmit_power= random.uniform(random.split(key + j)[0], (1,))[0] * 2,
            energy_level= 4000,
            task_intensity= 1,
            carrier_frequency= 5,
            coordinates=user_coords
        ))
    
    nodes.append(Node(
        node_id=i,
        users=users,
        coordinates=node_coords
    ))

# Create edges between all nodes with random weights
edges = []
for i in range(N):
    for j in range(i+1, N):
        edges.append(Edge(nodes[i].user_list[0], nodes[j].user_list[0], random.normal(key, (1,))))
        
# Create the graph
graph = Graph(nodes= nodes, edges= edges)

# Get number of nodes and edges
print(f"Number of Nodes: {graph.get_num_nodes()}")
print(f"Number of Edges: {graph.get_num_edges()}")
print(f"Number of Users: {graph.get_num_users()}")

# Create a UAV
uav = Uav(uav_id= 1, energy_level= 100000 , initial_node= nodes[0], final_node= nodes[len(nodes)-1], total_data_processing_capacity= 1000, velocity= 1, uav_system_bandwidth= 15, cpu_frequency= 2, height= UAV_HEIGHT)

uav_has_reached_final_node = False
break_flag = False

while(not break_flag):
    
    # Check if the game has been played in the current node
    if(not uav.get_finished_business_in_node()):
        # Start playing the game inside the current node
        done = False
        user_strategies = jnp.ones(U) * 0.5  # Strategies for all users
        method = "cvxpy"
        uav_bandwidth = uav.get_uav_bandwidth()
        uav_cpu_frequency = uav.get_cpu_frequency()
        uav_total_data_processing_capacity = uav.get_total_data_processing_capacity()

        user_channel_gains = jnp.zeros(U)
        user_transmit_powers = jnp.zeros(U)
        user_data_in_bits = jnp.zeros(U)

        for idx, user in enumerate(uav.get_current_node().get_user_list()):
            # Assing the channel gain and transmit power to the user
            user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
            user_transmit_powers = user_transmit_powers.at[idx].set(user.get_transmit_power())
            user_data_in_bits = user_data_in_bits.at[idx].set(user.get_user_bits())
            user.set_user_strategy(user_strategies[idx])
            
            
        iteration_counter = 0
        while(not done):
            
            iteration_counter += 1
            previous_strategies = user_strategies

            # Iterate over users and pass the other users' strategies
            for idx, user in enumerate(uav.get_current_node().get_user_list()):
                # Exclude the current user's strategy
                #print("Playing game with user: ", idx)
                
                # Exclude the current user's strategy
                other_user_strategies = jnp.concatenate([user_strategies[:idx], user_strategies[idx+1:]])
                
                # Exclude the current user's channel gain, transmit power and data in bits
                other_user_channel_gains = jnp.concatenate([user_channel_gains[:idx], user_channel_gains[idx+1:]])
                other_user_transmit_powers = jnp.concatenate([user_transmit_powers[:idx], user_transmit_powers[idx+1:]])
                other_user_data_in_bits = jnp.concatenate([user_data_in_bits[:idx], user_data_in_bits[idx+1:]])
                
                
                if method == "cvxpy":
                    # Play the submodular game
                    maximized_utility, percentage_offloaded = user.play_submodular_game_cvxpy(other_user_strategies, 1, 1, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
                                                                                            uav_cpu_frequency, uav_total_data_processing_capacity, 2, uav.get_current_coordinates(), uav.get_height())
                    
                    # Update the user's strategy
                    user_strategies = user_strategies.at[idx].set(percentage_offloaded[0][0])
                    
                    # Update user's channel gain
                    user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())
                    
                else:
                    # Play the submodular game
                    maximized_utility, percentage_offloaded = user.play_submodular_game_scipy(other_user_strategies, 1, 1, uav_bandwidth, other_user_channel_gains, other_user_transmit_powers, other_user_data_in_bits, 
                                                                                            uav_cpu_frequency, uav_total_data_processing_capacity, 2, uav.get_current_coordinates(), uav.get_height())
                    
                    # Update the user's strategy
                    user_strategies = user_strategies.at[idx].set(percentage_offloaded[0])
                    
                    # Update user's channel gain
                    user_channel_gains = user_channel_gains.at[idx].set(user.get_channel_gain())

            # Check how different the strategies are from the previous iteration    
            strategy_difference = jnp.linalg.norm(user_strategies - previous_strategies)
            
            # Check if the strategies have converged
            if strategy_difference < CONVERGENCE_THRESHOLD:
                done = True
        uav.set_finished_business_in_node(True)
        print("The UAV has finished its business in the current node")
        print(f"Converged with strategy difference: {strategy_difference} in {iteration_counter} iterations")
        
        if (uav_has_reached_final_node):
            print("The UAV has reached the final node")
            break_flag = True
        #print(f"Final Strategies: {user_strategies}")
    else:
        # Decide to which node to move next randomly from the ones availalbe that are not visited
        next_node = uav.get_random_unvisited_next_node(nodes= graph.get_nodes(), key= key)
        if (uav.travel_to_node(next_node)):
            print("The UAV has reached the next node")
            # Check if the UAV has reached the final node
            if uav.get_current_node() == uav.get_final_node():
                uav_has_reached_final_node = True
        else:
            print("The UAV has not reached the next node because it has not enough energy")
            break

trajectory = uav.get_visited_nodes()
trajectory_ids = []
for node in trajectory:
    trajectory_ids.append(node.get_node_id())
    
print(" The UAV trajectory is: ", trajectory_ids)


