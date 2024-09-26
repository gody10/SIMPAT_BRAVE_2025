import jax
import jax.numpy as jnp
from jax import random
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

nodes = []
for i in range(N):
    users = []
    for j in range(U):
        users.append(AoiUser(user_id= j, data_in_bits= random.uniform(random.key(j), (1,))[0], transmit_power= random.uniform(random.key(j), (1,))[0], channel_gain= random.uniform(random.key(j), (1,))[0], energy_level= random.uniform(random.key(j), (1,))[0]))
    nodes.append(Node(node_id= i, users= users, coordinates= (random.uniform(random.key(i), (1,))[0], random.uniform(random.key(i), (1,))[0], random.uniform(random.key(i), (1,))[0])))

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
uav = Uav(uav_id= 1, energy_level= 100, initial_node= nodes[0], final_node= nodes[len(nodes)-1], total_data_processing_capacity= 1000, velocity= 1)

uav.travel_to_node(nodes[1])

# Execute Algorithms


