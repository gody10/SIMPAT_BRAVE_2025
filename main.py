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
N = 10
U = 20

nodes = []
for i in range(N):
    users = []
    for j in range(U):
        users.append(AoiUser(user_id= j, data_in_bits= random.uniform(random.key(j), (1,))[0]))
    nodes.append(Node(node_id= i, users= users))

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
uav = Uav(uav_id= 1, energy_level= 100, initial_node_id= 0, final_node_id= 9, total_data_processing_capacity= 1000)


