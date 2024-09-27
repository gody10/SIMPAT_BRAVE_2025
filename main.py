import jax.numpy as jnp
import logging
from jax import random
from Utility_functions import generate_node_coordinates
from Graph import Graph
from Edge import Edge
from Node import Node
from AoiUser import AoiUser
from Uav import Uav
from Algorithms import Algorithms

# Create a random key
key = random.PRNGKey(10)

# Setup logging
logging.basicConfig(
    filename='algorithm_logs.log',  # Log file name
    filemode='w',            # Mode: 'w' for overwrite, 'a' for append
    level=logging.INFO,      # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format of the log messages
)

# Create N nodes with U users in them
N = 1
U = 10
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
logging.info("Number of Nodes: %s", graph.get_num_nodes())
logging.info("Number of Edges: %s", graph.get_num_edges())
logging.info("Number of Users: %s", graph.get_num_users())

# Create a UAV
uav = Uav(uav_id= 1, initial_node= nodes[0], final_node= nodes[len(nodes)-1], capacity= 10000, total_data_processing_capacity= 1000, velocity= 1, uav_system_bandwidth= 15, cpu_frequency= 2, height= UAV_HEIGHT)

# Create the algorithm object
algorithm = Algorithms(number_of_users= U, number_of_nodes= N, uav= uav, graph= graph, key= key, convergence_threshold= CONVERGENCE_THRESHOLD)

# Run the Random Walk Algorithm
success_var = algorithm.run_random_walk_algorithm(solving_method= "cvxpy")

if success_var:
    logging.info("Algorithm has successfully reached the final node!")
else:
    logging.info("Algorithm failed to reach the final node!")
