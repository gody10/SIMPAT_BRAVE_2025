import jax.numpy as jnp
from jax import random

def generate_node_coordinates(key, existing_nodes, min_distance_between_nodes=50):
	"""
	Generate random coordinates for a new node that are at least a minimum distance away from all existing nodes
	
	Parameters:
	key : PRNGKey
		Random key for generating random numbers
	existing_nodes : list
		List of existing nodes
	min_distance_between_nodes : float
		Minimum distance to maintain between nodes
	"""
	
	# Create a random key
	key = random.PRNGKey(10)
	
	# Generate random coordinates until we find one that is sufficiently far from all existing nodes
	while True:
		key += 1
		# Create 3 subkeys for generating random numbers
		key, subkey1, subkey2 = random.split(key, 3)
		new_coords = (random.uniform(key, (1,))[0] * 1000, 
					  random.uniform(subkey1, (1,))[0] * 1000, 
					  random.uniform(subkey2, (1,))[0] * 100)

		# Check the distance with all existing nodes
		if all(jnp.sqrt((new_coords[0] - node.coordinates[0]) ** 2 + 
							(new_coords[1] - node.coordinates[1]) ** 2 + 
							(new_coords[2] - node.coordinates[2]) ** 2) >= min_distance_between_nodes
			   for node in existing_nodes):
			return new_coords
		# Update key for the next iteration to ensure randomness
		key, _ = random.split(key)