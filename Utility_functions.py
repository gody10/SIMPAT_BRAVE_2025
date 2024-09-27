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
        new_coords = (random.uniform(key, (1,))[0] * 100, 
                      random.uniform(key, (1,))[0] * 100, 
                      random.uniform(key, (1,))[0] * 100)
        
        # Check the distance with all existing nodes
        if all(jnp.linalg.norm(jnp.array(new_coords) - jnp.array(node.coordinates)) >= min_distance_between_nodes
               for node in existing_nodes):
            return new_coords
        # Update key for the next iteration to ensure randomness
        key, _ = random.split(key)