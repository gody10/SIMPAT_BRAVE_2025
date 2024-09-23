import jax
import jax.numpy as jnp
from jax import random
from Graph import Graph
from Node import Node
from AoiUser import AoiUser

# def selu(x, alpha=1.67, lmbda=1.05):
#   return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

# x = jnp.arange(5.0)
# print(selu(x))

# key = random.key(10)
# x = random.normal(key, (1000000,))

# print(key)
# print(x)

# Create dummy examples for all classes just to see they are imported correctly
user1 = User(1, "John", 30)
user2 = User(2, "Jane", 25)
user3 = User(3, "Joe", 35)
node1 = Node([user1, user2])
node2 = Node([user3])
graph = Graph([node1, node2], [(node1, node2)])
print(graph)

