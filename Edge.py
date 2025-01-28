from Node import Node

class Edge:
	"""
	Class to represent an edge in a graph
	Represents the connection between two nodes and the weight of the connection
	"""
	def __init__(self, user1 : Node, user2 : Node, weight : float):
		"""
		Initialize the edge
		
		Parameters:
		user1 : AoiUser
			User 1 of the edge
		user2 : AoiUser
			User 2 of the edge
		weight : float
		"""
		self.user1 = user1
		self.user2 = user2
		self.w = weight
  
	def get_distance(self)->float:
		"""
		Get the distance between the two nodes
		
		Returns:
		float
			Distance between the two nodes
		"""
		node_1_coordinates = self.user1.get_coordinates()
		node_2_coordinates = self.user2.get_coordinates()
  
		euclidean_distance = ((node_1_coordinates[0] - node_2_coordinates[0])**2 + (node_1_coordinates[1] - node_2_coordinates[1])**2 + (node_1_coordinates[2] - node_2_coordinates[2])**2 )**0.5
  
		return euclidean_distance

	def __str__(self) -> str:
		return f"User 1: {self.user1}, User 2: {self.user2}, Weight: {self.w}"