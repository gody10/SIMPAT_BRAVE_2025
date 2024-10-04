from AoiUser import AoiUser

class Edge:
	"""
	Class to represent an edge in a graph
	Represents the connection between two nodes and the weight of the connection
	"""
	def __init__(self, user1 : AoiUser, user2 : AoiUser, weight : float):
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

	def __str__(self) -> str:
		return f"User 1: {self.user1}, User 2: {self.user2}, Weight: {self.w}"