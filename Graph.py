from typing import List
from Node import Node
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Graph:
	"""
	Class supporting the Graph data structure of SIMPAT-BRAVE PROJECT
	Represents the graph of the project where the UAV will navigate
	"""
 
	def __init__(self, nodes : List[Node], edges : List) -> None:
		"""
		Initialize the graph
		
		Parameters:
		nodes : list
			List of nodes in the graph
		edges : list
			List of edges in the graph
		"""
		self.nodes = nodes
		self.edges = edges
	
	def plot_3d_graph(self):
		"""
		Plot the 3D graph of the project with custom node colors and edge weights.
		"""

		# Create a NetworkX graph
		G = nx.Graph()

		# Add nodes with (x, y, z) coordinates
		for node in self.nodes:
			# Increase the IDs of the nodes
			node_id = node.get_node_id()  # Increase ID by 100
			G.add_node(node_id, pos=node.get_coordinates())

		# Add edges with weights
		for edge in self.edges:
			user1 = edge.user1.get_node_id()
			user2 = edge.user2.get_node_id()
			G.add_edge(user1, user2, weight=edge.w)

		# Extract positions for nodes to plot them in 3D
		pos = nx.get_node_attributes(G, 'pos')
		x_vals = [pos[node][0] for node in G.nodes()]
		y_vals = [pos[node][1] for node in G.nodes()]
		z_vals = [pos[node][2] for node in G.nodes()]

		# 3D Plot
		fig = plt.figure(figsize=(10, 10))
		ax = fig.add_subplot(111, projection='3d')

		# Define colors for nodes
		node_colors = []
		for node in G.nodes():
			if node == 0:  # Node with ID 0
				node_colors.append("blue")
			elif node == 5:  # Node with ID 5
				node_colors.append("red")
			else:
				node_colors.append("green")

		# Draw nodes with colors
		ax.scatter(x_vals, y_vals, z_vals, s=100, c=node_colors, depthshade=True)

		# Draw edges and plot their weights
		for edge in G.edges():
			x_edge = [pos[edge[0]][0], pos[edge[1]][0]]
			y_edge = [pos[edge[0]][1], pos[edge[1]][1]]
			z_edge = [pos[edge[0]][2], pos[edge[1]][2]]
			ax.plot(x_edge, y_edge, z_edge, color="gray", alpha=0.5)

			# Calculate the midpoint for placing the weight label
			mid_x = (x_edge[0] + x_edge[1]) / 2
			mid_y = (y_edge[0] + y_edge[1]) / 2
			mid_z = (z_edge[0] + z_edge[1]) / 2
			weight = G[edge[0]][edge[1]]['weight']
			ax.text(mid_x, mid_y, mid_z, f'{weight:.2f}', color='purple', fontsize=15)

		# Annotate nodes with updated labels
		for node_id, (x, y, z) in pos.items():
			ax.text(x, y, z, f'{node_id}', size=20, zorder=1, color='k')

		plt.title("3D Graph Visualization with Node Colors and Edge Weights")
		#plt.show()
		plt.savefig("3d_realistic_graph_2.png")

	def get_node_by_id(self, node_id):

		"""
		Returns the Node object with the given node_id.
		Raises KeyError if not found.
		"""
		
		# Loop through the nodes to find the node with the given node_id
		for node in self.nodes:
			if node.get_node_id() == node_id:
				return node

	def get_neighbors(self, node_id : int):
		"""
		Get the neighbors of a node in the graph
		
		Parameters:
		node_id : int
			ID of the node whose neighbors are to be found
		
		Returns:
		list
			List of edges of the neighbors of the node
		"""
		neighbors = []
		for edge in self.edges:
			if edge.user1.get_node_id() == node_id:
				neighbors.append(edge)
			# elif edge.user2.get_node_id() == node_id:
			# 	neighbors.append(edge)
		
		return neighbors

	def add_node(self, node : Node)->None:
		"""
		Add a node to the graph
		
		Parameters:
		node : any
			Node to be added to the graph
		"""
		self.nodes.append(node)
		
	def add_edge(self, edge)->None:
		"""
		Add an edge to the graph
		
		Parameters:
		edge : tuple
			Edge to be added to the graph
		"""
		self.edges.append(edge)
		
	def get_num_nodes(self)->int:
		"""
		Get the number of nodes in the graph
		
		Returns:
		int
			Number of nodes in the graph
		"""
		return len(self.nodes)
	
	def get_nodes(self)->List[Node]:
		"""
		Get the nodes in the graph
		
		Returns:
		list
			List of nodes in the graph
		"""
		return self.nodes
	
	def get_num_edges(self)->int:
		"""
		Get the number of edges in the graph
		
		Returns:
		int
			Number of edges in the graph
		"""
		return len(self.edges)
	
	def get_num_users(self)->int:
		"""
		Get the number of users in the graph
		
		Returns:
		int
			Number of users in the graph
		"""
		num_users = 0
		for node in self.nodes:
			num_users += len(node.user_list)
		return num_users
	
	def get_edges(self)->List:
		"""
		Get the edges in the graph
		
		Returns:
		list
			List of edges in the graph
		"""
		return self.edges
		
	def __str__(self)->str:
		return f"Nodes: {self.nodes}\nEdges: {self.edges}"