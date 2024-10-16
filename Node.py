from typing import List, Tuple
from AoiUser import AoiUser

class Node:
	"""
	Class supporting the Node data structure of SIMPAT-BRAVE PROJECT
	This is also known as Area of Interest (AOI) in the writeup
	"""
	
	def __init__(self, node_id : int, users = List[AoiUser], coordinates : Tuple = (0,0,0), radius: float = 2) -> None:
		"""
		Initialize the node
		
		Parameters:
		users : List[User]
			List of users in the node
		node_id : int
			ID of the node
		coordinates : Tuple
			Coordinates of the node
		radius : float
			radius of the node
		"""
		self.user_list = users
		self.node_id = node_id
		self.coordinates = coordinates
		self.radius = radius
		
		self.total_bit_data = 0
		for user in self.user_list:
			self.total_bit_data += user.get_user_bits()
   
	def get_user_ids(self)->List[int]:
		"""
		Get the IDs of the users in the node

		Returns:
		List[int]
			IDs of the users in the node
		"""
		return [user.get_user_id() for user in self.user_list]
		
	def add_user(self, user : AoiUser)->None:
		"""
		Add a user to the node
		
		Parameters:
		user : User
			User to be added to the node
		"""
		self.user_list.append(user)
		
	def calculate_total_bit_data(self)->None:
		"""
		Calculate the total data in bits of the node
		
		Returns:
		float
			Total data in bits of the node
		"""
		self.total_bit_data = 0
		for user in self.user_list:
			self.total_bit_data += user.get_user_bits()
		
		
	def get_radius(self)->float:
		"""
		Get the radius of the node
		
		Returns:
		float
			Radius of the node
		"""
		return self.radius
		
	def get_node_total_data(self)->float:
		"""
		Get the total data in bits of the node
		
		Returns:
		float
			Total data in bits of the node
		"""
		return self.total_bit_data
	
	def get_node_id(self)->int:
		"""
		Get the ID of the node
		
		Returns:
		int
			ID of the node
		"""
		return self.node_id
	
	def get_coordinates(self)->Tuple:
		"""
		Get the coordinates of the node
		
		Returns:
		Tuple
			Coordinates of the node
		"""
		return self.coordinates
	
	def get_user_list(self)->List[AoiUser]:
		"""
		Get the list of users in the node
		
		Returns:
		List[User]
			List of users in the node
		"""
		return self.user_list
	
	def calculate_total_energy_for_all_user_data_processing(self, uav_processor_power: float, uav_processor_frequency: float)->float:
		"""
		Calculate the total energy for all users data processing
		
		Returns:
		float
			Total energy for all users data processing
		"""
		#total_energy = 0
		#for user in self.get_user_list():
			#total_energy += user.calculate_total_consumed_energy()
		total_energy = uav_processor_power * max((self.get_node_total_data() / uav_processor_frequency), 1)

		return total_energy
		
	def __str__(self)->str:
		return f"Users: {self.user_list}"