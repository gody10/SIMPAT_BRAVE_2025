from typing import List, Tuple
from AoiUser import AoiUser

class Node:
    """
    Class supporting the Node data structure of SIMPAT-BRAVE PROJECT
    This is also known as Area of Interest (AOI) in the writeup
    """
    
    def __init__(self, node_id : int, users = List[AoiUser], coordinates : Tuple = (0,0,0)) -> None:
        """
        Initialize the node
        
        Parameters:
        users : List[User]
            List of users in the node
        node_id : int
            ID of the node
        coordinates : Tuple
            Coordinates of the node
        """
        self.user_list = users
        self.node_id = node_id
        self.coordinates = coordinates
        
        self.total_bit_data = 0
        for user in self.user_list:
            self.total_bit_data += user.get_user_bits()
        
    def add_user(self, user : AoiUser)->None:
        """
        Add a user to the node
        
        Parameters:
        user : User
            User to be added to the node
        """
        self.user_list.append(user)
        
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
        
    def __str__(self)->str:
        return f"Users: {self.user_list}"