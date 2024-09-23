from typing import List
from User import User

class Node:
    """
    Class supporting the Node data structure of SIMPAT-BRAVE PROJECT
    This is also known as Area of Interest (AOI) in the writeup
    
    Attributes:
    user_list : List[User]
        List of users in the node
    """
    
    def __init__(self, users = List[User]  ) -> None:
        self.user_list = users
        
    def add_user(self, user : User)->None:
        """
        Add a user to the node
        
        Parameters:
        user : User
            User to be added to the node
        """
        self.user_list.append(user)
        
    def __str__(self)->str:
        return f"Users: {self.user_list}"