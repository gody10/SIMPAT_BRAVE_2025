class AoiUser:
    """
    Class supporting the User data structure of SIMPAT-BRAVE PROJECT
    Represents a user in an Area of Interest (AOI)
    """

    def __init__(self, user_id:int, data_in_bits : float)->None:
        """
        Initialize the user
        
        Parameters:
        user_id : int
            ID of the user
        data_in_bits : float
            Amount of data in bits of the user
        """
        self.user_id = user_id
        self.data_in_bits = data_in_bits

    def get_user_bits(self)->float:
        """
        Get the amount of data in bits of the user
        
        Returns:
        float
            Amount of data in bits of the user
        """
        return self.data_in_bits
        
    def __str__(self)->str:
        return f"User ID: {self.user_id}"