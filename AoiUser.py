class AoiUser:
    """
    Class supporting the User data structure of SIMPAT-BRAVE PROJECT
    Represents a user in an Area of Interest (AOI)

    Attributes:
        - TO BE FILLED
    """

    def __init__(self, user_id:int, user_name:str, user_age:int)->None:
        self.user_id = user_id
        self.user_name = user_name
        self.user_age = user_age
        
    def __str__(self)->str:
        return f"User ID: {self.user_id}, User Name: {self.user_name}, User Age: {self.user_age}"