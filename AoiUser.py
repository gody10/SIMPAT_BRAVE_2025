import jax.numpy as jnp
import cvxpy as cp

class AoiUser:
    """
    Class supporting the User data structure of SIMPAT-BRAVE PROJECT
    Represents a user in an Area of Interest (AOI)
    """

    def __init__(self, user_id:int, data_in_bits : float, transmit_power: float, channel_gain: float, energy_level: float)->None:
        """
        Initialize the user
        
        Parameters:
        user_id : int
            ID of the user
        data_in_bits : float
            Amount of data in bits of the user
        transmit_power : float
            Transmit power of the user
        channel_gain : float
            Channel gain of the user
        total_energy_capacity : float
            Total energy capacity of the user
        """
        self.user_id = user_id
        self.data_in_bits = data_in_bits
        self.transmit_power = transmit_power
        self.channel_gain = channel_gain
        self.energy_level = energy_level
        self.white_noise = 1e-9
        self.current_strategy = 0
        self.current_data_rate = 0
        self.current_time_overhead = 0
        self.current_consumed_energy = 0
        self.current_total_overhead = 0

    def get_user_bits(self)->float:
        """
        Get the amount of data in bits of the user
        
        Returns:
        float
            Amount of data in bits of the user
        """
        return self.data_in_bits
    
    def get_transmit_power(self)->float:
        """
        Get the transmit power of the user
        
        Returns:
        float
            Transmit power of the user
        """
        return self.transmit_power
    
    def get_channel_gain(self)->float:
        """
        Get the channel gain of the user
        
        Returns:
        float
            Channel gain of the user
        """
        return self.channel_gain
    
    def set_user_strategy(self, strategy: float)->None:
        """
        Set the strategy of the user
        
        Parameters:
        strategy : float
            Strategy of the user
        """
        self.current_strategy = strategy
        
    def get_user_strategy(self)->float:
        """
        Get the strategy of the user
        
        Returns:
        float
            Strategy of the user
        """
        return self.current_strategy
    
    def adjust_energy(self, energy_used: float)->bool:
        """
        Adjust the energy level of the UAV
        
        Parameters:
        energy : float
            Energy level to be adjusted
        """
        if self.energy_level - energy_used < 0:
            print("Energy level goes under 0 - Action not allowed")
            return False
        else:
            self.energy_level -= energy_used
            return True
        
    def get_current_energy_level(self)->float:
        """
        Get the current energy level of the user
        
        Returns:
        float
            Current energy level of the user
        """
        return self.energy_level
    
    def get_current_data_rate(self)->float:
        """
        Get the current data rate of the user
        
        Returns:
        float
            Current data rate of the user
        """
        return self.current_data_rate
    
    def get_current_time_overhead(self)->float:
        """
        Get the current time overhead of the user
        
        Returns:
        float
            Current time overhead of the user
        """
        return self.current_time_overhead
    
    def get_current_consumed_energy(self)->float:
        """
        Get the current consumed energy of the user
        
        Returns:
        float
            Current consumed energy of the user
        """
        return self.current_consumed_energy
    
    def get_current_total_overhead(self)->float:
        """
        Get the current total overhead of the user
        
        Returns:
        float
            Current total overhead of the user
        """
        return self.current_total_overhead
        
    def calculate_data_rate(self, uav_bandwidth: float, other_users_transmit_powers: list, other_users_channel_gains: list)->None:
        """
        Calculate the data rate of the user
        
        Parameters:
        uav_bandwidth : float
            Bandwidth of the UAV
        
        Returns:
        float
            Data rate of the user
        """
        self.current_data_rate = uav_bandwidth* jnp.log(1 + ((self.transmit_power * self.channel_gain) / (self.white_noise + jnp.sum(other_users_transmit_powers * other_users_channel_gains))))
    
    def calculate_time_overhead(self, data_rate: float, phi: list, other_user_strategies: list, other_user_bits: list, uav_total_capacity: float, uav_cpu_frequency: float)->None:
        """
        Calculate the time overhead of the user
        
        Parameters:
        data_rate : float
            Data rate of the user
        user_strategy : float
            User strategy that shows the percentage of data that the user will offload to the UAV
        """
        
        self.current_time_overhead = (self.data_in_bits * self.get_user_strategy()) / data_rate + ((phi* self.get_user_strategy() *self.data_in_bits) / ((1 - (jnp.sum(other_user_strategies * other_user_bits)/ uav_total_capacity)) * uav_cpu_frequency))
    
    def calculate_consumed_energy(self)->None:
        """
        Calculate the consumed energy of the user during the offloading process
        """
        self.current_consumed_energy = ((self.get_user_strategy() * self.get_user_bits())/self.calculate_data_rate) * self.transmit_power
        self.adjust_energy(self.current_consumed_energy)

    
    def calculate_total_overhead(self, T: float)->None:
        """
        Calculate the total overhead of the user during the offloading process over a period T
        
        Parameters:
        T : float
            Time that that timeslot t lasted
        """
        self.current_total_overhead = (self.current_time_overhead/T) + (self.current_consumed_energy/self.energy_level)
        
    def play_submodular_game(self, other_people_strategies: list, cost: float)->float:
        
        # Set as Variable the bits that the user will send to each MEC server
        percentage_offloaded = cp.Variable((1, 1), name = 'percentage_offloaded', nonneg=True)

        # Set the objective function and the constraints
        objective = cp.Maximize((self.get_user_bits() * jnp.exp(percentage_offloaded/jnp.sum(other_people_strategies))) - (cost*jnp.exp(self.get_current_total_overhead)))
        constraints = [percentage_offloaded >=0 , percentage_offloaded <= 1]

        # Create Problem
        prob = cp.Problem(objective, constraints)

        # Solve the problem and get the solution which is the maximum utility of the user
        solution = prob.solve(verbose=False)
        
        return (solution, percentage_offloaded.value)
        
    def __str__(self)->str:
        return f"User ID: {self.user_id}"