import gymnasium as gym
from Graph import Graph
from Uav import Uav
from Node import Node
from typing import Any, Tuple
import logging

class Qenv(gym.Env):
    """
    Environment for Q-learning algorithm.
    Includes the Graph and the Uav interaction with the Nodes.
    Also includes the submodular game that the users play within the nodes.
    """
    
    def __init__(self, graph: Graph, uav: Uav, number_of_users: list, convergence_threshold: float, n_actions: float,
                n_observations: float, solving_method: str) -> None:
        """
        Initialize the environment.
        
        Parameters:
            graph (Graph): The graph of the environment.
            uav (Uav): The UAV in the environment.
            number_of_users (list): The number of users in each node.
            convergence_threshold (float): The threshold for convergence.
            n_actions (float): The number of actions.
            n_observations (float): The number of observations.
            solving_method (str): The method to use for solving the game.
        """
        self.graph = graph
        self.uav = uav
        self.number_of_users = number_of_users
        self.convergence_threshold = convergence_threshold
        self.n_actions = n_actions
        self.n_observations = n_observations
        self.solving_method = solving_method
        
        self.action_space = gym.spaces.Discrete(n_actions)
        self.observation_space = gym.spaces.Discrete(n_observations)
        
        self.reset()
        logging.info("Environment has been successfully initialized!")
    
    def step(self, action: Node) -> Tuple[Node, float, bool, dict]:
        """
        Perform an action in the environment and calculate the reward.
        
        Parameters:
            action (int): The action to perform.

        Returns:
            observation (Node): The new observation.
            reward (float): The reward for the action.
            done (bool): Whether the episode is done.
            info (dict): Additional information.
        """

        done = False
        info = {}
        temp_reward = 0
        # Perform the action
        self.uav.travel_to_node(action)
        
        # Check if the UAV has reached the final node to end the episode
        if (self.uav.get_current_node() == self.uav.get_final_node()):
            done = True
        
        
        self.reward += temp_reward
        
        return (self.observation, self.reward, done, info)
    
    def reset(self, **kwargs: Any) -> Node:
        """
        Reset the environment to the initial state.
        """
        self.observation = self.uav.get_initial_node()
        self.reward = 0
        
        return self.observation