# agents.py

import gymnasium as gym
import numpy as np

class Agent:
    def __init__(
        self,
        environment: gym.Env,
        
        # instead of providing agents with a learning algorithm
        # we will subclass specific agents that implement an algorithm
        
        ):

        self.environment = environment
        self.policy = np.array(environment.action_space.shape)

    def pick_action(self, observation):
        pass

    def update_policy(self):
        pass 

    def random_action(self):
        return self.environment.action_space.sample()

class RandomAgent(Agent):

    def pick_action(self, observation):
        return super().random_action()
    

    

class QLearningAgent(Agent):
    
    def __init__(
            self, 
            environment: gym.Env,
            epsilon, 
            ):
        super().__init__(environment)
        
        action_space_dimension = environment.action_space.n
        observation_space_dimension = environment.observation_space.shape
        Q_shape = observation_space_dimension + (action_space_dimension,)
        self.Q_values = np.zeros(Q_shape)
    
    def pick_action(self, observation):
        # epsilon-greedy policy
        if np.random.random() < self.epsilon:
            return super().random_action()
        else:
            return self.getBestAction(observation)
    
    def getBestAction(self, observation):
        # returns the action with the highest Q-value 
        return np.argmax(self.Q_values[observation, :])
    
