import time
import gymnasium as gym 
import numpy as np
from gym_robotics_custom import RoboticsObservationWrapper
from buffer import ReplayBuffer
from model import *
from agent import *

if __name__ == '__main__':

    env_name = 'FrankaKitchen-v1'
    max_episodes_steps = 500
    replay_buffer_size = 1000000
    tasks = ['microva']
    task_no_spaces = task.replace(" ", "_")
    gamma = 0.99
    tau = 0.005
    alpha = 0.1
    target_update_interval = 1
    updates_per_step = 4
    hidden_size = 512
    learning_rate = 0.0001
    batch_size = 64
    

    env = gym.make(env_name, max_episode_steps=max_episodes_steps, tasks_to_complete=[task], render_mode='human')
    env = RoboticsObservationWrapper(env, goal=task)