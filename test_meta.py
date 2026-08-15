import time
import gymnasium as gym 
import numpy as np
from gym_robotics_custom import RoboticsObservationWrapper
from buffer import ReplayBuffer
from model import *
from agent import *
from meta_agent import *

if __name__ == '__main__':

    env_name = 'FrankaKitchen-v1'
    max_episodes_steps = 1200
    replay_buffer_size = 1000000
    tasks = ['top burner', 'microwave', 'hinge cabinet', 'slide cabinet']
    gamma = 0.99
    tau = 0.005
    alpha = 0.1
    target_update_interval = 1
    updates_per_step = 4
    hidden_size = 512
    learning_rate = 0.0001
    batch_size = 64
    live_test = True
    generate_score = True
    
    if live_test:
        env = gym.make(env_name, max_episode_steps=max_episodes_steps, tasks_to_complete=tasks, render_mode='human')
        env = RoboticsObservationWrapper(env)

        meta_agent = MetaAgent(env, tasks, max_episode_steps=max_episodes_steps)

        meta_agent.initilise_agents()

        meta_agent.test()

        env.close()


    if generate_score:
        print(f"Generating performance score")
        env = gym.make(env_name, max_episode_steps=max_episodes_steps, tasks_to_complete=tasks)
        env = RoboticsObservationWrapper(env)

        observation, info = env.reset()

        observation_size = observation.shape[0]

        meta_agent = MetaAgent(env,tasks, max_episode_steps=max_episodes_steps)

        meta_agent.initilise_agents()

        perf_score_epochs = 10

        total_score = 0

        for i in range(perf_score_epochs):
            score = meta_agent.test()
            total_score += score

        scuccsess_ratio = ((total_score / len(tasks)) / perf_score_epochs) * 100

        print(f"Success ratio {scuccsess_ratio:.2f}%")