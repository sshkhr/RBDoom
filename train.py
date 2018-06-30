import argparse
from datetime import datetime
import random
import torch

from memory import ReplayMemory
from environment import DoomEnvironment
from agent import DQNAgent 
from time import time,sleep

parser = argparse.ArgumentParser(description='RBDoom')


if __name__ == '__main__':

    env = DoomEnvironment(config_file_path = "scenarios/basic.cfg", scenario_file_path = "scenarios/basic.wad",
    	                  resolution = (64,64), stack_size = 4)
    actions = env.get_actions()

    resolution = (64, 64)
    memory = ReplayMemory(resolution = resolution, stack_size = 4)
    
    agent = DQNAgent(action_count = len(actions), replay_memory =  memory)

    print("Starting the training!")
    agent.train(env)

    env.game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    env.watch(agent)

