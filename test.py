from memory import ReplayMemory
from environment import DoomEnvironment
from agent import DQNAgent 
from time import time,sleep


if __name__ == '__main__':

    env = DoomEnvironment(config_file_path = "scenarios/basic.cfg", scenario_file_path = "scenarios/basic.wad",)
    actions = env.get_actions()

    resolution = (64, 64)
    memory = ReplayMemory(resolution = resolution)
    
    agent = DQNAgent(action_count = len(actions), replay_memory =  memory)

    agent.load_saved_agent("saved_models/RBDoom_DQN.pth")

    env.watch(agent)

