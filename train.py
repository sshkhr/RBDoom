from memory import ReplayMemory
from environment import DoomEnvironment
from agent import DQNAgent 
from time import time,sleep


if __name__ == '__main__':

    env = DoomEnvironment("scenarios/basic.cfg")
    actions = env.get_actions()

    resolution = (64, 64)
    memory = ReplayMemory(resolution = resolution)
    
    agent = DQNAgent(action_count = len(actions), replay_memory =  memory)

    print("Starting the training!")
    agent.train(env)

    env.game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    env.watch(agent)

