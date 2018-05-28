from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep

# Creates and initializes ViZDoom environment
class DoomEnvironment:

    def __init__(self, config_file_path, show_window = False, mode = Mode.PLAYER, screen_resolution = ScreenResolution.RES_640X480):
        self.game = DoomGame()
        self.game.load_config(config_file_path)
        self.initialize_environment(show_window, mode, screen_resolution)

    def initialize_environment(self, show_window, mode, screen_resolution):
        self.game.set_window_visible(show_window)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_format(screen_format)
        self.game.set_screen_resolution(screen_resolution)
        self.game.init()
        print("DOOM Game Initialized")

    def preprocess(image, resolution = (64, 64)):
        img = skimage.transform.resize(image, resolution)
        img = img.astype(np.float32)
        return img


    def run(self, agent, episodes_to_run = 100):


    def watch(self, agent, episodes_to_watch = 10, frame_repeat = 12):
        self.game.set_window_visible(True)
        self.game.set_mode(Mode.ASYNC_PLAYER)
        self.game.init()

        for episode_num in range(episodes_to_watch):
            game.new_episode()

            while not self.game.is_episode_finished():
                
                state = preprocess(self.game.get_state().screen_buffer)
                state = state.reshape([1, 1, resolution[0], resolution[1]])
                best_action_index = agent.get_best_action(state)

                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                self.game.set_action(actions[best_action_index])
                for _ in range(frame_repeat):
                    self.game.advance_action()

            # Sleep between episodes
            sleep(1.0)
            score = game.get_total_reward()
            print("Episode Number:", episode_num,"Total score:", score)


