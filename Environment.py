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

    def run(self, agent):


