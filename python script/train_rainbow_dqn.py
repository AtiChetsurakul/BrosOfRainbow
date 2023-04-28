# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace
# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT, RIGHT_ONLY
from wrappers import FrameStack, ResizeObservation, GrayScaleObservation, SkipFrame

import torch
import numpy as np
import random

from agent import Agent

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
force_right = ["right"], ["right", "A"]
env = JoypadSpace(env, force_right)

env = FrameStack(ResizeObservation(GrayScaleObservation(SkipFrame(env, skip=4)), shape=84), num_stack=4)

n_actions = env.action_space.n
print("Action :",n_actions)

seed = 777

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
random.seed(seed)
seed_torch(seed)
env.seed(seed)

# parameters
num_frames = 1
memory_size = 10
batch_size = 32
target_update = 4

# train
# already tried (6.25e-5, 6.25e-7, 6.25e-10)
agent = Agent(env, memory_size, batch_size, target_update, n_step=3, lr=6.25e-5, select_model='dqn')

if __name__ == 'main':
    agent.train_live(num_frames)