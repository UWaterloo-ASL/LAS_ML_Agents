"""
Created on 2019-02-05 11:58 AM

@author: jack.lingheng.meng

Reference:
    1. https://github.com/Unity-Technologies/ml-agents/blob/master/notebooks/getting-started.ipynb
    2. https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Python-API.md
"""

# 1. Load dependencies
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mlagents.envs import UnityEnvironment

# 2. Set environment parameters
# Detect Operating System and Choose the Unity environment binary to launch
if sys.platform == "linux" or sys.platform == "linux2":
    env_name = os.path.join('..', 'LAS_Simulator_Linux', 'LAS_Simulator')
elif sys.platform == "win32":
    env_name = os.path.join('..', 'LAS_Simulator_Windows', 'LAS_Simulator')
elif sys.platform == "darwin":
    env_name = os.path.join('..', 'LAS_Simulator_Mac', 'LAS_Simulator')

train_mode = True  # Whether to run the environment in training or inference mode

# 3. Start the environment
# env = UnityEnvironment(file_name=env_name, , seed=1)
env = UnityEnvironment(file_name=None, seed=1)

# Set the default brain to work with
default_brain = env.brain_names[0]
brain = env.brains[default_brain]

# 4. Examine the observation and state spaces
# Reset the environment
env_info = env.reset(train_mode=train_mode)[default_brain]

# Examine the state space for the default brain
print("Agent state looks like: \n{}".format(env_info.vector_observations[0]))

# Examine the observation space for the default brain
for observation in env_info.visual_observations:
    print("Agent observations look like:")
    if observation.shape[3] == 3:
        plt.imshow(observation[0,:,:,:])
    else:
        plt.imshow(observation[0,:,:,0])

# 5. Take random actions in the environment

for episode in range(100):
    env_info = env.reset(train_mode=train_mode)[default_brain]
    done = False
    episode_rewards = 0
    while not done:
        action_size = brain.vector_action_space_size
        if brain.vector_action_space_type == 'continuous':
            env_info = env.step(np.random.randn(len(env_info.agents),
                                                action_size[0]))[default_brain]
        else:
            action = np.column_stack([np.random.randint(0, action_size[i], size=(len(env_info.agents))) for i in range(len(action_size))])
            env_info = env.step(action)[default_brain]
        episode_rewards += env_info.rewards[0]
        done = env_info.local_done[0]
    print("Total reward of episode {}: {}".format(episode, episode_rewards))

# 6. Close the environment when finished
env.close()