
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')

import gym
import numpy as np
from matplotlib import pyplot as plt


# In[2]:

env = gym.envs.make("Breakout-v0")


# In[3]:

print("Action space size: {}".format(env.action_space.n))
print(env.get_action_meanings())

observation = env.reset()
print("Observation space shape: {}".format(observation.shape))

plt.figure()
plt.imshow(env.render(mode='rgb_array'))

[env.step(2) for x in range(1)]
plt.figure()
plt.imshow(env.render(mode='rgb_array'))

env.render(close=True)
print(env.action_space)
A = np.array([0.1,0.1,0.1,0.1,0.1,0.5])
obs,_,_,_ = env.step(0)
print np.shape(obs)
# In[73]:

# Check out what a cropped image looks like
plt.imshow(observation[34:-16,:,:])

