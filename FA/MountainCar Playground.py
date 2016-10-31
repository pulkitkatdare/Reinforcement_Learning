
# coding: utf-8

# In[7]:

#get_ipython().magic(u'matplotlib inline')

import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
  sys.path.append("../") 

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import RBFSampler


# In[8]:

env = gym.envs.make("MountainCar-v0")
print("Action space size: {}".format(env.action_space.n))

# In[11]:
scaler = sklearn.preprocessing.StandardScaler()
#scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=1)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=1)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=1)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=1))
        ])
#featurizer.fit(scaler.transform(observation_examples))

state =  env.reset()
scaled = scaler.transform([state])
featurized = featurizer.transform(scaled)
print featurized

plt.figure()
plt.imshow(env.render(mode='rgb_array'))

[env.step(0) for x in range(10000)]
plt.figure()
plt.imshow(env.render(mode='rgb_array'))

env.render(close=True)


# In[6]:



