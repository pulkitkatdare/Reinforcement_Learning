
# coding: utf-8

# In[3]:

import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv


# In[4]:

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


# In[5]:

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI environment. env.P represents the transition probabilities of the environment.
        theta: Stopping threshold. If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
    

    V = np.zeros(env.nS,dtype = float)
    policy = np.zeros([env.nS, env.nA],dtype=float)
    delta = 1;
    prev_V = V;
    while (delta > theta):
        prev_V = np.array(V,dtype=float) 
        print delta
        for s in range(env.nS):
            val = np.zeros(env.nA)
            for a in range(env.nA):
                v = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    v += prob*(reward + discount_factor*V[next_state])
                val[a] = v
            V[s] = max(val);
            policy[s,np.argmax(val)] = 1
        delta = max(abs(V - prev_V))    
    # Implement!
    return policy, V


# In[6]:

policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")


# In[7]:

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

