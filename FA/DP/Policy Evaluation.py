
import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS, dtype = float)
    delta = 0 
    while True:
        delta = 0 
        for i in range(env.nS):
            Val = 0 
            for j in range(env.nA):
                prob, next_state, reward, done = env.P[i][j][0]
                Val = Val + policy[i,j]*prob*(reward + discount_factor*V[next_state])
            diff = abs(V[i] - Val)
            V[i] = Val
            delta = max(delta,diff)
        if (delta < theta):
            break

        # TODO: Implement!
        #if ((V - V_previous).any() < theta):
        #    print "True"
        #    break
    return np.array(V)

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)
# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
