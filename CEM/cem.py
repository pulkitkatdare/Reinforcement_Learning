
# coding: utf-8
import gym 
import numpy as np 
import math 
import logging 
import argparse
#env = gym.make('CartPole-v0')
#obs = env.reset()

def initialise(num_samples,mean,var,mean_b,var_b):
	#Intialise weights for the cross entropy method
	W = np.transpose(np.array([np.random.normal(u, np.sqrt(o), num_samples) for u, o in zip(mean, var)]))
	b = np.transpose(np.array([np.random.normal(mean_b, np.sqrt(var_b), num_samples)]))
	return W,b
def action(obs,W,b):
	y = W.dot(obs) + b[:]
	return y
def do_rollout(env,obs,W,b,y,num_time_steps,top_frac,num_samples,render=False):
	top_n = round(num_samples*top_frac)
	expected_reward = np.zeros(num_samples)
	for i in range(num_samples):
		W_iter = W[i]
		b_iter = b[i]
		obs = env.reset()
		for j in range(num_time_steps):
			y = action(obs,W_iter,b_iter)
			a = int(y< 0)
			obs, reward, done, info = env.step(a)
			expected_reward[i] = expected_reward[i] + reward
			if render and (j%3==0): env.render()
			if(done): break
	arg_sort = np.argsort(expected_reward)[::-1]
	arg_topn = arg_sort[0:int(top_n)]
	W_topn = W[arg_topn]#,axis=0)
	b_topn = b[arg_topn]
	mean_W  = np.mean(W_topn,axis=0)
	mean_b  = np.mean(b_topn)
	var_W   = np.var(W_topn,axis=0)
	var_b   = np.var(b_topn)
	return mean_W,mean_b,var_W,var_b,expected_reward   
	
	


# In[5]:

#initialising the gym enviornment 
env = gym.make('CartPole-v0')
obs = env.reset()
#env.monitor.start('/tmp/cartpole-experiment-1')
#intialising the parameters 
time_steps = 200
num_samples = 100
top_frac = 0.2
num_iters = 100


# In[6]:

#testing whether the envornment works
num_rows = np.shape(obs)[0]
mean = np.zeros(num_rows)
var  = np.ones(num_rows)
mean_b = 0
var_b  = 1
v = 0
for i in range(num_iters):
	print i
	W,b = initialise(num_samples,mean,var+v,mean_b,var_b+v)
	a   = action(obs,W,b)
	v = max(5 - (i/10), 0)
	mean,mean_b,var,var_b,exp_r = do_rollout(env,obs,W,b,a,time_steps,top_frac,num_samples)
	assert np.shape(mean)[0] == np.shape(var)[0] 
print exp_r
env.render(close=True)


# In[7]:

print mean

env = gym.make('CartPole-v0')

obs = env.reset()
v = 0 
W,b = initialise(1,mean,var,mean_b,var_b)
a   = action(obs,W,b)
a = int(a<0)
obs, reward, done, info = env.step(a)
t = 0
rewards = np.zeros(100)
for i_episode in range(100):
	obs = env.reset()
	W,b = initialise(1,mean,var,mean_b,var_b)
	a   = action(obs,W,b)
	a = int(a<0)
	obs, reward, done, info = env.step(a)
	while(done==False):
		t = t + 1 
		a   = action(obs,W,b)
		a = int(a<0)
		env.render()
		obs, reward, done, info = env.step(a)
		rewards[i_episode] += reward
		v = 0
		env.render()
	print t
print rewards
env.close()



