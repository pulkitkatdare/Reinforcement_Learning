
# coding: utf-8

# In[97]:


import numpy as np
import gym
#######Function Definitions in this part##################
def softmax(x):
    """
    N = np.shape(x)[0]
    print N
    e_x = np.exp(b-np.transpose(np.tile(np.max(b,axis=1),(N,1))))
    e_ix =  np.transpose(np.tile(np.sum(e_x,axis=1),(N,1)))
    out = e_x/e_ix
    return out
    """
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out
def Initialise(input_size=4,hidden_size=10,output_size=2,weight_scale =0.1):
    W1 = weight_scale*np.random.randn(input_size,hidden_size)
    b1 = np.zeros(hidden_size)
    W2 = weight_scale*np.random.randn(hidden_size,output_size)
    b2 = np.zeros(output_size)
    return W1,b1,W2,b2
def loss(input,W1,b1,W2,b2,y=None,reg=0.0):
    input_max = np.max(input)
    input_min = np.min(input)
    input_dev =  np.std(input)
    input = 1*(input-input_min)/(input_max - input_min)
    out_1 = input.dot(W1) + b1
    diff_1 = np.ones(np.shape(out_1))
    diff_1m = out_1
    diff_1[out_1<0] = 0
    out_1[out_1 < 0] = 0
    out_2 = out_1.dot(W2) + b2
    e_ix = out_1
    #print np.shape(out_2)
    diff_2 = np.ones(np.shape(out_2))
    diff_2[out_2<0] = 0
    e_x = out_2
    #sum_2 = np.sum(e_x,axis=1)
    out_2 = softmax(out_2)
    #print np.shape(out_2)
    #print np.shape(out_2)
    if (y==None): 
        #a = np.argmax(out_2)[0]
        a = np.expand_dims(out_2, axis=0)
        #print out_2
        a =  np.argmax(a,axis=1)
        #a = 0
        return a
    else :
        diff_2 = np.ones(np.shape(out_2))
        diff_2[e_x<0] = 0
        sum_2 = np.sum(e_x,axis=1)
        n = np.shape(input)[0]
        #print np.shape(out_2)
        loss_n = -np.sum(np.log(out_2[range(n),y]))
        loss_n += 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)  
        output_size = np.shape(W2)[1]
        hidden_size = np.shape(W2)[0]
        grad_b2 = np.zeros((n,output_size))
        sum = 0
        #print np.shape(out_2)
        #print np.shape(grad_b2)
        #print np.shape(sum_2)
        for i in range(n):
            for j in range(output_size):
                if (j==y[i]):
                    #print i,j
                    grad_b2[i,j] = grad_b2[i,j] + (-1/out_2[i,j])*(out_2[i,j] - out_2[i,j]*(1/sum_2[i]))
                else : 
                    grad_b2[i,j] = grad_b2[i,j] + (-1/out_2[i,j])*(-out_2[i,j]*(1/sum_2[i]))
        grad_W2 = np.transpose(e_ix).dot(grad_b2)
        #print np.shape(e_ix)
        #print np.shape(grad_b2)
        grad_out2 = grad_b2.dot(np.transpose(W2))
        grad_b2 = np.sum(grad_b2,axis=0)
        #print np.shape(grad_b2)
        grad_b1 = np.zeros((n,hidden_size))
        grad_b1 = np.multiply(grad_out2,diff_1)
        #print np.shape(grad_b1)
        grad_W1 = np.transpose(input).dot(grad_b1)
        grad_b1 = np.sum(grad_b1,axis=0)
        grad_W2 += reg*W2
        grad_W1 += reg*W1 
        return loss_n,grad_W2,grad_b2,grad_W1,grad_b1
def run_episode(W1,W2,b1,b2,env,obs,no_iter=200,learning_rate = 0.00001):
    y = []
    input_data = []
    r = 0
    for i in range(no_iter):
        input_data.append(obs)
        action = loss(obs,W1,b1,W2,b2)
        y.append(action)
        obs, reward, done, info = env.step(action[0])
        r = r  + reward
        if (done): 
            break
    y = np.asarray(y)
    input_data = np.asarray(input_data)
    loss_n,grad_W2,grad_b2,grad_W1,grad_b1 = loss(input_data,W1,b1,W2,b2,y)
    W1 = W1 -learning_rate*grad_W1
    b1 = b1 - learning_rate*grad_b1
    W2 = W2 - learning_rate*grad_W2
    b2 = b2 - learning_rate*grad_b2
    return r,W1,b1,W2,b2
        
        
                         




env = gym.make('CartPole-v0')
obs = env.reset()
W1,b1,W2,b2 = Initialise()
learning_rate = 0.0001 
rate_decay = 0.999
no_iter = 200
for i in range(1000):
    if(i%100 ==0): 
        print i
        learning_rate = learning_rate*rate_decay
    reward,W1,b1,W2,b2 = run_episode(W1,W2,b1,b2,env,obs,no_iter,learning_rate)
    obs = env.reset()
print reward
    

