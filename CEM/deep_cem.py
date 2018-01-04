
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import gym 
import numpy as np 
import math 
import logging 
import argparse

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size=4, hidden_size=100, output_size=2, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    H, O = W2.shape
    D, H = W1.shape
    N, D = X.shape
    prod_hidden = X.dot(W1)
    b_hidden = np.tile(b1,(N,1))
    sum_hidden = (prod_hidden + b_hidden) 
    exp_hidden = np.exp(-sum_hidden)
    add_exp_hidden = np.ones((N,H)) + exp_hidden
    #out_hidden = np.divide(np.ones((N,H)),add_exp_hidden)
    new_sum_hidden = sum_hidden
    new_sum_hidden[new_sum_hidden <0] = 0
    out_hidden = new_sum_hidden
    prod_out = out_hidden.dot(W2)
    b_out = np.tile(b2,(N,1))
    sum_out = (b_out + prod_out)
    #exp_out = np.exp(-sum_out) 
    #add_exp_out = np.ones((N,O)) + exp_out
    #output = np.divide(np.ones((N,O)),add_exp_out)
    ###########################
    exp_f = np.exp(sum_out)
    sum_f = np.sum(exp_f,axis =1)
    sum_f = np.tile(np.transpose(sum_f),(O,1))
    sum_f = np.transpose(sum_f)
    sum_f = np.divide(exp_f,sum_f)
    #n = range(N)
    #n = np.array(np.transpose(n))
    #true_f = sum_f[n,y[n]]
    #true_log_f = -np.log(true_f)
    #loss  = np.sum(true_log_f)
    #loss /= N
    #loss += 0.5*reg*(np.sum(W1*W1) + np.sum(b1*b1) + np.sum(b2*b2) + np.sum(W2*W2))
    ###########################
    scores = sum_out
    #: check points 
    #print np.shape(prod_hidden)
    #scores = (b_hidden + hidden_prod).dot(W2) + b_out
    # Compute the forward pass
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    n = range(N)
    n = np.array(np.transpose(n))
    true_f = sum_f[n,y[n]]
    true_log_f = -np.log(true_f)
    loss  = np.sum(true_log_f)
    loss /= N
    loss += 0.5*reg*(np.sum(W1*W1) + np.sum(b1*b1) + np.sum(b2*b2) + np.sum(W2*W2))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    dW2 = np.zeros((H,O))
    db2 = np.zeros(O)
    for i in range(N):
        for j in range(O):
            if (j == y[i]):
               dW2[:,j]=dW2[:,j] - np.transpose(out_hidden[i])
               db2[j] = db2[j] -1 
            dW2[:,j] = dW2[:,j] + sum_f[i,j]*np.transpose(out_hidden[i])
            db2[j] = db2[j] + sum_f[i,j]
    dW2 /= N
    db2 /= N
    dW2 = dW2 + reg*W2
    db2 = db2 + reg*b2
    grads['W2'] = dW2
    grads['b2'] = db2
    dW1 = np.zeros((D,H))
    db1 = np.zeros(H)
    for i in range(N):
         for j in range(H):
             grad1 = -W2[j,y[i]]
             grad2 = 0	
             for k in range(O):
                 grad2 = sum_f[i,k]*W2[j,k] + grad2
                 if (new_sum_hidden[i,j] != 0):
                     dW1[:,j] = dW1[:,j] + (grad1 + grad2)*np.transpose(X[i])
                     db1[j]   = db1[j] + (grad1+grad2)
    dW1 /= N
    db1 /= N
    dW1 = dW1 + reg*W1
    db1 = db1 + reg*b1
    grads['W1'] = dW1
    grads['b1'] = db1
    #for i in range(N):
    #	for j in range(H):
    #	    for k in range(O):
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    scores = self.loss(X)

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = np.zeros(X.shape[0])	
    #pred = np.dot(X,self.W)
    y_pred = self.loss(X)
    y_pred = np.argmax(y_pred,axis=1) 
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred




# In[ ]:


def do_rollout(env,obs,out,num_time_steps,top_frac,render=False):
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
    arg_topn = arg_sort[0:top_n]
    W_topn = W[arg_topn]#,axis=0)
    b_topn = b[arg_topn]
    mean_W  = np.mean(W_topn,axis=0)
    mean_b  = np.mean(b_topn)
    var_W   = np.var(W_topn,axis=0)
    var_b   = np.var(b_topn)
    return mean_W,mean_b,var_W,var_b,expected_reward   
    
    

