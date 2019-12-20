import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  f = X@W
  correct_class = np.reshape(np.max(f,axis=1),(X.shape[0],1))
  prob = np.exp(f-correct_class)/np.sum(np.exp(f-correct_class),axis=1,keepdims=True)

  for i in range(X.shape[0]):
    loss -= np.log(prob[i,y[i]])
    for j in range(W.shape[1]):
      if(y[i]==j):
        dW[:,j] += X[i,:].T*(prob[i,j]-1)
      else:
        dW[:,j] += X[i,:].T*prob[i,j]
  loss /= X.shape[0]
  loss+=0.5*reg*reg*np.sum(W*W)
  dW = dW/X.shape[0] +reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)



  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X@W
  correct_class = np.reshape(np.max(f,axis=1),(X.shape[0],1))
  prob = np.exp(f-correct_class)/np.sum(np.exp(f-correct_class),axis=1,keepdims=True)
  loss = np.sum(-1*np.log(prob[range(X.shape[0]),y]))
  prob[range(X.shape[0]),y] -=1
  dW = X.T.dot(prob)
                                 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss/=X.shape[0]
  dW/=X.shape[0]
  loss+=0.5*reg*reg*np.sum(W*W)
  dW += reg*W
  return loss, dW

