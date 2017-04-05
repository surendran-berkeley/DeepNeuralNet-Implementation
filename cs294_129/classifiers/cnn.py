import numpy as np

from cs294_129.layers import *
from cs294_129.fast_layers import *
from cs294_129.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-1, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    #self.filter_size=filter_size
    C, H, W = input_dim
    stride= 1 # Assuming based on the facts provided in loss
    pad= (filter_size-1)/2
    pool= 2
    W_c= 1+(W+ 2*pad - filter_size)/stride
    H_c= 1+(H + 2*pad - filter_size)/stride
    W_p= 1+(W_c - pool)/2
    H_p= 1+(H_c - pool)/2
    self.params["W1"]=np.random.normal(scale= weight_scale, size= (num_filters, C, filter_size, filter_size))
    self.params["b1"]= np.zeros(num_filters)
    self.params["W2"] = np.random.normal(scale = weight_scale, size = (num_filters*W_p*H_p,hidden_dim ))
    self.params["b2"] = np.zeros(hidden_dim)
    self.params["W3"]= np.random.normal (scale = weight_scale, size=(hidden_dim, num_classes) )
    self.params ["b3"] = np.zeros(num_classes)
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    reg= self.reg
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_out, conv_cache= conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    affine_1_out, affine_1_cache=  affine_relu_forward(conv_out, W2, b2)
    affine_2_out, affine_2_cache = affine_forward(affine_1_out, W3, b3)
    scores= affine_2_out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    soft_loss, dscores = softmax_loss(scores, y)
    reg_1= np.sum(0.5*reg*(W1))
    reg_2= np.sum(0.5*reg*(W2**2))
    reg_3=np.sum(0.5*reg*(W3**2))
    regularization= reg_1+reg_2+reg_3
    loss= soft_loss+regularization

    ## Time for grads to graduate ... lol
    dx_3, grads["W3"], grads["b3"]= affine_backward(dscores, affine_2_cache)
    dx_2, grads["W2"], grads["b2"] = affine_relu_backward(dx_3, affine_1_cache)
    dx_1, grads["W1"], grads["b1"] = conv_relu_pool_backward (dx_2, conv_cache)

    grads["W3"]+= reg*W3
    grads["W2"]+= reg*W2
    grads["W1"]+= reg*W1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
