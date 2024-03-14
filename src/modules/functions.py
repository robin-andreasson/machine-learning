import numpy as np
import math
# file containing functions that are needed in all of them 


# fully connected layer, applies a linear transformation on input x through matrix w
class linear():
    def __init__(self, in_features, out_features):

        self.k = math.sqrt(1. / in_features)

        self.w = np.random.uniform(low=-self.k, high=self.k, size=(in_features, out_features))
        self.b = np.random.uniform(low=-self.k, high=self.k, size=out_features)

    
    def __call__(self, x):
        # apply transformation and add bias for numeric stability but also to shift the curve as needed
        x = np.matmul(x, self.w) + self.b
         
        return x
 

# we normalize each feature vector in x, preventing several issues but notably things like vanishing gradients
class layer_normalization():
    def __init__(self, features):
        self.scale = np.ones(features)
        self.shift = np.zeros(features)


    def __call__(self, x):

        # numeric stability
        e = 1e-5

        # get the mean and the difference between each feature and the mean
        # we also reshape the output vector from size n to a matrix of size (n, 1) because of broadcasting issues
        m = x.mean(axis=1)[:,None]
        d = x - m

        # get variance
        v = np.mean(d ** 2, axis=1)[:,None] + e

        # normalize
        x = d / np.sqrt(v)
        
        # apply learnable parameters scale and shift
        x = self.scale * x + self.shift

        return x


class batch_normalization():
    def __init__(self):
        print("init")

    def __call__(self, x):
        print(x)

# we randomly 'drop' neurons, removing its impact in future operations. It forces the network to generalize better and be more robust since it 
# can't rely on specific connections or neurons for the output, mostly used if you want to prevent overfitting
class dropout():
    def __init__(self, rate):
        self.p_rate = rate

    def __call__(self, x): 
        # create a mask of probabilities for each feature in x, if probability passes the p_rate threshold, then we can drop the neuron
        drop_mask = np.random.rand(*x.shape)
        
        # conditional checking for dropping out any neurons
        x[ drop_mask < self.p_rate ] = 0

        return x


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
