import numpy as np
from scipy import special

def relu(x):

    x[x < 0] = 0

    return x

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))


def gelu(x):
    return 0.5 * x * (1 + special.erf(x / np.sqrt(2)))


#x = np.random.uniform(-1, 1, size=(5, 5))
#x = np.ones((5, 5))
#
#print(x)
#
#x = sigmoid(x)
#
#print(x)
