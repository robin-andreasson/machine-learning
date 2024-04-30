import numpy as np

# contains different weight and bias initializers

def default(fan_in, fan_out):
    k = np.sqrt(1. / fan_in)

    w = np.random.uniform(low=-k, high=k, size=(fan_in, fan_out))
    b = np.random.uniform(low=-k, high=k, size=fan_out)

    return w, b

# for xavier initializers, gain should change depending on activation: hyperbolic tangent = 1, sigmoid = 4 and relu = sqrt(2)
def xavier_uniform(fan_in, fan_out, gain = 1):
    k = gain * np.sqrt(6. / (fan_in + fan_out))

    w = np.random.uniform(low=-k, high=k, size=(fan_in, fan_out))
    b = np.zeros(fan_out)

    return w, b

def xavier_normal(fan_in, fan_out, gain = 1):
    std = gain * np.sqrt(2. / (fan_in + fan_out))

    w = np.random.normal(loc=0, scale=std, size=(fan_in, fan_out))
    b = np.zeros(fan_out)

    return w, b

def he_uniform(fan_in, fan_out):
    low_k = np.sqrt(6 / fan_in)
    high_k = np.sqrt(6 / fan_out)

    w = np.random.uniform(-low_k, high_k, size=(fan_in, fan_out))
    b = np.zeros(fan_out)

    return w, b

def he_normal(fan_in, fan_out):
    std = np.sqrt(2 / fan_in)

    w = np.random.normal(loc=0, scale=std, size=(fan_in, fan_out))
    b = np.zeros(fan_out)

    return w, b

def uniform(fan_in, fan_out, low, high):
    w = np.random.uniform(low, high, size=(fan_in, fan_out))
    b = np.random.uniform(low, high, size=fan_out)

    return w, b

def normal(fan_in, fan_out, mean, std):
    w = np.random.normal(loc=mean, scale=std, size=(fan_in, fan_out))
    b = np.random.normal(loc=mean, scale=std, size=fan_out)

    return w, b

def ones(fan_in, fan_out):
    w = np.ones(shape=(fan_in, fan_out))
    b = np.ones(fan_out)

    return w, b

def zeros(fan_in, fan_out):
    w = np.zeros(shape=(fan_in, fan_out))
    b = np.zeros(fan_out)

    return w, b
