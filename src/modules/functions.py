import numpy as np

def softmax(x):

    ex = np.exp(x)
    sum = np.sum(ex, axis=-1)

    return ex / np.expand_dims(sum, axis=-1)


# fully connected layer, applies a linear transformation on input x through matrix w
class linear():
    def __init__(self, out_rows, out_cols):

        self.k = np.sqrt(1. / out_rows)

        self.w = np.random.uniform(low=-self.k, high=self.k, size=(out_rows, out_cols))
        self.b = np.random.uniform(low=-self.k, high=self.k, size=out_cols)

    
    def __call__(self, x):
        # apply transformation and add bias to shift the curve as needed (also numeric stability)
        x = np.matmul(x, self.w) + self.b
         
        return x
 

# we normalize each feature vector in x
class layer_normalization():
    def __init__(self, features):
        self.scale = np.ones(features)
        self.shift = np.zeros(features)


    def __call__(self, x):

        # numeric stability
        e = 1e-5

        # get the mean and the difference between each feature
        # we also reshape the output vector from size n to a matrix of size (n, 1) for broadcasting compatibility
        m = np.expand_dims(x.mean(axis=-1), -1)
        d = x - m

        # get variance
        v = np.expand_dims(np.mean(d ** 2, axis=-1), -1) + e

        # normalize
        x = d / np.sqrt(v)
        
        # apply learnable scale and shift
        x = self.scale * x + self.shift

        return x


class batch_normalization():
    def __init__(self):
        print("init")

    def __call__(self, x):
        print(x)

# we randomly 'drop' neurons, removing its impact in the current iteration. 
# it tries to force the network to be more robust and generalize better since it can't rely on specific connections / nodes for the output
class dropout():
    def __init__(self, rate):
        self.p_rate = rate

    def __call__(self, x): 
        # create a mask of probabilities for each feature in x, if probability passes the p_rate threshold, then we can drop the neuron
        mask = np.random.rand(*x.shape)
        
        # conditional checking for dropping out any neurons
        x[ mask < self.p_rate ] = 0

        return x


class attention():
    def __init__(self, heads, model_size, model_head_size, mask, drop, p_rate):

        # hyperparameters
        self.heads = heads
        self.model_size = model_size
        self.model_head_size = model_head_size

        # extra functionality
        self.mask = mask
        self.drop = drop

        # classes
        self.dropout = dropout(p_rate)
        self.fc = linear(model_size, model_size)
        self.q_projection = linear(model_size, model_size)
        self.k_projection = linear(model_size, model_size)
        self.v_projection = linear(model_size, model_size)

    def multi_head_attention(self, q, k, v):

        # call learned q, k and v projection respectively, reshape to get each head and then swap head and sequence axes
        q = self.q_projection(q).reshape((q.shape[0], q.shape[1], self.heads, self.model_head_size)).swapaxes(1, 2)
        k = self.k_projection(k).reshape((k.shape[0], k.shape[1], self.heads, self.model_head_size)).swapaxes(1, 2)
        v = self.v_projection(v).reshape((v.shape[0], v.shape[1], self.heads, self.model_head_size)).swapaxes(1, 2)

        # get contextualized sequence
        v = self.scaled_dot_product(q, k, v)

        # swap back head and sequence axes, then concat each feature node
        v = np.reshape(v.swapaxes(1, 2), newshape=(v.shape[0], v.shape[2], self.model_size))

        v = self.fc(v)

        return v


    def scaled_dot_product(self, q, k, v):

        # we transpose the last two axes of K, making each key be column based instead of row based for matrix multiplication compatibility
        x = np.matmul(q, k.swapaxes(-1, -2)) / np.sqrt(self.model_head_size)

        if self.mask:
            # mask out products where the associated key would be regarded as subsequent in respect to the current query,
            # making each new output 'word' only be a product of keys that are found at and before itself
            i = np.triu_indices(x.shape[-1], k=1)
            x[:, :, *i] = -np.inf


        # weigh each product
        x = softmax(x)

        if self.drop:
            x = self.dropout(x)

        # transform the attention scores/weights with the value tensor
        x = np.matmul(x, v)

        return x
        

def positional_encoding(n, context_size, model_size):

    encoding_mask = np.zeros((context_size, model_size))

    for row, col in np.ndindex((context_size, model_size // 2)):

        position_representation = row / np.power(n, 2 * col / model_size)

        encoding_mask[row, 2*col] = np.sin(position_representation)
        encoding_mask[row, 2*col + 1] = np.cos(position_representation)

    return encoding_mask




#input = np.array([
#    [
#        [0.3, 0.1, 0.2],
#        [0.4, 0.5, 0.01],
#        [0.9, 0.25, 0.7]
#    ],
#    [
#        [0.3, 0.1, 0.2],
#        [0.4, 0.5, 0.01],
#        [0.9, 0.25, 0.7]
#    ]
#])
#
#w = np.array([
#    [
#        [3.67643248], 
#        [4.15059614], 
#        [5.75738124],
#    ],
#    [
#        [3.67643248], 
#        [4.15059614], 
#        [5.75738124],
#    ]
#])

#input = np.array([
#    [
#        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
#        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
#        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
#    ],
#    [
#        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
#        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
#        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
#    ],
#])

#input = np.array([
#    [
#        [0.2, 0.1, 0.3, 0.2],
#        [0.5, 0.1, 0.1, 0.3],
#    ],
#    [
#        [0.9, 0.8, 0.7, 0.4],
#        [0.3, 0.6, 0.5, 0.5],
#    ],
#])
#
#
#np.set_printoptions(suppress=True, precision=8)
#mask = positional_encoding(n=100, context_size=2, model_size=4)
#mask = mask(input)
#
#print(mask)
#print("\n")
#print(input)
#print("\n")
#print(mask + input)
#print("\n")

#print(softmax(x))

#print(input.reshape(2, 3, 3, 9))
#print("\n")
#print(input.reshape(2, 3, 3, 9).swapaxes(1, 2))
#print("\n")
#
#att = attention(9, 27, 3,  mask=True, drop=True, p_rate=0.1)
#
#y = att.multi_head_attention(q=input, k=input, v=input)
#
#print(y)


#a = np.arange(18).reshape(2, 3, 3)
#indices = np.triu_indices(3, k=1)
#
#a[:, *indices] = -1
#
#
#print(a)
