import numpy as np
from modules.functions import linear, layer_normalization, dropout, softmax, attention, positional_encoding
from modules.activations import gelu

class gpt():
    def __init__(self, layers, heads, model_size, vocab_size, context_size):

        self.layers = layers

        self.dropout = dropout(rate=0.1)

        self.encodings = positional_encoding(n=100, context_size=context_size, model_size=model_size)

        # layer normalization
        self.normalization1 = np.full(layers, layer_normalization(model_size))
        self.normalization2 = np.full(layers, layer_normalization(model_size))
        self.normalization3 = layer_normalization(model_size)

        # attention sub-layer
        self.self_attention = np.full(layers, attention(heads, model_size, model_size // heads, mask=True, drop=True, p_rate=0.1))

        # position-wise feed forward network
        self.ffn1 = np.full(layers, linear(model_size, model_size * 4))
        self.ffn2 = np.full(layers, linear(model_size * 4, model_size))

        # output projection
        self.fc = linear(model_size, vocab_size)





    def __call__(self, x):

        x = x + self.encodings

        x = self.dropout(x)

        for layer in range(self.layers):

            # save current state for future preservation of identity
            residual = x

            # normalize our input by each feature vector
            x = self.normalization1[layer](x)

            # apply masked self attention and get contextualized sequence
            x = self.self_attention[layer].multi_head_attention(q=x, k=x, v=x)

            # apply dropout and then add the residual connection, dropped features will lose the newly acquired representation of previous sub-layer
            x = self.dropout(x) + residual

            x = self.normalization2[layer](x)

            residual = x

            # position-wise feed forward, we transform the network to a larger dimension (larger representation) 
            # Apply gelu for non-linearity and transform the network back to its original dimensionality
            x = self.ffn1[layer](x)
            x = gelu(x)
            x = self.ffn2[layer](x)

            x = self.dropout(x) + residual

        
        # normalize our input one last time
        x = self.normalization3(x)

        # project the network to vocabulary size and apply softmax, we then get a probability distribution of our vocabulary for each row
        x = softmax(self.fc(x))

        return x



np.set_printoptions(suppress=True, precision=8)

model = gpt(layers=12, model_size=512, heads=8, vocab_size=10000, context_size=1024)
input = np.random.rand(2, 1024, 512)

output = model(input)

print(output)
