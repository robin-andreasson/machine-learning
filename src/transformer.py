import numpy as np
from modules.functions import linear, layer_normalization, dropout, softmax, attention
from modules.activations import gelu


class encoder():
    def __init__(self, layers, heads, model_size):
        self.layers = layers

        # layer normalization
        self.normalization1 = np.full(layers, layer_normalization(model_size))
        self.normalization2 = np.full(layers, layer_normalization(model_size))

        # position-wise feed forward
        self.ffn1 = np.full(layers, linear(model_size, model_size * 4))
        self.ffn2 = np.full(layers, linear(model_size * 4, model_size))

        # attention sub-layer
        self.self_attention = np.full(layers, attention(heads, model_size, model_size // heads, mask=False, drop=True, p_rate=0.1))

    def __call__(self, x):

        # embeddings

        for layer in range(self.layers):
            # positional encoding (only on new tokens?)
    
            residual = x

            x = self.self_attention[layer].multi_head_attention(q=x, k=x, v=x)

            x = self.normalization1[layer](x + residual)

            residual = x

            x = self.ffn1[layer](x)
            x = gelu(x)
            x = self.ffn2[layer](x)

            x = self.normalization2[layer](x + residual)


        return x 


class decoder():
    def __init__(self, layers, heads, model_size, vocab_size):

        self.layers = layers

        # layer normalization
        self.normalization1 = np.full(layers, layer_normalization(model_size))
        self.normalization2 = np.full(layers, layer_normalization(model_size))
        self.normalization3 = np.full(layers, layer_normalization(model_size))

        # position-wise feed forward
        self.ffn1 = np.full(layers, linear(model_size, model_size * 4))
        self.ffn2 = np.full(layers, linear(model_size * 4, model_size))

        # output projection
        self.fc = linear(model_size, vocab_size)

        # attention sub-layers
        self.masked_self_attention = np.full(layers, attention(heads, model_size, model_size // heads, mask=True, drop=True, p_rate=0.1))
        self.encoder_decoder_attention = np.full(layers, attention(heads, model_size, model_size // heads, mask=False, drop=True, p_rate=0.1))


    def __call__(self, x, ex):

        # embeddings

        for layer in range(self.layers):
            # positional encoding (only on new tokens?)

            residual = x

            x = self.masked_self_attention[layer].multi_head_attention(q=x, k=x, v=x)

            x = self.normalization1[layer](x + residual)

            residual = x

            x = self.encoder_decoder_attention[layer].multi_head_attention(q=ex, k=ex, v=x)

            x = self.normalization2[layer](x + residual)

            residual = x

            x = self.ffn1[layer](x)
            x = gelu(x)
            x = self.ffn2[layer](x)

            x = self.normalization3[layer](x + residual)


        x = softmax(self.fc(x))

        return x



class transformer():
    def __init__(self, layers, heads, model_size, vocab_size):
        
        self.encoder = encoder(layers, heads, model_size)
        self.decoder = decoder(layers, heads, model_size, vocab_size) 


model = transformer(layers=12, model_size=512, heads=8, vocab_size=10000)
input = np.random.rand(2, 20, 512)

ex = model.encoder(input)

input = model.decoder(input, ex)

print(input)
