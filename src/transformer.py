from modules.functions import linear, layer_normalization, dropout, softmax
from modules.activations import gelu
import math
# transformer

class transformer():
    def __init__(self, layers, model_size, heads, context_size, vocab_size):

        self.layers = layers
        self.model_size = model_size
        self.model_head_size = math.floor(model_size / heads)
        self.heads = heads
        self.context_size = context_size
        self.vocab_size = vocab_size


        # will need to add one for each layer
        self.d_normalization1 = layer_normalization(model_size)
        self.d_normalization2 = layer_normalization(model_size)
        self.d_normalization3 = layer_normalization(model_size)

        self.d_ffn1 = linear(model_size, model_size * 4)
        self.d_ffn2 = linear(model_size * 4, model_size)

        self.e_normalization1 = layer_normalization(model_size)
        self.e_normalization2 = layer_normalization(model_size)


        self.e_ffn1 = linear(model_size, model_size * 4)
        self.e_ffn2 = linear(model_size * 4, model_size)


        # tbh, don't know if this is correct
        self.final_ffn1 = linear(model_size, vocab_size)
        self.final_ffn2 = linear(vocab_size, 1)

    def encoder(self, x):

        # embeddings

        for _ in range(self.layers):
            # positional encoding here or with embeddings

            # save current state for future connection
            residual_1 = x

            # multi-head attention

            # add & normalize
            x = x + residual_1
            x = self.e_normalization1(x)

            # save current state for future connection
            residual_2 = x

            # position-wise feed forward network
            x = self.e_ffn1(x)
            x = gelu(x)
            x = self.e_ffn2(x)

            # add & normalize
            x = x + residual_2
            x = self.e_normalization2(x)


        return x




    def decoder(self, x):


        encoder_output = self.encoder(x)

        # embeddings 

        for _ in range(self.layers):
            # positional encoding here or with embeddings

            # save current state for future connection
            residual_1 = x

            # masked multi-head attention


            # add & noprmalize
            x = x + residual_1
            x = self.d_normalization1(x)

            # save current state for future connection
            residual_2 = x

            # multi-head attention with Q and K coming from the encoder

            # add & normalize
            x = x + residual_2
            x = self.d_normalization2(x)

            # save current state for future connection
            residual_3 = x

            # position-wise feed forward network with gelu activation
            x = self.d_ffn1(x)
            x = gelu(x)
            x = self.d_ffn2(x)

            # add & noprmalize
            x = x + residual_3
            x = self.d_normalization3(x)

        # project x into shape (vocab_size, 1) so that we can apply softmax
        x = self.final_ffn2(self.final_ffn1(x))

        # return a probability distribution of the vocabulary
        return softmax(x)


    def run(self):
        # run the encoder

        # run the decoder

