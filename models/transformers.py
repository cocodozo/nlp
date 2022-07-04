import tensorflow as tf


def scaled_dot_product_attention(inputs, return_attention_scores=False, attention_mask=None):

    Q, V, K = inputs

    dot_products = tf.matmul(Q, K, transpose_b=True)

    dim = tf.cast(Q.shape[-1], Q.dtype)
    attention_logits = dot_products / dim

    if attention_mask is not None:
        attention_logits += (1 - attention_mask) * -1e9

    attention_scores = tf.nn.softmax(attention_logits, axis=-1)

    output = tf.matmul(attention_scores, V)

    if return_attention_scores:
        return output, attention_scores
    else:
        return output


class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, num_heads, head_size, return_attention_scores=False, **kwargs):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.head_size = head_size
        self.return_attention_scores = return_attention_scores

        self.ffQ = tf.keras.layers.Dense(self.head_size)
        self.ffV = tf.keras.layers.Dense(self.head_size)
        self.ffK = tf.keras.layers.Dense(self.head_size)
        self.ffO = tf.keras.layers.Dense(self.head_size * self.num_heads)

    def split(self, X, ):

        X = tf.reshape(X, (X.shape[0], -1, self.num_heads, self.head_size))
        return tf.transpose(X, [0, 2, 1, 3])

    def call(self, inputs, attention_mask=None):

        Q, V, K = inputs

        assert Q.shape[-1] == V.shape[-1] == K.shape[-1] == self.num_heads * self.head_size

        Q = self.ffQ(self.split(Q))
        V = self.ffV(self.split(V))
        K = self.ffK(self.split(K))

        scaled_attention, attention_scores = scaled_dot_product_attention([Q, V, K], True, attention_mask)

        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
        attention_output = tf.reshape(scaled_attention, (Q.shape[0], -1, self.num_heads * self.head_size))

        attention_output = self.ffO(attention_output)

        if self.return_attention_scores:
            return attention_output, attention_scores
        else:
            return attention_output


class SelfAttentionBlock(tf.keras.Model):

    def __init__(self, num_heads, head_size, hidden_dim, dropout_rate=0., ff_activation=None, return_attention_scores=False, **kwargs):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.head_size = head_size
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.ff_activation = ff_activation
        self.return_attention_scores = return_attention_scores

        self.dim = self.num_heads * self.head_size

        self.mha = MultiHeadAttention(self.num_heads, self.head_size, self.return_attention_scores)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.ff1 = tf.keras.layers.Dense(self.hidden_dim, activation=self.ff_activation)
        self.ff2 = tf.keras.layers.Dense(self.dim, activation=self.ff_activation)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, attention_mask=None, training=True):

        Y = self.ln1(inputs)
        if self.return_attention_scores:
            Y, attention_scores = self.mha([Y, Y, Y], attention_mask)
        else:
            Y = self.mha([Y, Y, Y])
        _Y = Y + inputs

        Y = self.ln2(_Y)
        Y = self.ff1(Y)
        Y = self.ff2(Y)
        Y = self.dropout(Y, training)
        Y = Y + _Y

        if self.return_attention_scores:
            return Y, attention_scores
        else:
            return Y


def look_ahead_mask(size):
    return tf.linalg.band_part(tf.ones((size, size)), -1, 0)


# sab = SelfAttentionBlock(12, 32, 512, ff_activation='gelu', return_attention_scores=True)

# a = tf.random.normal((32, 128, 12*32))
# b, attention_scores = sab(a)
# print(a.shape, b.shape)
# print(attention_scores.shape)
# print(attention_scores[0, 0, 0, :])
# print(tf.reduce_sum(attention_scores[0, 0, 0, :]))

# sab.summary(line_length=120)


# import matplotlib.pyplot as plt
# def plot_attention_simple( Q,VK, output, attention):

  
#     fig,(ax,ax2) = plt.subplots(1,2,figsize=(10,4))

#     ax.matshow(attention)
#     ax.set_xticks(range(len(VK)))
#     ax.set_yticks(range(len(Q)))

#     labels = [str(elem) for elem in VK]
#     ax.set_xticklabels(labels, rotation=90)

#     labels = [str(elem) for elem in Q]
#     ax.set_yticklabels(labels)
#     ax.set_xlabel("Value=Key")
#     ax.set_ylabel("Query")

#     res=ax2.matshow(output)
#     ax2.axis("off")

#     fig.colorbar(res)
#     fig.tight_layout()

# import numpy as np
# VK=np.random.randint(-1,2,[2,7,2]).astype(np.float32)
# Q =np.random.randint(-1,2,[2,7,2]).astype(np.float32)

# output, attention_scores = scaled_dot_product_attention([Q,VK,VK], True, look_ahead_mask(7))
# plot_attention_simple(Q[0,:,:], VK[0,:,:], output[0,:,:], attention_scores[0,:,:])
# plt.show()