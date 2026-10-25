import tensorflow as tf
from src.model.rope import RotaryEmbedding

class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self,
        hidden_dim,
        num_heads,
        max_sequence_length=2048,
        dropout=0.0,
        rope_theta=10000.0,
        **kwargs
    ):

        super().__init__(**kwargs)


        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads"
            )


        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout_rate = dropout
        self.max_sequence_length = max_sequence_length
        self.q_projection = tf.keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            name="query_projection"
        )
        self.k_projection = tf.keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            name="key_projection"
        )
        self.v_projection = tf.keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            name="value_projection"
        )
        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_sequence_length=max_sequence_length,
            theta=rope_theta
        )
        self.output_projection = tf.keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            name="output_projection"
        )
        self.dropout = tf.keras.layers.Dropout(
            dropout
        )

    def split_heads(self,
        x
    ):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x,
            (
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim
            )
        )
        x = tf.transpose(
            x,
            perm=[0,2,1,3]
        )
        return x
    def merge_heads(self, x):
        x = tf.transpose( x,
            perm=[0,2,1,3]
        )
        batch = tf.shape(x)[0]
        seq = tf.shape(x)[1]
        x = tf.reshape(
            x,
            (
                batch,
                seq,
                self.hidden_dim
            )
        )
        return x

    def causal_mask(
        self,
        sequence_length
    ):
        
        mask = tf.linalg.band_part(
            tf.ones(
                (
                    sequence_length,
                    sequence_length
                )
            ),
            -1,
            0
        )
        mask = 1.0 - mask
        return mask[None,None,:,:]

    def call(
        self,
        inputs,
        training=False
    ):
        sequence_length = tf.shape(inputs)[1]
        q = self.q_projection(inputs)
        k = self.k_projection(inputs)
        v = self.v_projection(inputs)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        q = self.rope(q)
        k = self.rope(k)
        scale = tf.cast(
            tf.math.sqrt(
                tf.cast(
                    self.head_dim,
                    tf.float32
                )
            ), q.dtype)
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = (attention_scores/scale)
        mask = self.causal_mask(sequence_length)
        mask = tf.cast(mask, attention_scores.dtype)
        attention_scores += (
            mask *
            tf.cast(
                -1e4,
                attention_scores.dtype
            )
        )
        attention_weights = tf.nn.softmax(
            attention_scores,
            axis=-1
        )
        attention_weights = self.dropout(
            attention_weights,
            training=training
        )
        output = tf.matmul(attention_weights, v)
        output = self.merge_heads(output)
        output = self.output_projection(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim":self.hidden_dim,
                "num_heads":self.num_heads,
                "max_sequence_length":
                    self.max_sequence_length,
                "dropout":
                    self.dropout_rate
            }
        )

        return config
