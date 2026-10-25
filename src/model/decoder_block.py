import tensorflow as tf
from src.model.rmsnorm import RMSNorm
from src.model.attention import CausalSelfAttention
from src.model.feedforward import SwiGLU


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_heads, intermediate_dim,
        max_sequence_length=2048,
        dropout=0.0,
        rope_theta=10000.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout
        self.attention_norm = RMSNorm()
        self.attention = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
            dropout=dropout,
            rope_theta=rope_theta
        )
        self.ffn_norm = RMSNorm()
        self.feed_forward = SwiGLU(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout
        )
        self.dropout = tf.keras.layers.Dropout(
            dropout
        )

    def call(self, inputs, training=False):
        residual = inputs
        x = self.attention_norm(inputs)
        x = self.attention(x, training=training)
        x = self.dropout(x, training=training)
        x = residual + x
        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x, training=training)
        x = self.dropout(x,training=training)
        x = residual + x
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim":
                    self.hidden_dim,

                "num_heads":
                    self.num_heads,

                "intermediate_dim":
                    self.intermediate_dim,

                "dropout":
                    self.dropout_rate
            }
        )

        return config
