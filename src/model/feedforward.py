import tensorflow as tf


class SwiGLU(tf.keras.layers.Layer):
    """
    SwiGLU feed-forward network.

    Formula:

    FFN(x)=Down(
              SiLU(Gate(x))
              *
              Up(x)
           )

    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        dropout=0.0,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout


        self.gate_proj = tf.keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            name="gate_projection"
        )


        self.up_proj = tf.keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            name="up_projection"
        )


        self.down_proj = tf.keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            name="down_projection"
        )


        self.dropout = tf.keras.layers.Dropout(
            dropout
        )


    def call(
        self,
        inputs,
        training=False
    ):

        gate = self.gate_proj(inputs)

        gate = tf.keras.activations.silu(
            gate
        )

        value = self.up_proj(inputs)


        x = gate * value


        x = self.down_proj(x)


        x = self.dropout(
            x,
            training=training
        )

        return x


    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout_rate
            }
        )

        return config
