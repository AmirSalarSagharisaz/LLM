import tensorflow as tf


class RMSNorm(tf.keras.layers.Layer):
    """
    Root Mean Square Layer Normalization.

    RMSNorm(x) =
        x / sqrt(mean(x^2) + epsilon) * gamma

    Used in modern decoder-only transformers.
    """

    def __init__(
        self,
        epsilon=1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.epsilon = epsilon


    def build(self, input_shape):

        hidden_dim = input_shape[-1]

        self.gamma = self.add_weight(
            name="gamma",
            shape=(hidden_dim,),
            initializer="ones",
            trainable=True
        )

        super().build(input_shape)


    def call(self, inputs):

        input_dtype = inputs.dtype

        x = tf.cast(inputs, tf.float32)

        rms = tf.math.rsqrt(
            tf.reduce_mean(
                tf.square(x),
                axis=-1,
                keepdims=True
            )
            +
            self.epsilon
        )

        x = x * rms

        x = x * self.gamma

        return tf.cast(
            x,
            input_dtype
        )


    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "epsilon": self.epsilon
            }
        )

        return config
