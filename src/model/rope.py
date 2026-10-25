import tensorflow as tf


class RotaryEmbedding(tf.keras.layers.Layer):
    def __init__(self, head_dim, max_sequence_length=2048, theta=10000.0, **kwargs):
        super().__init__(**kwargs)
        if head_dim % 2 != 0:
            raise ValueError(
                "RoPE requires an even head dimension."
            )
        self.head_dim = head_dim
        self.max_sequence_length = max_sequence_length
        self.theta = theta
        self._build_cache()

    def _build_cache(self):
        half_dim = self.head_dim // 2
        frequencies = (
            1.0/(self.theta**(tf.range(half_dim, dtype=tf.float32)/half_dim)))
        positions = tf.range(self.max_sequence_length, dtype=tf.float32)


        angles = tf.einsum(
            "i,j->ij",
            positions,
            frequencies
        )
        self.cos_cache = tf.cos(
            angles
        )
        self.sin_cache = tf.sin(
            angles
        )
    def rotate_half(self,x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = tf.stack([-x2, x1], axis=-1)

        shape = tf.shape(x)
        return tf.reshape(rotated, shape)

    def call(self, x, sequence_length=None):

        if sequence_length is None:
            sequence_length = tf.shape(x)[2]


        cos = self.cos_cache[:sequence_length]
        sin = self.sin_cache[:sequence_length]
        cos = tf.cast(cos,x.dtype)
        sin = tf.cast(sin,x.dtype)
        cos = tf.reshape(cos,
            (
                1,
                1,
                sequence_length,
                self.head_dim // 2
            ))
        sin = tf.reshape(sin,
            (
                1,
                1,
                sequence_length,
                self.head_dim // 2
            ))

        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated_x1 = (x1 * cos - x2 * sin)
        rotated_x2 = (x1 * sin + x2 * cos)
        output = tf.stack(
            [
                rotated_x1,
                rotated_x2
            ], axis=-1)
        output = tf.reshape(
            output,
            tf.shape(x)
        )
        return output

    def get_config(self):

        config = super().get_config()
        config.update(
            {
                "head_dim":
                    self.head_dim,

                "max_sequence_length":
                    self.max_sequence_length,

                "theta":
                    self.theta,
            }
        )

        return config
