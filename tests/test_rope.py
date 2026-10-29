import tensorflow as tf
from src.model.rope import RotaryEmbedding

def main():
    batch, heads, seq, head_dim = 2, 12, 512, 64
    x = tf.random.normal((batch, heads, seq, head_dim))
    rope = RotaryEmbedding(head_dim=head_dim, max_sequence_length=seq)
    y = rope(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
    print("Mean difference:", tf.reduce_mean(tf.abs(x - y)).numpy())

if __name__ == "__main__":
    main()