import tensorflow as tf

from src.model.rope import RotaryEmbedding



def main():

    batch = 2
    heads = 12
    seq = 512
    head_dim = 64


    x = tf.random.normal(
        (
            batch,
            heads,
            seq,
            head_dim
        )
    )


    rope = RotaryEmbedding(
        head_dim=head_dim,
        max_sequence_length=seq
    )


    y = rope(x)


    print(
        "Input:",
        x.shape
    )


    print(
        "Output:",
        y.shape
    )


    difference = tf.reduce_mean(
        tf.abs(
            x-y
        )
    )


    print(
        "Mean difference:",
        difference.numpy()
    )



if __name__ == "__main__":
    main()
