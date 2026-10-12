import tensorflow as tf

from src.model.rmsnorm import RMSNorm
from src.model.feedforward import SwiGLU


def main():

    x = tf.random.normal(
        (2,512,768)
    )


    rms = RMSNorm()

    y = rms(x)

    print(
        "RMSNorm:",
        y.shape
    )


    ffn = SwiGLU(
        hidden_dim=768,
        intermediate_dim=2048
    )


    z = ffn(y)

    print(
        "SwiGLU:",
        z.shape
    )


if __name__ == "__main__":
    main()
