import tensorflow as tf
from src.model.attention import CausalSelfAttention

def main():
    batch, seq, hidden, heads = 2, 128, 768, 12

    x = tf.random.normal((batch, seq, hidden))
    attention = CausalSelfAttention(
        hidden_dim=hidden,
        num_heads=heads,
        max_sequence_length=512,
        dropout=0.1,
    )
    y = attention(x, training=True)
    
    print("Input:", x.shape)
    print("Output:", y.shape)
    print("Parameters:", attention.count_params())

if __name__ == "__main__":
    main()