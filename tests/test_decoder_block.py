import tensorflow as tf
from src.model.decoder_block import DecoderBlock

def main():
    batch, sequence, hidden_dim, heads, intermediate = 2, 128, 768, 12, 2048
    x = tf.random.normal((batch, sequence, hidden_dim))
    block = DecoderBlock(
        hidden_dim=hidden_dim,
        num_heads=heads,
        intermediate_dim=intermediate,
        max_sequence_length=512,
        dropout=0.1,
    )
    y = block(x, training=True)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Parameters:", block.count_params())
    
if __name__ == "__main__":
    main()