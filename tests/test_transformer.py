import tensorflow as tf

from src.model.transformer import DecoderOnlyTransformer



def main():


    vocab_size = 32000

    batch = 2

    sequence = 128



    model = DecoderOnlyTransformer(

        vocab_size=vocab_size,

        hidden_dim=768,

        num_layers=12,

        num_heads=12,

        intermediate_dim=2048,

        max_sequence_length=512,

        dropout=0.1

    )


    input_ids = tf.random.uniform(

        (

            batch,

            sequence

        ),

        minval=0,

        maxval=vocab_size,

        dtype=tf.int32

    )



    logits = model(

        input_ids,

        training=True

    )


    print(

        "Input shape:",

        input_ids.shape

    )


    print(

        "Logits shape:",

        logits.shape

    )


    print(

        "Parameters:",

        model.count_params()

    )



if __name__ == "__main__":

    main()
