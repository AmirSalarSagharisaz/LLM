import tensorflow as tf

from src.model.decoder_block import DecoderBlock
from src.model.rmsnorm import RMSNorm



class DecoderOnlyTransformer(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        max_sequence_length=2048,
        dropout=0.0,
        rope_theta=10000.0,
        tie_embeddings=True,
        **kwargs
    ):

        super().__init__(**kwargs)



        self.vocab_size = vocab_size

        self.hidden_dim = hidden_dim

        self.num_layers = num_layers

        self.num_heads = num_heads

        self.intermediate_dim = intermediate_dim

        self.max_sequence_length = max_sequence_length

        self.dropout_rate = dropout

        self.tie_embeddings = tie_embeddings



        #
        # Token embedding
        #

        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_dim,
            name="token_embedding"
        )



        self.embedding_dropout = tf.keras.layers.Dropout(
            dropout
        )



        #
        # Transformer blocks
        #

        self.decoder_blocks = []


        for i in range(num_layers):

            self.decoder_blocks.append(

                DecoderBlock(

                    hidden_dim=hidden_dim,

                    num_heads=num_heads,

                    intermediate_dim=intermediate_dim,

                    max_sequence_length=max_sequence_length,

                    dropout=dropout,

                    rope_theta=rope_theta,

                    name=f"decoder_block_{i}"

                )

            )



        #
        # Final normalization
        #

        self.final_norm = RMSNorm()



        #
        # Language modeling head
        #

        if not tie_embeddings:

            self.lm_head = tf.keras.layers.Dense(
                vocab_size,
                use_bias=False,
                name="lm_head"
            )



    def call(
        self,
        input_ids,
        training=False
    ):

        """
        Forward pass.

        Input:

            input_ids

            shape:

            (batch, sequence_length)


        Output:

            logits

            shape:

            (batch, sequence_length, vocab_size)

        """



        #
        # Token embeddings
        #

        x = self.token_embedding(
            input_ids
        )



        x = self.embedding_dropout(
            x,
            training=training
        )



        #
        # Transformer layers
        #

        for block in self.decoder_blocks:

            x = block(
                x,
                training=training
            )



        #
        # Final normalization
        #

        x = self.final_norm(
            x
        )



        #
        # Output logits
        #

        if self.tie_embeddings:


            logits = tf.matmul(

                x,

                self.token_embedding.embeddings,

                transpose_b=True

            )


        else:

            logits = self.lm_head(x)



        return logits



    def get_config(self):

        config = super().get_config()

        config.update(

            {

                "vocab_size":
                    self.vocab_size,

                "hidden_dim":
                    self.hidden_dim,

                "num_layers":
                    self.num_layers,

                "num_heads":
                    self.num_heads,

                "intermediate_dim":
                    self.intermediate_dim,

                "max_sequence_length":
                    self.max_sequence_length,

                "dropout":
                    self.dropout_rate,

                "tie_embeddings":
                    self.tie_embeddings,

            }

        )

        return config
