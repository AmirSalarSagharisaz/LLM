import tensorflow as tf

from src.data.text_stream import TextStream
from src.data.tokenizer_dataset import TokenizerDataset
from src.data.packing import SequencePacker



class TensorFlowDatasetBuilder:



    def __init__(
        self,
        files,
        tokenizer_path,
        sequence_length,
        batch_size
    ):

        self.files = files

        self.encoder = TokenizerDataset(
            tokenizer_path
        )

        self.packer = SequencePacker(
            sequence_length
        )

        self.batch_size = batch_size



    def generator(self):


        stream = TextStream(
            self.files
        )


        for text in stream:


            tokens = (
                self.encoder.encode(
                    text
                )
            )


            for sample in self.packer.pack(
                tokens
            ):

                yield sample



    def build(self):


        dataset = tf.data.Dataset.from_generator(

            self.generator,

            output_signature=(

                tf.TensorSpec(
                    shape=(None,),
                    dtype=tf.int32
                ),

                tf.TensorSpec(
                    shape=(None,),
                    dtype=tf.int32
                )

            )

        )


        dataset = dataset.batch(
            self.batch_size,
            drop_remainder=True
        )


        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )


        return dataset
