from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import models
from tokenizers import trainers
from tokenizers import pre_tokenizers
from tokenizers import decoders
from tokenizers import normalizers

from src.tokenizer.tokenizer_utils import save_tokenizer



class TextFileIterator:
    """
    Streaming text iterator.

    Prevents loading large datasets
    completely into RAM.
    """

    def __init__(
        self,
        files,
        chunk_size=1024*1024
    ):

        self.files = files
        self.chunk_size = chunk_size



    def __iter__(self):

        for file in self.files:

            with open(
                file,
                "r",
                encoding="utf-8"
            ) as f:

                while True:

                    text = f.read(
                        self.chunk_size
                    )

                    if not text:
                        break

                    yield text



def train_byte_bpe_tokenizer(
    text_files,
    vocab_size=32000,
    output_directory="tokenizer"
):


    tokenizer = Tokenizer(
        models.BPE(
            unk_token="<unk>"
        )
    )


    tokenizer.normalizer = (
        normalizers.Sequence(
            []
        )
    )


    tokenizer.pre_tokenizer = (
        pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
    )


    tokenizer.decoder = (
        decoders.ByteLevel()
    )


    trainer = trainers.BpeTrainer(

        vocab_size=vocab_size,

        min_frequency=2,

        special_tokens=[

            "<pad>",

            "<unk>",

            "<bos>",

            "<eos>"

        ]

    )


    iterator = TextFileIterator(
        text_files
    )


    tokenizer.train_from_iterator(
        iterator,
        trainer=trainer
    )


    save_tokenizer(
        tokenizer,
        output_directory
    )


    return tokenizer



if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--files",
        nargs="+",
        required=True
    )


    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000
    )


    parser.add_argument(
        "--output",
        default="tokenizer"
    )


    args = parser.parse_args()



    train_byte_bpe_tokenizer(

        args.files,

        args.vocab_size,

        args.output

    )
