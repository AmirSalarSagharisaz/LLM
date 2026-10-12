from tokenizers import Tokenizer



class TokenizerDataset:


    def __init__(
        self,
        tokenizer_path
    ):

        self.tokenizer = Tokenizer.from_file(
            tokenizer_path
        )



    def encode(
        self,
        text
    ):

        result = self.tokenizer.encode(
            text
        )

        return result.ids
