from pathlib import Path
from tokenizers import Tokenizer



def save_tokenizer(
    tokenizer,
    output_directory
):
    """
    Save tokenizer files.

    Creates:

    tokenizer.json
    """

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    tokenizer_path = (
        output_directory
        /
        "tokenizer.json"
    )


    tokenizer.save(
        str(tokenizer_path)
    )


    return tokenizer_path



def load_tokenizer(
    tokenizer_path
):
    """
    Load trained tokenizer.
    """

    return Tokenizer.from_file(
        str(tokenizer_path)
    )
