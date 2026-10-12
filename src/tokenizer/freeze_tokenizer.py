from src.tokenizer.tokenizer_config import (
    create_tokenizer_config
)



if __name__ == "__main__":


    create_tokenizer_config(

        "tokenizer/tokenizer.json",

        "tokenizer/tokenizer_config.json"

    )


    print(
        "Tokenizer frozen successfully."
    )
