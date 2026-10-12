from src.tokenizer.tokenizer_utils import load_tokenizer



def main():

    tokenizer = load_tokenizer(
        "tokenizer/tokenizer.json"
    )


    text = (
        "The transformer model "
        "uses rotary embeddings."
    )


    encoded = tokenizer.encode(
        text
    )


    print(
        "Tokens:"
    )

    print(
        encoded.tokens
    )


    print(
        "IDs:"
    )

    print(
        encoded.ids
    )


    decoded = tokenizer.decode(
        encoded.ids
    )


    print(
        "Decoded:"
    )

    print(
        decoded
    )



if __name__ == "__main__":
    main()
