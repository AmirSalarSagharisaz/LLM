from src.tokenizer.tokenizer_utils import load_tokenizer



def main():

    tokenizer = load_tokenizer(
        "tokenizer/tokenizer.json"
    )


    vocab = tokenizer.get_vocab()


    print("=" * 60)
    print("TOKENIZER VOCABULARY VERIFICATION")
    print("=" * 60)


    print(
        f"Vocabulary size: {len(vocab)}"
    )


    print("\nSpecial tokens:")

    special_tokens = [

        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>"

    ]


    for token in special_tokens:

        if token in vocab:

            print(
                f"{token}: ID {vocab[token]}"
            )

        else:

            print(
                f"{token}: NOT FOUND"
            )


    print("=" * 60)



if __name__ == "__main__":
    main()
