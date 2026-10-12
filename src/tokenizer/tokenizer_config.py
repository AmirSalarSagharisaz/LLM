import json
from pathlib import Path



def create_tokenizer_config(
    tokenizer_path,
    output_path
):

    config = {

        "tokenizer_type":
            "Byte-Level BPE",

        "tokenizer_file":
            str(tokenizer_path),

        "vocab_size":
            32000,

        "special_tokens":
        {

            "pad_token":
                "<pad>",

            "unk_token":
                "<unk>",

            "bos_token":
                "<bos>",

            "eos_token":
                "<eos>"

        },

        "frozen":
            True

    }


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )
