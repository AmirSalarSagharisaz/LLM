from pathlib import Path
import json



def validate():


    with open(
        "configs/dataset_manifest.json",
        "r"
    ) as f:

        manifest=json.load(f)



    print("="*60)

    print("DATASET VALIDATION")

    print("="*60)



    tokenizer = Path(
        manifest["tokenizer"]["path"]
    )


    assert tokenizer.exists(), \
        "Tokenizer missing"



    for file in manifest["dataset"]["files"]:

        path = Path(file)

        assert path.exists(), \
            f"Missing {file}"

        print(
            "Found:",
            file
        )



    print()

    print(
        "Tokenizer:",
        tokenizer
    )


    print()

    print(
        "Total tokens:",
        manifest["statistics"]["total_tokens"]
    )


    print()

    print(
        "Validation successful."
    )



if __name__=="__main__":
    validate()
