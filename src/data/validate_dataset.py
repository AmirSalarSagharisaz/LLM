from pathlib import Path
import json

def validate():
    with open("configs/dataset_manifest.json", "r") as f:
        manifest=json.load(f)
    print("="*60)
    print("DATASET VALIDATION")
    print("="*60)
    tokenizer = Path(manifest["tokenizer"]["path"])
    assert tokenizer.exists()
    for file in manifest["dataset"]["files"]:
        path = Path(file)
        assert path.exists(), f"Missing {file}"
        print("Found:", file)
    print("Tokenizer:", tokenizer)
    print("Total tokens:", manifest["statistics"]["total_tokens"])
    print("Validation successful")

if __name__=="__main__":
    validate()
