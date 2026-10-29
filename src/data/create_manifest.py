from pathlib import Path
import json
import re
from datetime import datetime

def natural_sort_key(path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]

def load_tokenizer_analysis():
    path = Path("tokenizer/tokenizer_analysis.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_manifest():
    data_dir = Path("data/raw")
    files = sorted(data_dir.glob("*.txt"), key=natural_sort_key)
    analysis = (load_tokenizer_analysis())
    manifest = {
        "created": datetime.now().isoformat(),
        "dataset": {"name": "DTR2024_2", "format": "text", "file_count": len(files), "files": [str(x) for x in files]},
        "tokenizer":{"path": "tokenizer/tokenizer.json", "analysis": "tokenizer/tokenizer_analysis.json", "vocab_size": analysis["vocabulary"]["vocab_size"]},
        "statistics":{"total_characters": analysis["dataset"]["total_characters"], "total_tokens": analysis["dataset"]["total_tokens"], 
                      "characters_per_token": analysis["dataset"]["characters_per_token"]},
        "training":{"objective": "causal_language_modeling", "sequence_length": 512, "order_preserved": True}
    }

    with open("configs/dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)


if __name__ == "__main__":
    create_manifest()
