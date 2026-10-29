import json
from pathlib import Path
class DatasetManifest:
    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    @property
    def files(self):
        return [Path(file) for file in self.data["dataset"]["files"]]

    @property
    def tokenizer_path(self):
        return self.data["tokenizer"]["path"]

    @property
    def vocab_size(self):
        return self.data["tokenizer"]["vocab_size"]
