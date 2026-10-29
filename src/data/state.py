import json
from pathlib import Path
class DatasetState:
    def __init__(self,
        path):
        self.path=Path(path)

    def save(self,state):
        with open(self.path, "w") as f:
            json.dump(state, f, indent=4)

    def load(self):
        if not self.path.exists():
            return None
        
        with open(self.path) as f:
            return json.load(f)
