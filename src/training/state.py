from dataclasses import dataclass, asdict
from typing import Dict
import json
import time
@dataclass
class TrainingState:
    global_step: int = 0
    epoch: int = 0
    dataset_shard: int = 0
    dataset_batch: int = 0
    tokens_seen: int = 0
    total_loss: float = 0.0
    optimizer_steps: int = 0
    start_time: float = 0.0
    last_checkpoint_time: float = 0.0
    tokenizer_hash: str = ""
    dataset_hash: str = ""
    config_hash: str = ""
    trainer_version: str = "1.0"

    def to_dict(self):
        return asdict(self)

    def save_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_json(cls, path):
        with open(path) as f:
            data=json.load(f)
        return cls(**data)

    def update_checkpoint_time(self):
        self.last_checkpoint_time = time.time()
