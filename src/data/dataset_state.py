from dataclasses import dataclass
@dataclass
class DatasetPosition:
    shard_index:int = 0
    batch_index:int = 0

    def to_dict(self):
        return {"shard_index": self.shard_index, "batch_index": self.batch_index}

    def restore(self,data):
        self.shard_index=data["shard_index"]
        self.batch_index=data["batch_index"]
