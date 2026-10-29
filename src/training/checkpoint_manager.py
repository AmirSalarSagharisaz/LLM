import os
import tensorflow as tf

class CheckpointManager:
    def __init__(self,model, optimizer, training_state, dataset_cursor, directory,max_to_keep=5):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, training_state=training_state, dataset_cursor=dataset_cursor)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory, max_to_keep=max_to_keep)
    def save(self):
        step = int(self.checkpoint.training_state.global_step.numpy())
        path = self.manager.save(checkpoint_number=step)
        print(f"Checkpoint saved: {path}")
        return path
    
    def restore_latest(self):
        latest = self.manager.latest_checkpoint
        if latest: 
            self.checkpoint.restore(latest).expect_partial()
            print(f"Checkpoint restored: {latest}")
            return True
        print("No checkpoint available")
        return False

    def latest_checkpoint(self):
        return self.manager.latest_checkpoint
