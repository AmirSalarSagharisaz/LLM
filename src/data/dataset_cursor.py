import tensorflow as tf
class DatasetCursor(tf.Module):
    def __init__(self,
    dataset):
        super().__init__()
        self.dataset = dataset
        self.batch_index = tf.Variable(0, dtype=tf.int64, trainable=False, name="dataset_batch_index")
        self.samples_seen = tf.Variable(0, dtype=tf.int64, trainable=False, name="dataset_samples_seen")
        self.epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="dataset_epoch")

    def __iter__(self):
        start_index = int(self.batch_index.numpy())
        for index, batch in enumerate(self.dataset):
            if index < start_index:
                continue
            yield batch
            self.batch_index.assign(index + 1)
            self.samples_seen.assign_add(1)

    def reset_epoch(self):
        self.epoch.assign_add(1)
        self.batch_index.assign(0)

    def reset(self):
        self.batch_index.assign(0)
        self.samples_seen.assign(0)

    def state_dict(self):
        return {"epoch": int(self.epoch.numpy()), "batch_index": int(self.batch_index.numpy()), "samples_seen": int(self.samples_seen.numpy())}