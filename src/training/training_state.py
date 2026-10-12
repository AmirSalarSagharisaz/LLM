import tensorflow as tf


class TrainingState(tf.Module):
    """
    Persistent training state.

    Saved inside TensorFlow checkpoints.
    """

    def __init__(self):

        super().__init__()

        # optimizer/model step

        self.global_step = tf.Variable(
            0,
            dtype=tf.int64,
            trainable=False,
            name="global_step"
        )


        # epoch tracking

        self.epoch = tf.Variable(
            0,
            dtype=tf.int64,
            trainable=False,
            name="epoch"
        )


        # batch position inside epoch

        self.batch_step = tf.Variable(
            0,
            dtype=tf.int64,
            trainable=False,
            name="batch_step"
        )


        # useful for token based training

        self.tokens_seen = tf.Variable(
            0,
            dtype=tf.int64,
            trainable=False,
            name="tokens_seen"
        )


    def increment_step(self):

        self.global_step.assign_add(1)

        self.batch_step.assign_add(1)


    #def increment_step(
    #    self,
    #    tokens=0
    #):

    #    self.global_step.assign_add(
    #        1
    #    )

    #    self.batch_step.assign_add(
    #        1
    #    )

    #    self.tokens_seen.assign_add(
    #        tf.cast(
    #            tokens,
    #            tf.int64
    #        )
    #    )


    def next_epoch(self):

        self.epoch.assign_add(1)

        self.batch_step.assign(0)


    def reset_batch(self):

        self.batch_step.assign(0)


    def state_dict(self):

        return {

            "global_step":
                int(self.global_step.numpy()),

            "epoch":
                int(self.epoch.numpy()),

            "batch_step":
                int(self.batch_step.numpy()),

            "tokens_seen":
                int(self.tokens_seen.numpy())

        }
