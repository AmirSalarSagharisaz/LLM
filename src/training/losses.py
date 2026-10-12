import tensorflow as tf



loss_fn = (
    tf.keras.losses
    .SparseCategoricalCrossentropy(
        from_logits=True
    )
)



def compute_loss(
    targets,
    logits
):

    return loss_fn(
        targets,
        logits
    )
