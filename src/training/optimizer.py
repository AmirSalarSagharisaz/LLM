import tensorflow as tf

def create_optimizer(learning_rate,weight_decay):
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta_1=0.9,
        beta_2=0.95,
        epsilon=1e-8
    )
    return optimizer
