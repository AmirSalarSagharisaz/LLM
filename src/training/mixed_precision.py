import tensorflow as tf



def enable_mixed_precision():

    policy = tf.keras.mixed_precision.Policy(
        "mixed_float16"
    )

    tf.keras.mixed_precision.set_global_policy(
        policy
    )


    print(
        "Mixed precision enabled:"
    )

    print(
        policy
    )
