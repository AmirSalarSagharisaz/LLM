class TrainingConfig:


    # training
    epochs = 2

    batch_size = 2

    sequence_length = 512


    # optimization

    learning_rate = 3e-4

    weight_decay = 0.1


    gradient_accumulation_steps = 8



    # checkpoint

    checkpoint_dir = (
        "checkpoints"
    )


    save_frequency = 1000



    # evaluation

    eval_frequency = 5000



    # precision

    mixed_precision = True
