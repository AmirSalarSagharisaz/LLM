class TrainingConfig:
    epochs = 2
    batch_size = 2
    sequence_length = 512
    learning_rate = 3e-4
    weight_decay = 0.1
    gradient_accumulation_steps = 8
    checkpoint_dir = ("checkpoints")
    save_frequency = 1000
    eval_frequency = 5000
    mixed_precision = True
