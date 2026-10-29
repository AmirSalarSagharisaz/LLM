import json
import math
import time
import random
import platform
import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from src.tokenizer.tokenizer_utils import load_tokenizer
from src.data.tf_dataset import TensorFlowDatasetBuilder
from src.data.dataset_cursor import DatasetCursor
from src.model.transformer import DecoderOnlyTransformer
from src.training.training_state import TrainingState
from src.training.checkpoint_manager import CheckpointManager
from src.training.trainer import Trainer

CONFIG = {
    "experiment": {"name": "decoder_pretraining", "output_dir": "runs"},
    "seed": 42,
    "hardware": {"mixed_precision": True, "xla": True},
    "model": {
        "hidden_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_dim": 2048,
        "max_sequence_length": 256,
        "dropout": 0.0,
    },
    "training": {
        "epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "clip_norm": 1.0,
        "save_frequency": 1000,
        "log_frequency": 100,
    },
    "optimizer": {
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "beta_1": 0.9,
        "beta_2": 0.95,
        "epsilon": 1e-8,
    },
    "scheduler": {
        "warmup_steps": 2000,
        "total_steps": 100000,
        "minimum_learning_rate": 1e-5,
    },
    "tokenizer": {"path": "/home/nlp/generator/tokenizer/tokenizer.json"},
    "dataset": {
        "train_path": "/home/nlp/generator/data/raw",
        "validation_path": "/home/nlp/generator/data/ultra",
        "sequence_length": 256,
        "shuffle_buffer": 10000,
    },
    "checkpoint": {"directory": "checkpoints", "max_to_keep": 2, "resume": True},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def setup_hardware(config):
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU devices detected: {len(gpus)}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU detected")

    if config["hardware"]["xla"]:
        print("XLA compilation enabled")
        tf.config.optimizer.set_jit(True)

    if config["hardware"]["mixed_precision"]:
        print("Mixed precision enabled")
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")


def create_run_directory(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(config["experiment"]["output_dir"]) / f"{config['experiment']['name']}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_metadata(directory, config):
    metadata = {
        "time": datetime.datetime.now().isoformat(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "platform": platform.platform(),
        "config": config,
    }
    with open(directory / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_learning_rate, warmup_steps, total_steps, minimum_learning_rate):
        super().__init__()
        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.minimum_learning_rate = minimum_learning_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        warmup_lr = self.peak_learning_rate * step / tf.maximum(warmup, 1)
        progress = (step - warmup) / tf.maximum(total - warmup, 1)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine = 0.5 * (1 + tf.cos(math.pi * progress))
        decay_lr = self.minimum_learning_rate + (
            self.peak_learning_rate - self.minimum_learning_rate
        ) * cosine

        return tf.where(step < warmup, warmup_lr, decay_lr)


def create_optimizer(config):
    opt, sched = config["optimizer"], config["scheduler"]
    schedule = WarmupCosineSchedule(
        peak_learning_rate=opt["learning_rate"],
        warmup_steps=sched["warmup_steps"],
        total_steps=sched["total_steps"],
        minimum_learning_rate=sched["minimum_learning_rate"],
    )
    return tf.keras.optimizers.AdamW(
        learning_rate=schedule,
        weight_decay=opt["weight_decay"],
        beta_1=opt["beta_1"],
        beta_2=opt["beta_2"],
        epsilon=opt["epsilon"],
    )


def build_dataset(file_path, tokenizer_path, config, training=True):
    builder = TensorFlowDatasetBuilder(
        files=[Path(file_path)],
        tokenizer_path=tokenizer_path,
        sequence_length=config["dataset"]["sequence_length"],
        batch_size=config["training"]["batch_size"],
    )
    dataset = builder.build()

    if training:
        dataset = dataset.shuffle(
            buffer_size=config["dataset"]["shuffle_buffer"],
            seed=config["seed"],
            reshuffle_each_iteration=False,
        )
    return dataset


def create_model(vocab_size, config):
    m = config["model"]
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        hidden_dim=m["hidden_dim"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        intermediate_dim=m["intermediate_dim"],
        max_sequence_length=m["max_sequence_length"],
        dropout=m["dropout"],
    )
    model(tf.zeros((1, m["max_sequence_length"]), dtype=tf.int32))
    return model

def main():
    print("\n" + "=" * 70)
    print("Transformer Pretraining")
    print("=" * 70)

    set_seed(CONFIG["seed"])
    setup_hardware(CONFIG)

    run_directory = create_run_directory(CONFIG)
    save_metadata(run_directory, CONFIG)
    print(f"Run directory: {run_directory}")

    print("\nLoading tokenizer...")
    tokenizer_path = CONFIG["tokenizer"]["path"]
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size:,}")

    print("\nCreating datasets...")
    train_dataset = build_dataset(CONFIG["dataset"]["train_path"], tokenizer_path, CONFIG, training=True)
    validation_dataset = build_dataset(CONFIG["dataset"]["validation_path"], tokenizer_path, CONFIG, training=False)
    train_cursor = DatasetCursor(train_dataset)
    validation_cursor = DatasetCursor(validation_dataset)

    print("\nCreating model...")
    model = create_model(vocab_size, CONFIG)
    print(f"Parameters: {model.count_params():,}")

    print("\nCreating optimizer...")
    optimizer = create_optimizer(CONFIG)
    training_state = TrainingState()

    checkpoint_directory = Path(CONFIG["checkpoint"]["directory"]) / CONFIG["experiment"]["name"]
    checkpoint_manager = CheckpointManager(
        model=model,
        optimizer=optimizer,
        training_state=training_state,
        dataset_cursor=train_cursor,
        directory=str(checkpoint_directory),
        max_to_keep=CONFIG["checkpoint"]["max_to_keep"],
    )

    if CONFIG["checkpoint"]["resume"]:
        if checkpoint_manager.restore_latest():
            print("Checkpoint restored successfully")
        else:
            print("No checkpoint found. Starting new run.")

    print("\nCreating trainer...")
    train_cfg = CONFIG["training"]
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        training_state=training_state,
        checkpoint_manager=checkpoint_manager,
        dataset_cursor=train_cursor,
        validation_cursor=validation_cursor,
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        clip_norm=train_cfg["clip_norm"],
        save_frequency=train_cfg["save_frequency"],
        log_frequency=train_cfg["log_frequency"],
        tensorboard_dir=str(run_directory / "tensorboard"),
    )

    print("\nInitial training state:")
    print(training_state.state_dict())
    print("\nStarting training...")
    start = time.time()
    trainer.train(epochs=train_cfg["epochs"])
    elapsed = time.time() - start
    print("\nTraining finished")
    print(f"Total time: {elapsed / 3600:.2f} hours")
    print("Final state:")
    print(training_state.state_dict())

if __name__ == "__main__":
    main()