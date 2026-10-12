from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tokenizers import Tokenizer

from src.model.transformer import DecoderOnlyTransformer

IGNORE_INDEX = -100


def configure_gpu() -> None:
    """Enable on-demand GPU memory allocation where possible."""
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # The runtime may already have initialized the device.
            pass


def variable_path(variable: Any) -> str:
    """Return a stable-ish readable Keras variable path across Keras versions."""
    return str(getattr(variable, "path", None) or variable.name)


def flatten_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    """Collect nested layers without duplicates, including layers inside blocks."""
    if hasattr(model, "_flatten_layers"):
        try:
            candidates = model._flatten_layers(include_self=False, recursive=True)
            unique: list[tf.keras.layers.Layer] = []
            seen: set[int] = set()
            for layer in candidates:
                if id(layer) not in seen:
                    seen.add(id(layer))
                    unique.append(layer)
            return unique
        except (TypeError, AttributeError):
            pass

    unique = []
    seen: set[int] = set()

    def visit(parent: tf.keras.layers.Layer) -> None:
        for child in getattr(parent, "layers", []):
            if id(child) not in seen:
                seen.add(id(child))
                unique.append(child)
                visit(child)

    visit(model)
    return unique


def latest_checkpoint_path(checkpoint_dir: Path) -> str:
    """Resolve a TensorFlow checkpoint prefix from a checkpoint directory."""
    checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))
    if not checkpoint:
        raise FileNotFoundError(
            f"No TensorFlow checkpoint found in: {checkpoint_dir}\n"
            "Pass --base-checkpoint-dir pointing to the directory containing "
            "the checkpoint .index and .data files."
        )
    return checkpoint


def build_and_restore_base_model(args: argparse.Namespace) -> tf.keras.Model:
    """Build the same architecture/configuration used by pretraining, then restore it."""
    model = DecoderOnlyTransformer(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_dim=args.intermediate_dim,
        max_sequence_length=args.sequence_length,
        dropout=0.0,
        rope_theta=10000.0,
        tie_embeddings=True,
        name="decoder_only_transformer",
    )

    # Materialize variables before restoring a TensorFlow object checkpoint.
    _ = model(tf.zeros([1, args.sequence_length], dtype=tf.int32), training=False)

    checkpoint_path = latest_checkpoint_path(Path(args.base_checkpoint_dir))
    status = tf.train.Checkpoint(model=model).restore(checkpoint_path)
    # The pretraining checkpoint also contains optimizer/training-state objects;
    # only model weights are required for IFT, so extra checkpoint entries are OK.
    status.assert_existing_objects_matched()
    status.expect_partial()
    print(f"[base] restored model weights from: {checkpoint_path}")
    return model


def prepare_int4_lora(model: tf.keras.Model, rank: int, alpha: int) -> list[Any]:
    """Quantize Dense kernels to Keras int4 and enable LoRA on each Dense layer."""
    if not hasattr(model, "quantize"):
        raise RuntimeError(
            "This installed Keras version does not expose model.quantize(). "
            "The QLoRA-style script requires a Keras 3 version with native int4 "
            "quantization support. Check `python -c 'import keras, tensorflow as tf; "
            "print(keras.__version__, tf.__version__)'` before changing packages."
        )

    dense_layers = [
        layer for layer in flatten_layers(model)
        if isinstance(layer, tf.keras.layers.Dense)
    ]
    if not dense_layers:
        raise RuntimeError(
            "No tf.keras.layers.Dense layers were found. This script expects Dense "
            "projections in the decoder attention and feed-forward blocks."
        )

    # Quantize only linear layers. Keep the tied token embedding and RMSNorm in
    # their original precision, as in common QLoRA configurations.
    model.quantize(
        "int4",
        filters=lambda layer: isinstance(layer, tf.keras.layers.Dense),
    )

    quantized_count = sum(
        1 for layer in dense_layers
        if getattr(layer, "quantization_mode", None) == "int4"
    )
    if quantized_count == 0:
        raise RuntimeError(
            "Keras returned from model.quantize('int4'), but none of the Dense "
            "layers reports int4 quantization. Check the installed Keras version "
            "and whether your custom model's Dense layers are compatible."
        )

    enabled_count = 0
    for layer in dense_layers:
        if not callable(getattr(layer, "enable_lora", None)):
            raise RuntimeError(
                f"Dense layer {layer.name!r} has no enable_lora() method. "
                "Use a Keras release whose Dense layer supports LoRA."
            )
        if not getattr(layer, "lora_enabled", False):
            layer.enable_lora(rank=rank, lora_alpha=alpha)
        enabled_count += int(bool(getattr(layer, "lora_enabled", False)))

    # Train only A/B adapter variables. Other model variables may still appear in
    # model.trainable_variables, but they are deliberately excluded from updates.
    adapters = [
        variable for variable in model.trainable_variables
        if "lora_kernel_a" in variable_path(variable)
        or "lora_kernel_b" in variable_path(variable)
    ]
    adapters.sort(key=variable_path)
    if not adapters:
        raise RuntimeError(
            "LoRA was enabled, but no LoRA adapter variables were found in "
            "model.trainable_variables."
        )

    print(
        f"[qlora-style] Dense layers={len(dense_layers)}, int4={quantized_count}, "
        f"LoRA enabled={enabled_count}, trainable adapter tensors={len(adapters)}, "
        f"trainable adapter parameters={sum(int(np.prod(v.shape)) for v in adapters):,}"
    )
    return adapters


def find_special_token_id(tokenizer: Tokenizer, candidates: list[str]) -> int | None:
    for spelling in candidates:
        token_id = tokenizer.token_to_id(spelling)
        if token_id is not None:
            return int(token_id)
    return None


def make_prompt(record: dict[str, Any]) -> tuple[str, str]:
    instruction = record.get("instruction", record.get("prompt", ""))
    extra_input = record.get("input", "")
    response = record.get(
        "output",
        record.get("response", record.get("completion", "")),
    )
    if not isinstance(instruction, str) or not isinstance(response, str):
        raise ValueError("Each JSONL row must contain string instruction/prompt and output/response fields.")
    if not instruction.strip() or not response.strip():
        raise ValueError("Each JSONL row must have a non-empty instruction and response.")

    prompt = f"### Instruction:\n{instruction.strip()}\n"
    if isinstance(extra_input, str) and extra_input.strip():
        prompt += f"### Input:\n{extra_input.strip()}\n"
    prompt += "### Response:\n"
    return prompt, response.strip()


def encode_jsonl(
    jsonl_path: Path,
    tokenizer: Tokenizer,
    max_sequence_length: int,
    pad_id: int,
    eos_id: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode JSONL IFT rows; labels are ignored for the prompt and padding."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Instruction dataset not found: {jsonl_path}")

    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    max_total_length = max_sequence_length + 1

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                prompt_text, answer_text = make_prompt(record)
            except Exception as exc:
                raise ValueError(f"Invalid IFT row at {jsonl_path}:{line_number}: {exc}") from exc

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids
            answer_ids = tokenizer.encode(answer_text, add_special_tokens=False).ids
            if eos_id is not None:
                answer_ids.append(eos_id)
            if not answer_ids:
                raise ValueError(f"Empty tokenized response at {jsonl_path}:{line_number}")

            # Preserve at least one response token. If the prompt is too long,
            # trim its left side so the response header/end of prompt survives.
            prompt_ids = prompt_ids[-max_sequence_length:]
            answer_ids = answer_ids[: max_total_length - len(prompt_ids)]
            token_ids = prompt_ids + answer_ids

            # The prompt is context, not a prediction target for IFT. The loss is
            # applied only to the response tokens and any appended EOS token.
            labels = token_ids.copy()
            labels[: len(prompt_ids)] = [IGNORE_INDEX] * len(prompt_ids)

            if len(token_ids) < max_total_length:
                missing = max_total_length - len(token_ids)
                token_ids.extend([pad_id] * missing)
                labels.extend([IGNORE_INDEX] * missing)
            else:
                token_ids = token_ids[:max_total_length]
                labels = labels[:max_total_length]

            # The model predicts token t+1 from input token t. Therefore inputs
            # and targets both have max_sequence_length entries, already shifted.
            x = np.asarray(token_ids[:-1], dtype=np.int32)
            y = np.asarray(labels[1:], dtype=np.int32)
            if not np.any(y != IGNORE_INDEX):
                raise ValueError(
                    f"No response targets remain after truncation at {jsonl_path}:{line_number}. "
                    "Shorten the prompt or increase --sequence-length."
                )
            all_inputs.append(x)
            all_targets.append(y)

    if not all_inputs:
        raise ValueError(f"No valid JSONL examples found in {jsonl_path}")

    return np.stack(all_inputs), np.stack(all_targets)


def make_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=max(1, len(inputs)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def adapter_files(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "adapter.npz", output_dir / "adapter_manifest.json"


def save_adapter_weights(
    adapter_variables: list[Any],
    output_dir: Path,
    global_step: int,
    rank: int,
    alpha: int,
) -> None:
    adapter_path, manifest_path = adapter_files(output_dir)
    payload = {f"arr_{i:05d}": variable.numpy() for i, variable in enumerate(adapter_variables)}
    temp_path = adapter_path.with_suffix(".npz.tmp")
    with temp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temp_path, adapter_path)

    manifest = {
        "format": "keras_int4_lora_adapter_v1",
        "global_step": int(global_step),
        "rank": int(rank),
        "alpha": int(alpha),
        "num_adapter_tensors": len(adapter_variables),
        "variables": [
            {"index": i, "path": variable_path(variable), "shape": [int(d) for d in variable.shape]}
            for i, variable in enumerate(adapter_variables)
        ],
        "note": "Base model weights are stored in the separate pretraining checkpoint.",
    }
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temp_manifest, manifest_path)
    print(f"[save] adapter tensors saved to {adapter_path} (step={global_step})")


def load_adapter_weights(adapter_variables: list[Any], output_dir: Path) -> int:
    adapter_path, manifest_path = adapter_files(output_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("num_adapter_tensors", -1)) != len(adapter_variables):
            raise ValueError(
                "Adapter tensor count does not match this model. Make sure the same "
                "model architecture, rank, and target-layer policy are being used."
            )
        expected_variables = manifest.get("variables", [])
        for i, variable in enumerate(adapter_variables):
            if i < len(expected_variables):
                expected_shape = tuple(expected_variables[i].get("shape", []))
                if tuple(variable.shape) != expected_shape:
                    raise ValueError(
                        f"Adapter shape mismatch for {variable_path(variable)}: "
                        f"model={tuple(variable.shape)}, file={expected_shape}"
                    )
    else:
        manifest = {}

    with np.load(adapter_path, allow_pickle=False) as data:
        if len(data.files) != len(adapter_variables):
            raise ValueError(
                f"Adapter file has {len(data.files)} tensors; model expects {len(adapter_variables)}."
            )
        for i, variable in enumerate(adapter_variables):
            key = f"arr_{i:05d}"
            if key not in data:
                raise ValueError(f"Missing {key} in adapter file {adapter_path}")
            value = data[key]
            if tuple(value.shape) != tuple(variable.shape):
                raise ValueError(
                    f"Adapter shape mismatch for {variable_path(variable)}: "
                    f"model={tuple(variable.shape)}, file={tuple(value.shape)}"
                )
            variable.assign(value)

    print(f"[resume] loaded adapter tensors from {adapter_path}")
    return int(manifest.get("global_step", 0))


def masked_loss_sum(logits: tf.Tensor, targets: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    valid = tf.not_equal(targets, IGNORE_INDEX)
    safe_targets = tf.where(valid, targets, tf.zeros_like(targets))
    token_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=safe_targets,
        logits=tf.cast(logits, tf.float32),
    )
    valid_float = tf.cast(valid, token_losses.dtype)
    return tf.reduce_sum(token_losses * valid_float), tf.reduce_sum(valid_float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4-bit LoRA-style IFT for DecoderOnlyTransformer")
    parser.add_argument("--base-checkpoint-dir", default="checkpoints/decoder_pretraining")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--train-jsonl", default="data/ift/train.jsonl")
    parser.add_argument("--validation-jsonl", default="data/ift/validation.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/ift_qlora")

    # These must match the architecture used to create the base checkpoint.
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--intermediate-dim", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=256)

    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no step limit")
    parser.add_argument("--save-every-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-adapter", action="store_true", help="Load output-dir/adapter.npz before training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rank < 1 or args.alpha < 1:
        raise ValueError("--rank and --alpha must both be positive integers")
    if args.sequence_length < 8:
        raise ValueError("--sequence-length is too small; use at least 8")

    configure_gpu()
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer JSON not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = find_special_token_id(
        tokenizer,
        ["<|endoftext|>", "</s>", "<eos>", "<|eot_id|>", "[EOS]"],
    )
    pad_id = find_special_token_id(tokenizer, ["<pad>", "[PAD]", "<|pad|>"])
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    if eos_id is None:
        print("[data] no common EOS token found in tokenizer.json; responses will not get an appended EOS token")

    train_x, train_y = encode_jsonl(
        Path(args.train_jsonl), tokenizer, args.sequence_length, pad_id, eos_id
    )
    val_x, val_y = encode_jsonl(
        Path(args.validation_jsonl), tokenizer, args.sequence_length, pad_id, eos_id
    )
    train_ds = make_dataset(train_x, train_y, args.batch_size, True, args.seed)
    val_ds = make_dataset(val_x, val_y, args.batch_size, False, args.seed)
    print(f"[data] train examples={len(train_x):,}; validation examples={len(val_x):,}")

    model = build_and_restore_base_model(args)
    adapter_variables = prepare_int4_lora(model, args.rank, args.alpha)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    adapter_path, _ = adapter_files(output_dir)
    if args.resume_adapter and adapter_path.exists():
        global_step = load_adapter_weights(adapter_variables, output_dir)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        global_clipnorm=1.0,
    )

    @tf.function(reduce_retracing=True)
    def train_step(input_ids: tf.Tensor, targets: tf.Tensor):
        with tf.GradientTape() as tape:
            logits = model(input_ids, training=True)
            loss_sum, valid_count = masked_loss_sum(logits, targets)
            loss = loss_sum / tf.maximum(valid_count, 1.0)
        gradients = tape.gradient(loss, adapter_variables)
        pairs = [(grad, variable) for grad, variable in zip(gradients, adapter_variables) if grad is not None]
        if not pairs:
            raise RuntimeError("No LoRA gradients were produced. Check layer support and training data.")
        optimizer.apply_gradients(pairs)
        return loss, loss_sum, valid_count

    def evaluate() -> float:
        total_loss = 0.0
        total_tokens = 0.0
        for batch_x, batch_y in val_ds:
            logits = model(batch_x, training=False)
            loss_sum, valid_count = masked_loss_sum(logits, batch_y)
            total_loss += float(loss_sum.numpy())
            total_tokens += float(valid_count.numpy())
        if total_tokens == 0:
            return float("nan")
        return total_loss / total_tokens

    stop_training = False
    for epoch in range(1, args.epochs + 1):
        loss_total = 0.0
        tokens_total = 0.0
        batch_count = 0
        for batch_x, batch_y in train_ds:
            loss, loss_sum, valid_count = train_step(batch_x, batch_y)
            global_step += 1
            loss_total += float(loss_sum.numpy())
            tokens_total += float(valid_count.numpy())
            batch_count += 1

            if global_step == 1 or global_step % 20 == 0:
                mean_loss = loss_total / max(tokens_total, 1.0)
                print(
                    f"[train] epoch={epoch}/{args.epochs} step={global_step} "
                    f"batch_loss={float(loss.numpy()):.4f} epoch_loss={mean_loss:.4f}"
                )

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_adapter_weights(adapter_variables, output_dir, global_step, args.rank, args.alpha)

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break

        val_loss = evaluate()
        train_loss = loss_total / max(tokens_total, 1.0)
        val_ppl = float(np.exp(min(val_loss, 20.0))) if np.isfinite(val_loss) else float("nan")
        print(
            f"[epoch] epoch={epoch} batches={batch_count} train_loss={train_loss:.4f} "
            f"validation_loss={val_loss:.4f} validation_ppl={val_ppl:.2f}"
        )
        save_adapter_weights(adapter_variables, output_dir, global_step, args.rank, args.alpha)
        if stop_training:
            break

    print(f"[done] adapter directory: {output_dir.resolve()}")
    print("[done] The original pretraining checkpoint is unchanged; only adapter.npz was trained/saved.")
    print("[done] For a resumed run, add --resume-adapter (optimizer state restarts).")


if __name__ == "__main__":
    main()
