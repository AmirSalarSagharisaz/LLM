import time
import math
import tensorflow as tf
from src.training.losses import compute_loss
from src.training.metrics import LanguageModelMetrics
class Trainer:
    def __init__(self,
        model,
        optimizer,
        training_state,
        checkpoint_manager,
        dataset_cursor,
        validation_cursor=None,
        gradient_accumulation_steps=1,
        clip_norm=1.0,
        save_frequency=1000,
        log_frequency=100,
        tensorboard_dir="logs"
    ):

        self.model = model
        self.optimizer = optimizer
        self.training_state = training_state
        self.checkpoint_manager = checkpoint_manager
        self.dataset_cursor = dataset_cursor
        self.validation_cursor = validation_cursor
        self.gradient_accumulation_steps = (gradient_accumulation_steps)
        self.clip_norm = clip_norm
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        self.metrics = LanguageModelMetrics()
        self.validation_metrics = (LanguageModelMetrics())
        self.writer = tf.summary.create_file_writer(tensorboard_dir)
        self.use_loss_scaling = isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer)
        self.start_time = None
        self.total_parameters = (self.model.count_params())
        print(f"Model parameters: "f"{self.total_parameters:,}")

    def _current_step(self):
        return int(self.training_state.global_step.numpy())

    def _learning_rate(self):
        lr = self.optimizer.learning_rate
        if callable(lr):
            return float(lr(self.optimizer.iterations).numpy())
        return float(tf.keras.backend.get_value(lr))

    def _tokens_per_second(self):
        elapsed = (time.time()-self.start_time)
        tokens = (self.training_state.tokens_seen.numpy())
        return tokens/max(elapsed, 1)

    def _remaining_time(self):
        return "unknown"

    def _check_loss(self, loss):
        if tf.reduce_any(tf.math.is_nan(loss)):
            raise RuntimeError("NaN loss detected")

        if tf.reduce_any(tf.math.is_inf(loss)):
            raise RuntimeError("Infinite loss detected")
        
    def _compute_gradients(self, inputs, targets):
        with tf.GradientTape() as tape:
            logits = self.model(inputs, training=True)
            loss = compute_loss(targets, logits)
            self._check_loss(loss)
            if self.use_loss_scaling:
                scaled_loss = (self.optimizer.get_scaled_loss(loss))
            else:
                scaled_loss = loss
        gradients = tape.gradient(scaled_loss,self.model.trainable_variables)

        if self.use_loss_scaling:
            gradients = (self.optimizer.get_unscaled_gradients(gradients))
        return loss, logits, gradients

    def _create_gradient_buffer(self):
        with tf.device("/CPU:0"):
            return [tf.zeros_like(variable, dtype=tf.float32) for variable in self.model.trainable_variables]

    def _reset_gradient_buffer(self):
        return self._create_gradient_buffer()

    def _apply_gradients(self, gradients):
        gradient_norm = (tf.linalg.global_norm(gradients))
        if self.clip_norm:
            gradients, _ = tf.clip_by_global_norm(gradients,self.clip_norm)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return gradient_norm

    def _write_training_summary(self, loss, gradient_norm):
        step = self._current_step()
        with self.writer.as_default():
            tf.summary.scalar("train/loss", loss, step=step)
            tf.summary.scalar("train/perplexity", tf.exp(loss), step=step)
            tf.summary.scalar("train/learning_rate", self._learning_rate(), step=step)
            tf.summary.scalar("train/gradient_norm", gradient_norm, step=step)
            tf.summary.scalar("train/tokens_per_second", self._tokens_per_second(), step=step)
        self.writer.flush()



    # =====================================================
    # Single training step
    # =====================================================


    def train_step(
        self,
        inputs,
        targets,
        gradient_buffer,
        accumulation_index
    ):


        loss, logits, gradients = (
            self._compute_gradients(
                inputs,
                targets
            )
        )


        accumulation_scale = tf.cast(

            self.gradient_accumulation_steps,

            loss.dtype

        )


        gradients = [

            gradient /
            accumulation_scale

            for gradient in gradients

        ]



        #gradient_buffer = [

        #    accumulated + gradient

        #    for accumulated, gradient in zip(

        #        gradient_buffer,

        #        gradients

        #    )

        #]

        new_gradient_buffer = []


        for accumulated, gradient in zip(
            gradient_buffer,
            gradients
        ):

            with tf.device("/CPU:0"):

                accumulated = accumulated + tf.cast(
                    gradient,
                    tf.float32
                )


            new_gradient_buffer.append(
                accumulated
            )


        gradient_buffer = new_gradient_buffer



        optimizer_update = (

            (accumulation_index + 1)

            %

            self.gradient_accumulation_steps

            == 0

        )



        gradient_norm = None


        self.training_state.tokens_seen.assign_add(

            tf.cast(
                tf.size(targets),
                tf.int64
            )

        )


        if optimizer_update:


            gradient_norm = (
                self._apply_gradients(
                    gradient_buffer
                )
            )


            gradient_buffer = (
                self._reset_gradient_buffer()
            )


            self.training_state.increment_step(
                tokens=tf.size(targets)
            )



            self._write_training_summary(

                loss,

                gradient_norm

            )



        self.metrics.update(

            loss,

            targets,

            logits

        )


        return gradient_buffer



    # =====================================================
    # Flush incomplete accumulation
    # =====================================================


    def _flush_gradients(
        self,
        gradient_buffer
    ):


        gradient_norm = (
            tf.linalg.global_norm(
                gradient_buffer
            )
        )


        if gradient_norm > 0:


            self._apply_gradients(

                gradient_buffer

            )


            self.training_state.increment_step()



    # =====================================================
    # Validation
    # =====================================================


    def validate(self):


        if self.validation_cursor is None:

            return None



        print(
            "\nRunning validation..."
        )


        self.validation_metrics.reset()



        for inputs, targets in self.validation_cursor:


            logits = self.model(

                inputs,

                training=False

            )


            loss = compute_loss(

                targets,

                logits

            )


            self.validation_metrics.update(

                loss,

                targets,

                logits

            )



        results = (
            self.validation_metrics.results()
        )


        step = self._current_step()



        with self.writer.as_default():


            tf.summary.scalar(

                "validation/loss",

                results["loss"],

                step=step

            )


            tf.summary.scalar(

                "validation/perplexity",

                results["perplexity"],

                step=step

            )


        self.writer.flush()



        print(
            "Validation:"
        )


        print(
            results
        )


        return results



    # =====================================================
    # Epoch training
    # =====================================================


    def train_epoch(self):


        epoch = int(

            self.training_state
            .epoch
            .numpy()

        )


        print(
            f"\nStarting epoch {epoch}"
        )


        self.metrics.reset()



        gradient_buffer = (
            self._create_gradient_buffer()
        )



        for step, batch in enumerate(

            self.dataset_cursor

        ):


            inputs, targets = batch



            gradient_buffer = self.train_step(

                inputs,

                targets,

                gradient_buffer,

                step

            )



            global_step = (
                self._current_step()
            )



            if (

                global_step > 0

                and

                global_step %

                self.save_frequency

                == 0

            ):


                print(
                    "Saving checkpoint..."
                )


                self.checkpoint_manager.save()



            if step % self.log_frequency == 0:


                results = (
                    self.metrics.results()
                )


                print(

                    f"Epoch {epoch} "

                    f"Step {step} "

                    f"Loss={results['loss']:.4f} "

                    f"Accuracy={results['accuracy']:.4f} "

                    f"Perplexity={results['perplexity']:.4f} "

                    f"Tokens/s="
                    f"{self._tokens_per_second():.2f}"

                )



        self._flush_gradients(

            gradient_buffer

        )



        validation_results = (
            self.validate()
        )



        self.training_state.next_epoch()



        self.dataset_cursor.reset_epoch()



        results = (
            self.metrics.results()
        )


        if validation_results:

            results.update(
                validation_results
            )


        print(
            "\nEpoch completed"
        )


        print(
            results
        )


        return results



    # =====================================================
    # Full training
    # =====================================================


    def train(
    self,
    epochs
    ):

        self.start_time = time.time()


        for epoch in range(epochs):

            print(
                f"Starting epoch {epoch}"
            )


            self.metrics.reset()


            gradient_buffer = (
                self._create_gradient_buffer()
            )


            for step, (inputs, targets) in enumerate(
                self.dataset_cursor
            ):


                gradient_buffer = self.train_step(

                    inputs,

                    targets,

                    gradient_buffer,

                    step

                )


                if step % self.log_frequency == 0:

                    results = (
                        self.metrics.results()
                    )

                    print(
                        results
                    )


                if (

                    self._current_step()

                    %

                    self.save_frequency

                    ==0

                    and self._current_step()!=0

                ):

                    self.checkpoint_manager.save()



            self.dataset_cursor.reset_epoch()


            self.training_state.next_epoch()


            self.checkpoint_manager.save()