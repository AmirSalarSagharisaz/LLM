import tensorflow as tf



class LanguageModelMetrics:



    def __init__(self):

        self.loss = (
            tf.keras.metrics.Mean()
        )


        self.accuracy = (
            tf.keras.metrics.SparseCategoricalAccuracy()
        )



    def update(
        self,
        loss,
        targets,
        logits
    ):

        self.loss.update_state(
            loss
        )

        self.accuracy.update_state(
            targets,
            logits
        )



    def results(self):

        loss = (
            self.loss.result()
            .numpy()
        )


        return {

            "loss":
                loss,


            "accuracy":
                self.accuracy.result()
                .numpy(),


            "perplexity":
                tf.exp(loss)
                .numpy()

        }



    def reset(self):

        self.loss.reset_state()

        self.accuracy.reset_state()
