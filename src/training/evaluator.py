import tensorflow as tf

from src.training.losses import compute_loss



class Evaluator:



    def __init__(
        self,
        model
    ):

        self.model=model



    def evaluate(
        self,
        dataset
    ):

        total_loss=0

        steps=0



        for x,y in dataset:


            logits = self.model(
                x,
                training=False
            )


            loss = compute_loss(
                y,
                logits
            )


            total_loss += loss

            steps +=1



        avg_loss = (
            total_loss /
            steps
        )


        return {

            "loss":
                float(avg_loss),


            "perplexity":
                float(
                    tf.exp(avg_loss)
                )

        }
