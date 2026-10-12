import tensorflow as tf
from pathlib import Path
import json



class CheckpointManager:


    def __init__(
        self,
        model,
        optimizer,
        state,
        directory,
        max_to_keep=5
    ):


        self.state = state


        self.directory = Path(directory)

        self.directory.mkdir(
            parents=True,
            exist_ok=True
        )


        self.checkpoint = tf.train.Checkpoint(

            model=model,

            optimizer=optimizer,

            step=tf.Variable(
                state.global_step
            )
        )


        self.manager=tf.train.CheckpointManager(

            self.checkpoint,

            str(self.directory),

            max_to_keep=max_to_keep
        )



    def save(self):


        self.state.global_step = int(
            self.checkpoint.step.numpy()
        )


        self.state.update_checkpoint_time()


        path=self.manager.save()



        metadata_path = (

            Path(
                str(path)+".json"
            )
        )


        with open(metadata_path,"w") as f:

            json.dump(
                self.state.to_dict(),
                f,
                indent=4
            )


        print(
            f"Checkpoint saved: {path}"
        )


        return path



    def restore_latest(self):


        latest=self.manager.latest_checkpoint


        if latest is None:

            print(
                "No checkpoint found"
            )

            return False



        self.checkpoint.restore(
            latest
        ).expect_partial()



        metadata=latest+".json"



        with open(metadata) as f:

            data=json.load(f)



        for k,v in data.items():

            setattr(
                self.state,
                k,
                v
            )



        print(
            "Restored checkpoint:"
        )

        print(
            self.state
        )


        return True
