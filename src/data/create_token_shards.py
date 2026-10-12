from pathlib import Path
import numpy as np

from src.data.manifest import DatasetManifest
from src.data.text_stream import TextStream
from src.data.tokenizer_dataset import TokenizerDataset



SHARD_SIZE = 10_000_000



def main():


    manifest = DatasetManifest(
        "configs/dataset_manifest.json"
    )


    tokenizer = TokenizerDataset(
        manifest.tokenizer_path
    )


    output = Path(
        "data/processed/tokens"
    )


    output.mkdir(
        parents=True,
        exist_ok=True
    )


    buffer=[]

    shard_id=0



    stream = TextStream(
        manifest.files
    )



    for text in stream:


        ids = tokenizer.encode(
            text
        )


        buffer.extend(ids)



        while len(buffer)>=SHARD_SIZE:


            shard = buffer[:SHARD_SIZE]


            buffer = buffer[
                SHARD_SIZE:
            ]


            np.save(

                output /
                f"shard_{shard_id:05d}.npy",

                np.array(
                    shard,
                    dtype=np.int32
                )

            )


            print(
                "Saved shard",
                shard_id
            )


            shard_id +=1



    if buffer:


        np.save(

            output /
            f"shard_{shard_id:05d}.npy",

            np.array(
                buffer,
                dtype=np.int32
            )

        )



if __name__=="__main__":
    main()
