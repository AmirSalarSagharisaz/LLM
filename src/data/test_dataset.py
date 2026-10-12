from src.data.manifest import DatasetManifest
from src.data.tf_dataset import TensorFlowDatasetBuilder



def main():


    manifest = DatasetManifest(
        "configs/dataset_manifest.json"
    )


    builder = TensorFlowDatasetBuilder(

        files=manifest.files,

        tokenizer_path=
            manifest.tokenizer_path,

        sequence_length=512,

        batch_size=2
    )


    dataset = builder.build()



    for x,y in dataset.take(1):

        print(
            "Input shape:",
            x.shape
        )

        print(
            "Target shape:",
            y.shape
        )

        print(
            "First tokens:",
            x[0][:20]
        )



if __name__ == "__main__":
    main()
