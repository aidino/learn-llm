from typing import Optional, Tuple
from typing import Annotated

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from zenml import ArtifactConfig, pipeline, step, log_metadata
from zenml import save_artifact, load_artifact
from zenml.client import Client


@step
def versioned_data_loader_step() -> (
    Annotated[
        Tuple[np.ndarray, np.ndarray],
        ArtifactConfig(
            name="my_dataset",
            tags=["digits", "computer vision", "classification"],
        ),
    ]
):
    """Loads the digits dataset as a tuple of flattened numpy arrays."""
    digits = load_digits()
    return (digits.images.reshape((len(digits.images), -1)), digits.target)


@step
def model_finetuner_step(
    model: ClassifierMixin, dataset: Tuple[np.ndarray, np.ndarray]
) -> Annotated[
    ClassifierMixin,
    ArtifactConfig(name="my_model", tags=["SVC", "trained"]),
]:
    """Finetunes a given model on a given dataset."""
    model.fit(dataset[0], dataset[1])
    accuracy = model.score(dataset[0], dataset[1])
    log_metadata(metadata={"accuracy": float(accuracy)})
    return model


@pipeline
def model_finetuning_pipeline(
    dataset_version: Optional[str] = None,
    model_version: Optional[str] = None,
):
    client = Client()
    # Either load a previous version of "my_dataset" or create a new one
    if dataset_version:
        dataset = client.get_artifact_version(
            name_id_or_prefix="my_dataset", version=dataset_version
        )
    else:
        dataset = versioned_data_loader_step()

    # Load the model to finetune
    # If no version is specified, the latest version of "my_model" is used
    model = client.get_artifact_version(
        name_id_or_prefix="my_model", version=model_version
    )

    # Finetune the model
    # This automatically creates a new version of "my_model"
    model_finetuner_step(model=model, dataset=dataset)


def main():
    # Save an untrained model as first version of "my_model"
    untrained_model = SVC(gamma=0.001)
    save_artifact(
        untrained_model, name="my_model", version="1", tags=["SVC", "untrained"]
    )

    # Create a first version of "my_dataset" and train the model on it
    model_finetuning_pipeline()

    # Finetune the latest model on an older version of the dataset
    model_finetuning_pipeline(dataset_version="1")

    # Run inference with the latest model on an older version of the dataset
    latest_trained_model = load_artifact("my_model")
    old_dataset = load_artifact("my_dataset", version="1")
    latest_trained_model.predict(old_dataset[0])


if __name__ == "__main__":
    main()