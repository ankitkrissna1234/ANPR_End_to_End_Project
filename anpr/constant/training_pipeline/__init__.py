import os

ARTIFACTS_DIR: str = "artifacts"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_DOWNLOAD_URL: str = (
    "https://github.com/ankitkrissna1234/Datasets_For_Projects/raw/refs/heads/main/anpr_dataset.zip"
)


"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_STATUS_FILE = "status.txt"

DATA_VALIDATION_ALL_REQUIRED_FILES = ["train", "test", "valid", "data.yaml"]


"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_PRETRAINED_WEIGHT_NAME: str = "yolo11n.pt"

MODEL_TRAINER_NO_EPOCHS: int = 10

MODEL_TRAINER_BATCH_SIZE: int = 16

MODEL_TRAINER_IMAGE_SIZE: int = 640
