import os, sys
import yaml
from anpr.logger import logging
from anpr.exception import SignException
from anpr.entity.config_entity import ModelTrainerConfig
from anpr.entity.artifacts_entity import ModelTrainerArtifact


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(
        self,
    ) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            os.system("unzip anpr_dataset.zip")
            os.system("rm anpr_dataset.zip")

            os.system(
                f"yolo task=detect mode=train data=data.yaml model='{self.model_trainer_config.weight_name}' epochs={self.model_trainer_config.no_epochs} imgsz={self.model_trainer_config.image_size} workers=0"
            )
            # os.system("cp runs/detect/train/weights/best.pt  ANPR_End_to_End_Project/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            # Check if best.pt exists
            best_model_path = "runs/detect/train/weights/best.pt"
            if os.path.exists(best_model_path):
                os.system(
                    f"cp {best_model_path} {self.model_trainer_config.model_trainer_dir}/"
                )
            else:
                raise FileNotFoundError(
                    "best.pt not found. Training might have failed."
                )

            os.system("rm -r runs")
            os.system("rm -r train")
            os.system("rm -rf test")
            os.system("rm -rf valid")
            os.system("rm -rf data.yaml")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="runs/detect/train/weights/best.pt",
            )
            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise SignException(e, sys)
