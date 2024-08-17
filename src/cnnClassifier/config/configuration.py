import sys
sys.path.append('C:/Users/Swarnendu/Desktop/End-to-end-Chest-Cancer-Classification-using-MLflow-DVC/src')
import os
 
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)



from pathlib import Path

def create_directories(paths):
    for path in paths:
        # Convert the path to a Path object if it's not already
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # Get the artifacts root directory from the config and ensure it exists
        self.artifacts_root = self.config['artifacts_root']
        create_directories([self.artifacts_root])

        # Define data_ingestion_dir
        self.data_ingestion_dir = os.path.join(self.artifacts_root, self.config['data_ingestion']['root_dir'])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']

        # Ensure that the root directory for data ingestion exists
        create_directories([self.data_ingestion_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=self.data_ingestion_dir,
            source_URL=config['source_URL'],
            local_data_file=os.path.join(self.data_ingestion_dir, "data.zip"),  # Correct file path for ZIP file
            unzip_dir=self.data_ingestion_dir
        )

        return data_ingestion_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    


    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = Path(r'C:\Users\Swarnendu\Desktop\End-to-end-Chest-Cancer-Classification-using-MLflow-DVC\artifacts\artifacts\data_ingestion')
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=training_data,
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE
        )

        return training_config

    



    def get_evaluation_config(self) -> EvaluationConfig:
        # Define the base directory where the script is executed
        base_dir = Path.cwd()

        # Define the relative path to your model file and data directory
        model_path = base_dir / "artifacts" / "training" / "model.keras"
        training_data_path = base_dir / "artifacts" / "artifacts" / "data_ingestion" / "Chest-CT-Scan-data"

        # Ensure paths exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not training_data_path.exists():
            raise FileNotFoundError(f"Training data directory not found: {training_data_path}")

        eval_config = EvaluationConfig(
            path_of_model=str(model_path),
            training_data=str(training_data_path),
            mlflow_uri="https://dagshub.com/banik-s/End-to-end-Chest-Cancer-Classification-using-MLflow-DVC.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config


      