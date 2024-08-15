import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(r'C:\Users\Swarnendu\Desktop\End-to-end-Chest-Cancer-Classification-using-MLflow-DVC\src')))

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training
from cnnClassifier import logger

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()

if __name__ == '__main__':
    try:
        # Convert file paths to Path objects
        config = ConfigurationManager(
            config_filepath=Path(r'C:\Users\Swarnendu\Desktop\End-to-end-Chest-Cancer-Classification-using-MLflow-DVC\config\config.yaml'),  # Convert to Path
            params_filepath=Path(r'C:\Users\Swarnendu\Desktop\End-to-end-Chest-Cancer-Classification-using-MLflow-DVC\params.yaml')   # Convert to Path
        )
        
        # Get the training configuration
        training_config = config.get_training_config()
        
        # Initialize the Training class
        training = Training(config=training_config)
        
        # Load the base model
        training.get_base_model()
        
        # Prepare the training and validation data generators
        training.train_valid_generator()
        
        # Train the model
        training.train()

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except Exception as e:
        print(f"An error occurred: {e}")
