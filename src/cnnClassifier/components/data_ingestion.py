import os
import zipfile
import gdown
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Fetch data from the URL and save it locally.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = str(self.config.local_data_file)  # Convert Path to string
            os.makedirs(self.config.root_dir, exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, zip_download_dir, quiet=False)  # Pass string path

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
            return zip_download_dir
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            raise e

    def extract_zip_file(self):
        """
        Extract the zip file into the data directory.
        """
        try:
            unzip_path = str(self.config.unzip_dir)  # Convert Path to string
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(str(self.config.local_data_file), 'r') as zip_ref:  # Convert Path to string
                zip_ref.extractall(unzip_path)
            logger.info(f"Files extracted successfully to {unzip_path}")
        except FileNotFoundError as e:
            logger.error(f"Error: {e}. Please check if the zip file exists at the specified path.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise e
