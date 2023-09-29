import os,sys
sys.path.append(os.getcwd())
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import numpy as np


if __name__ == "__main__":
    logging.info('Execution has Started')

    try:
        data_ingestion = DataIngestion()

        train_data_path,test_data_path = data_ingestion.initiate_ingestion()
        logging.info('Data Ingestion Completed')
        
        data_transformation = DataTransformation()
        train_data,test_data = data_transformation.initiate_transformation(train_data_path,test_data_path)
        logging.info('Data Transformation has completed')

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_train(train_data,test_data)
        logging.info('Modal Training has been Completed.')

    except Exception as e:
        raise CustomException(e,sys)