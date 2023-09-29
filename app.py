import os,sys
sys.path.append(os.getcwd())
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation




if __name__ == "__main__":
    logging.info('Execution has Started')

    try:
        data_ingestion = DataIngestion()

        train_data_path,test_data_path = data_ingestion.initiate_ingestion()
        logging.info('Data Ingestion Completed')
        
        data_transformation = DataTransformation()
        data_transformation.initiate_transformation(train_data_path,test_data_path)
        logging.info('Data Transformation has completed')

    except Exception as e:
        raise CustomException(e,sys)