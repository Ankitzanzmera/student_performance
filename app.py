import os,sys
sys.path.append(os.getcwd())
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion


if __name__ == "__main__":
    logging.info('Execution has Started')

    try:
        dataingestion = DataIngestion()

        train_data_path,test_data_path = dataingestion.initiate_ingestion()
        print(train_data_path,test_data_path)

    except Exception as e:
        raise CustomException(e,sys)