import sys,os
sys.path.append(os.getcwd())
from src.logger import logging
from src.exception import CustomException
from src.utils import read_data_from_sql
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    os.makedirs(os.path.join(os.getcwd(),'artifacts'),exist_ok=True)
    raw_data_path = os.path.join('artifacts','raw.csv')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_ingestion(self):
        try:
            df = read_data_from_sql()
            logging.info('Reading Completed From Mysql Database')

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)


            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)



        except Exception as e:
            raise CustomException(e,sys)


