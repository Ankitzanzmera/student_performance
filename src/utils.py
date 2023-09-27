import os,sys
sys.path.append(os.getcwd())
from src.logger import logging
from src.exception import CustomException
from dotenv import load_dotenv
import pymysql
import pandas as pd

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
database = os.getenv("db")

# print(host,"-",username,"-",password,"-",database)

def read_data_from_sql():
    logging.info('reading sql is Started')
    try:
        mydb = pymysql.connect(host=host,user=user,password=password,database=database)
        logging.info('Connection Established')
        df = pd.read_sql_query('select * from studentsperformance',mydb)

        return df


    except Exception as e:
        raise CustomException(e,sys)