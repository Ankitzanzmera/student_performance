import logging
import os
from datetime import datetime



dir_name = f"{datetime.now().strftime('%d_%m_%Y')}"  ## Directory Name
log_dir_path = os.path.join(os.getcwd(),'logs',dir_name) ## Directory path
os.makedirs(log_dir_path,exist_ok=True) # make Directory

file_name = f"{datetime.now().strftime('%H_%M_%S.log')}"  ## create file name
LOG_FILE_PATH = os.path.join(log_dir_path,file_name)   ## Log file path

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level= logging.INFO,
    format = "[ %(asctime)s ] %(lineno)d - %(name)s - %(levelname)s - %(message)s"
)