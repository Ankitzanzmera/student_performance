import os
from pathlib import Path
import logging

logging.basicConfig(level = logging.INFO)

list_of_file = [
    ".git/workflow/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_injestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_monitoring.py",
    "src/pipelines/__init__.py",
    "src/pipelines/training_pipeline.py",    
    "src/pipelines/prediction_pipeline.py",
    "src/logger.py",
    "src/exception.py",
    "src/utils.py"
    "app.py",
    "Dockerfile",
    "requirements.txt"
    "setup.py"
]

for filepath in list_of_file:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)

    
    ## Creating Directories
    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f'directory is created {filedir}')


    ## Creating Files
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'wb') as f:
            pass
        logging.info(f'File is created {filepath}')
    else:
        logging.info(f'file already exists')