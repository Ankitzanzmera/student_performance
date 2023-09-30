import os,sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from src.logger import logging
from src.exception import CustomException

class ModelMonitoring:

    def eval_metrics(self,y_test,y_pred):
        r2 = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        rmse = np.sqrt(mean_squared_error(y_test,y_pred))

        return (rmse,mae,r2)
        

    def initiate_model_monitoring(self,best_model_name,best_model,best_params,X_test,y_test):
        
        try:
            mlflow.set_registry_uri('https://dagshub.com/Ankitzanzmera/student_performance.mlflow')
            tracking_url_type_score = urlparse(mlflow.get_tracking_uri()).scheme
            logging.info(mlflow.get_tracking_uri())
            logging.info(f'traking url : {tracking_url_type_score}')

            with mlflow.start_run():
                y_pred = best_model.predict(X_test)
                logging.info('Got prediction')

                (rmse,mae,r2) = self.eval_metrics(y_test,y_pred)
                logging.info(f'Got metrics {rmse,mae,r2}')

                mlflow.log_params(best_params)  
                logging.info(f'logged params {best_params}')

                mlflow.log_metric('RMSE',rmse)
                mlflow.log_metric('MAE',mae)
                mlflow.log_metric('R2_SCORE',r2)
                logging.info('logged metrics')

                if tracking_url_type_score != 'file':
                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model,"Model",registered_model_name=best_model_name)
                    
                else:
                    mlflow.sklearn.log_model(best_model,'Model')
                    logging.info('Error occurred')

        except Exception as e:
            CustomException(e,sys)

                


