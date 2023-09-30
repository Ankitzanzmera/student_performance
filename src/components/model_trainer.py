import os,sys
sys.path.append(os.getcwd())
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import (LinearRegression,Ridge,Lasso,ElasticNet)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor)
from sklearn.neighbors import KNeighborsRegressor



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(os.getcwd(),'artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trained_config = ModelTrainerConfig()

    def initiate_model_train(self,train_data,test_data):
        try:
            
            X_train,y_train = (train_data[:,:-1],train_data[:,-1])
            X_test,y_test = (test_data[:,:-1],test_data[:,-1])
            logging.info('Data splited into Input and target variable')

            models = {
                'Linear_Regression':LinearRegression(),
                'Ridge':Ridge(),
                'Lasso':Lasso(),
                'ElasticNet':ElasticNet(),
                'Decision_Tree':DecisionTreeRegressor(),
                'SVM':SVR(),
                'Random_forest':RandomForestRegressor(),
                'AdaBoost':AdaBoostRegressor(),
                'Gradient_Boosting':GradientBoostingRegressor(),
                'Neighbors':KNeighborsRegressor(),
                'XGB':XGBRegressor(),
                'catboost':CatBoostRegressor(verbose=False) 
                }
            logging.info('Model Evaluation Started')
            model_report:dict = evaluate_model(models,X_train,y_train,X_test,y_test)
            
            best_model_score = max(model_report.values())[0]  ## will return model_score which have best accuracy
            best_model = max(model_report.values())[1]    ## Will return tuned model from model_report which have best score
            best_model_name = list(model_report.keys())[list(model_report.values()).index(max(model_report.values()))]  ## Will return name of the model which have best accuracy
            best_model_params = max(model_report.values())[2] 

            if best_model_score < 0.6:
                raise CustomException('No best model Found Which can satisfy our threshold value')
            tuned_model= max(model_report.values())[1]           
            logging.info(f'Evaluation has ended and got Best model {best_model_name}:{best_model}and its score on test data is {best_model_score}')

            save_object(self.model_trained_config.trained_model_file_path,tuned_model)

            return (best_model_name,best_model,best_model_params,X_test,y_test)

            

        except Exception as e:
            raise CustomException(e,sys)