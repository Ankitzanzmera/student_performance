import os,sys
sys.path.append(os.getcwd())
from dotenv import load_dotenv
import pickle
import pymysql
import pandas as pd
from sklearn.metrics import r2_score
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

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
    

def save_object(filepath:str,obj:object):

    try: 
        dir_name = os.path.dirname(filepath)
        os.makedirs(dir_name,exist_ok=True)

        with open(filepath,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(models_list,X_train,y_train,X_test,y_test):
    try:
        report = {}
        for i in range(len(models_list)):
            model = list(models_list.values())[i]
            parameters = get_params(list(models_list.keys())[i])

            grid_search = RandomizedSearchCV(model,parameters,verbose=False,n_iter=5)
            grid_search.fit(X_train,y_train)
            best_params = grid_search.best_params_

            model.set_params(**grid_search.best_params_)  
            ## Double asterick means grid search will gives best params in Dict so ** will unpack that dictionary, like if {'A':50} the A = 50
            model.fit(X_train,y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_data_score = r2_score(y_train,y_train_pred)
            test_data_score = r2_score(y_test,y_test_pred)

            report[list(models_list.keys())[i]] = [test_data_score,model,best_params]
            print(f'{list(models_list.keys())[i]} has done')
        
        return report

    except Exception as e:
        raise CustomException(e,sys)
    
def get_params(model_name:str):
    params = {
        "Linear_Regression": {},
        "Ridge": {
            'alpha': [0.01, 0.1, 1.0, 10.0],
        },
        "Lasso": {
            'alpha': [0.01, 0.1, 1.0, 10.0],
        },
        "ElasticNet": {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9],
        },
        "Decision_Tree": {
            'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        "SVM": {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto'] + [0.01, 0.1, 1.0],
        },
        "Random_forest": {
            'n_estimators': [8, 16, 32, 64, 128, 256],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
        },
        "AdaBoost": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
        },
        "Gradient_Boosting": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        "Neighbors": {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        },
        "XGB": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 4, 5],
            'min_child_weight': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
        },
        "catboost": {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5, 7],
        }
    }

    return params[model_name]


