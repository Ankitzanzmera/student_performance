import os,sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_pickle_file_path = os.path.join(os.getcwd(),'artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.datatransformation_config = DataTransformationConfig()

    def preprocessing_pipelines(self,num_feature,cate_feature):
        '''
            Makes data preprocessing step Automate 
        '''
        self.num_feature = num_feature
        self.cate_feature = cate_feature

        try:
            num_pipeline = Pipeline(
            steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cate_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )       

            main_pipeline = ColumnTransformer(
                [
                    ('cate_pipeline',cate_pipeline,self.cate_feature),
                    ('num_pipeline',num_pipeline,self.num_feature)
                ],
                remainder='passthrough'
            )
            return main_pipeline

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_transformation(self,train_path:str,test_path:str):
        ''' 
            Will Return pickle file of preprocessor   
        '''
        try:
            train_data = pd.read_csv(os.path.join(os.getcwd(),train_path))
            test_data = pd.read_csv(os.path.join(os.getcwd(),test_path))
            logging.info('Reading the Train and Test data')

            num_feature = [feature for feature in train_data.columns if train_data[feature].dtypes != "object"]
            num_feature.remove('math score')
            cate_feature = [feature for feature in train_data.columns if train_data[feature].dtypes == "object"]
            logging.info('Dividing Numerical and categorical Features')
            preprocessor_obj = self.preprocessing_pipelines(num_feature,cate_feature)


            target_feature = 'math score'

            train_data_input_feature = train_data.drop([target_feature],axis = 1)
            train_data_target_feature = train_data[target_feature]

            test_data_input_feature = test_data.drop([target_feature],axis = 1)
            test_data_target_feature = test_data[target_feature]
            logging.info('Splitted Dependent and Independent Variable')

            # fit_transform returns array
            preprocessed_train_data = preprocessor_obj.fit_transform(train_data_input_feature)
            preprocessed_test_data = preprocessor_obj.transform(test_data_input_feature)
            logging.info('Preprocessing of Data is done')

            preprocessed_train_data = np.c_[preprocessed_train_data,train_data_target_feature]
            preprocessed_test_data = np.c_[preprocessed_test_data,test_data_target_feature]

            logging.info('Concatenation is Done')

            save_object(filepath = self.datatransformation_config.preprocessor_pickle_file_path,obj = preprocessor_obj)

            logging.info('Proprocessing object is Saved')

            return (
                preprocessed_train_data,
                preprocessed_test_data,
                )

        except Exception as e:
            raise CustomException(e,sys)




