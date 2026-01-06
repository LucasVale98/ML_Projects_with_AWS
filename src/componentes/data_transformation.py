import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """This function is responsible for data transformation 
            object creation"""

        try:
            logging.info("Data Transformation initiated")

            numerical_columns = ['age', 'absences', 'G1', 'G2']
            categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus',
                                    'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 
                                    'guardian', 'schoolsup', 'famsup', 'paid', 
                                    'activities', 'nursery', 'higher', 
                                    'internet', 'romantic']  
           
            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            logging.info(f"Numerical pipeline created: {num_pipeline}")
          
            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])
            logging.info(f"Categorical pipeline created: {cat_pipeline}")
          
            # Combine pipelines
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            logging.info("Data Transformation completed")
            return preprocessor
       
        except Exception as e:
            logging.error("Error in get_data_transformer_object")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        """This function initiates data transformation on train 
             and test data and returns transformed arrays and preprocessor object"""
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test data read successfully")

            logging.info("Obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'G3'
            numerical_columns = ['age', 'absences', 'G1', 'G2']

            # Separate input features and target variable for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target variable for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data")

            # Apply transformation on training data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]

            # Apply transformation on testing data
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.to_numpy()]

            logging.info("Data transformation completed successfully")

            # Save the preprocessor object
            save_path = self.data_transformation_config.preprocessor_obj_file_path
            save_object(save_path, preprocessor_obj)
            logging.info(f"Preprocessor object saved at: {save_path}")
            
            return train_arr, test_arr, preprocessor_obj

        except Exception as e:
            logging.error("Error in initiate_data_transformation")
            raise CustomException(e, sys)