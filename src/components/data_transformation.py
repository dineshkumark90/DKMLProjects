import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_datatransformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ["reading score", "writing score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("numerical columns standard scaling done")
            logging.info("categorical columns encoding done")

            preprocessor = ColumnTransformer(
                [
                ("numerical_pipeline", num_pipeline, numerical_columns),
                ("categorical_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("train and test data read complete")

            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_datatransformer_object()
            target_col_name = "math score"
            numerical_columns = ["reading score", "writing score"]

            input_train_df = train_df.drop(columns=[target_col_name], axis=1)
            target_train_df = train_df[target_col_name]

            input_test_df = test_df.drop(columns=[target_col_name], axis=1)
            target_test_df = test_df[target_col_name]

            logging.info(
                f"Applying Preprocessing object on train and test df."
            )

            input_train_arr = preprocessing_obj.fit_transform(input_train_df)
            input_test_arr = preprocessing_obj.transform(input_test_df)

            train_arr = np.c_[input_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_test_arr, np.array(target_test_df)]

            logging.info(f"saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)

            



    
    

