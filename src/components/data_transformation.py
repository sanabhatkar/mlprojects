import os
import sys
from src.logger import logging
from src.exception import CustomException

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
    #target_col = "math_score"
    #numerical_cols = ['reading_score', 'writing_score']
    #categorical_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']


class DataTransformation:
    def __init__(self):
        self.data_transformer_config = DataTransformationConfig()
        self.target_col = "math_score"
        self.numerical_cols = ['reading_score', 'writing_score']
        self.categorical_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

    def get_data_transformer_obj(self):
        try:
            '''
            This function is used to create and apply pipelines for numerical and categorical variables
            '''
            #numerical_vars = ['reading_score', 'writing_score']
            #categorical_vars = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            #define the pipelines
            logging.info("Building numerical pipeline")
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Building categorical pipeline")
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            #combine the pipelines using column transformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, self.numerical_cols),
                    ("cat_pipeline", cat_pipeline, self.categorical_cols)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Completed reading test and train data into dataframes")

            preprocessor_obj = self.get_data_transformer_obj()

            input_feature_train_df = train_df.drop(columns=[self.target_col], axis=1)
            logging.info(f"input_feature_train_df:\n{input_feature_train_df.head(2)}")
            target_feature_train_df = train_df[self.target_col]

            input_feature_test_df = test_df.drop(columns=[self.target_col], axis=1)
            logging.info(f"input_feature_test_df:\n{input_feature_test_df.head(2)}")
            target_feature_test_df = test_df[self.target_col]

            logging.info("Applying preprocessor object on train & test df")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformer_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Completed saving the transformer object")

            return(
                train_arr,
                test_arr,
                self.data_transformer_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)