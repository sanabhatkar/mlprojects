import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:

            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")

            logging.info("Loading model file")
            model = load_object(model_path)
            logging.info("Loading preprocessor file")
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            logging.debug(f"Scaled Data:{data_scaled}")

            predictions = model.predict(data_scaled)
            logging.debug(f"predicted values:{predictions}")

            return predictions
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity: str,
                 parental_level_of_education:str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_df(self):
        try:
            data_dict = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }
            logging.info("Returning data frame")
            print(f"data dictionary:{data_dict}")

            data_df = pd.DataFrame(data_dict)
            print(f"data frame:{data_df}")
            return data_df

        except Exception as e:
            raise CustomException(e, sys)