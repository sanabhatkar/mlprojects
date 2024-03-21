import os
import sys

import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(obj, file_path):
    logging.info("Entered save_object()")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info("Exiting save_object()")
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models:dict):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            logging.info(f"Fitting the model --> {model_name}")

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred  = model.predict(X_test)

            train_R2_score = r2_score(y_train, y_train_pred)
            test_R2_score = r2_score(y_test, y_test_pred)

            logging.info(f"test score = {test_R2_score}")

            report[list(models.keys())[i]] = test_R2_score
        
        return report

    except Exception as e:
        raise CustomException(e, sys)