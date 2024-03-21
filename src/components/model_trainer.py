import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from src.utils import (save_object, evaluate_models)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.trained_model_config = ModelTrainerConfig()

    def initiate_model_training(self, train_data_arr, test_data_arr):
        try:
            logging.info("Started with model training")
            X_train, y_train, X_test, y_test = (train_data_arr[:,:-1],
                                                train_data_arr[:,-1],
                                                test_data_arr[:,:-1],
                                                test_data_arr[:,-1])
            logging.info("Completed reading train and test splits")

            models = {
                "Linear Regression":LinearRegression(),
                "Gradientboost Regressor":GradientBoostingRegressor(),
                "Adaboost Regressor":AdaBoostRegressor(),
                "RandomForest Regressor": RandomForestRegressor(),
                "Decision Tree Regressor":DecisionTreeRegressor()
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys()) [ list(model_report.values()).index(best_model_score) ]

            logging.info(f"best_model_score={best_model_score}")
            logging.info(f"best_model_name={best_model_name}")

            best_model = models[best_model_name]

            logging.info(f"best_model = {models[best_model_name]}")

            if best_model_score <= 0.6:
                raise Exception("Not an ideal model")
                logging.info("Best model is not ideal for prediction.")

            save_object(
                file_path=self.trained_model_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            raise CustomException(e, sys)