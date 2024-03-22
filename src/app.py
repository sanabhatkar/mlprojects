import sys
from flask import Flask, request, render_template
import pandas as pd

from src.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import CustomException

application = Flask(__name__)

app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET","POST"])
def predictdata():
    if request.method == "GET":
        logging.info("Inside method GET")
        return render_template("home.html")
    else:
        try:
            logging.info("Inside method POST")
            custom_data_obj = CustomData(gender=request.form.get("gender"),
                            race_ethnicity=request.form.get("ethnicity"),
                            parental_level_of_education=request.form.get("parental_level_of_education"),
                            lunch=request.form.get("lunch"),
                            test_preparation_course=request.form.get("test_preparation_course"),
                            reading_score=request.form.get("reading_score"),
                            writing_score=request.form.get("writing_score"))
            
            features_df = custom_data_obj.get_data_as_df()
            print(f"features_df:{features_df}")

            prediction_pl = PredictPipeline()
            results = prediction_pl.predict(features_df)
            logging.debug(f"results:{results}")
        except Exception as e:
            raise CustomException(e, sys)

        return render_template("home.html",results = results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")