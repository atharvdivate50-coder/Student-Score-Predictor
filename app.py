# -------------------------------------------
# Flask Web Application for Student Predictor
# -------------------------------------------

# Import libraries
from flask import Flask, render_template, request
import joblib
import os
import pandas as pd


# ---------------------------------
# STEP 1: Create Flask app
# ---------------------------------
app = Flask(__name__)


# ---------------------------------
# STEP 2: Load trained models
# ---------------------------------

# Load models from models folder (inside project)
score_model = joblib.load("models/score_model.pkl")
pass_model = joblib.load("models/pass_model.pkl")


# ---------------------------------
# STEP 3: Home page
# ---------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------
# STEP 4: Prediction logic
# ---------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # Get values from form
    study_hours = float(request.form["study_hours"])
    previous_score = float(request.form["previous_score"])
    technical_skill = float(request.form["technical_skill"])
    assignments_completed = float(request.form["assignments_completed"])
    attendance = float(request.form["attendance"])

    # Convert into dataframe (same format as training)
    student_data = pd.DataFrame([{
        "study_hours": study_hours,
        "previous_score": previous_score,
        "technical_skill": technical_skill,
        "assignments_completed": assignments_completed,
        "attendance": attendance
    }])

    # Predict score
    predicted_score = score_model.predict(student_data)[0]

    # Predict pass/fail
    predicted_pass = pass_model.predict(student_data)[0]
    pass_probability = pass_model.predict_proba(student_data)[0][1]

    result = "PASS" if predicted_pass == 1 else "FAIL"

    return render_template(
        "index.html",
        prediction_score=round(predicted_score, 2),
        prediction_result=result,
        probability=round(pass_probability * 100, 2)
    )


# ---------------------------------
# STEP 5: Run server
# ---------------------------------
if __name__ == "__main__":
    app.run(debug=True)
