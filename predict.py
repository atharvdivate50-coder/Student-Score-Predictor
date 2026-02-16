# ---------------------------------
# Student Exam Score Prediction
# This file loads trained models
# and predicts score + pass/fail
# ---------------------------------

# Import required libraries
import joblib
import os
import pandas as pd


# ---------------------------------
# STEP 1: Load trained models
# ---------------------------------
# Get current file location
current_dir = os.path.dirname(__file__)

# Create correct paths for saved models
score_model_path = os.path.join(current_dir, "..", "models", "score_model.pkl")
pass_model_path = os.path.join(current_dir, "..", "models", "pass_model.pkl")

# Load models
score_model = joblib.load(score_model_path)
pass_model = joblib.load(pass_model_path)


# ---------------------------------
# STEP 2: Define student input
# (You can change these values)
# ---------------------------------
study_hours = 5
previous_score = 70
technical_skill = 6
assignments_completed = 7
attendance = 80


# ---------------------------------
# STEP 3: Convert input into DataFrame
# We use same column names as training
# to avoid sklearn warning
# ---------------------------------
student_data = pd.DataFrame([{
    "study_hours": study_hours,
    "previous_score": previous_score,
    "technical_skill": technical_skill,
    "assignments_completed": assignments_completed,
    "attendance": attendance
}])


# ---------------------------------
# STEP 4: Predict final exam score
# ---------------------------------
predicted_score = score_model.predict(student_data)[0]


# ---------------------------------
# STEP 5: Predict Pass/Fail result
# ---------------------------------
predicted_pass = pass_model.predict(student_data)[0]

# Probability of passing
pass_probability = pass_model.predict_proba(student_data)[0][1]


# ---------------------------------
# STEP 6: Display results
# ---------------------------------
print("\n----- Student Prediction Result -----")

print("Predicted Score:", round(predicted_score, 2))

if predicted_pass == 1:
    print("Result: PASS")
else:
    print("Result: FAIL")

print("Probability of Passing:", round(pass_probability * 100, 2), "%")
