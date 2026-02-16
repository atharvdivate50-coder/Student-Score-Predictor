# Import libraries
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score


# ---------------------------------
# STEP 1: Load dataset
# ---------------------------------
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data", "student_data.csv")

df = pd.read_csv(data_path)


# ---------------------------------
# STEP 2: Input features
# ---------------------------------
X = df[[
    "study_hours",
    "previous_score",
    "technical_skill",
    "assignments_completed",
    "attendance"
]]

# Outputs
y_score = df["final_score"]   # for Linear Regression
y_pass = df["pass"]           # for Logistic Regression


# ---------------------------------
# STEP 3: Split dataset
# ---------------------------------
X_train, X_test, y_score_train, y_score_test = train_test_split(
    X, y_score, test_size=0.2, random_state=42
)

_, _, y_pass_train, y_pass_test = train_test_split(
    X, y_pass, test_size=0.2, random_state=42
)


# ---------------------------------
# STEP 4: Train Linear Regression
# Predict exam score
# ---------------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_score_train)

# Predict on test data
score_predictions = linear_model.predict(X_test)

# Check error
mae = mean_absolute_error(y_score_test, score_predictions)
print("Score Prediction MAE:", round(mae, 2))


# ---------------------------------
# STEP 5: Train Logistic Regression
# Predict Pass/Fail
# ---------------------------------
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_pass_train)

# Predict
pass_predictions = logistic_model.predict(X_test)

accuracy = accuracy_score(y_pass_test, pass_predictions)
print("Pass/Fail Accuracy:", round(accuracy * 100, 2), "%")


# ---------------------------------
# STEP 6: Save trained models
# ---------------------------------
model_path1 = os.path.join(current_dir, "..", "models", "score_model.pkl")
model_path2 = os.path.join(current_dir, "..", "models", "pass_model.pkl")

joblib.dump(linear_model, model_path1)
joblib.dump(logistic_model, model_path2)

print("Models saved successfully!")
