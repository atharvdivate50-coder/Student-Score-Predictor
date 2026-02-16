# Import required libraries
import pandas as pd
import os
from sklearn.model_selection import train_test_split


# ---------------------------------
# STEP 1: Load dataset using proper path
# ---------------------------------
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data", "student_data.csv")

df = pd.read_csv(data_path)


# ---------------------------------
# STEP 2: Select input features (X)
# These are factors affecting performance
# ---------------------------------
X = df[[
    "study_hours",
    "previous_score",
    "technical_skill",
    "assignments_completed",
    "attendance"
]]


# ---------------------------------
# STEP 3: Select outputs (y)
# We have two outputs:
# 1) final_score (for Linear Regression)
# 2) pass (for Logistic Regression)
# ---------------------------------
y_score = df["final_score"]
y_pass = df["pass"]


# ---------------------------------
# STEP 4: Split dataset into training and testing
# 80% training, 20% testing
# ---------------------------------
X_train, X_test, y_score_train, y_score_test = train_test_split(
    X, y_score, test_size=0.2, random_state=42
)

_, _, y_pass_train, y_pass_test = train_test_split(
    X, y_pass, test_size=0.2, random_state=42
)


# ---------------------------------
# STEP 5: Print dataset sizes
# ---------------------------------
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
