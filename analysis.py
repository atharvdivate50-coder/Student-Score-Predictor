# Import required libraries
import pandas as pd
import os

# ---------------------------------
# STEP 1: Find correct project path
# ---------------------------------
# __file__ = location of this script
current_dir = os.path.dirname(__file__)

# Go one folder up → then open data folder → open csv
data_path = os.path.join(current_dir, "..", "data", "student_data.csv")


# ---------------------------------
# STEP 2: Load dataset
# ---------------------------------
df = pd.read_csv(data_path)


# ---------------------------------
# STEP 3: Show first 5 rows
# ---------------------------------
print("First 5 rows of dataset:")
print(df.head())

print("\n----------------------\n")


# ---------------------------------
# STEP 4: Dataset information
# ---------------------------------
print("Dataset Info:")
print(df.info())

print("\n----------------------\n")


# ---------------------------------
# STEP 5: Count Pass vs Fail
# ---------------------------------
print("Pass/Fail count:")
print(df["pass"].value_counts())
