# Import required libraries
import pandas as pd
import random
import os

# List to store all student records
data = []

# We will generate data for 200 students
for i in range(200):

    # Study hours per day (0 to 12 hrs)
    study_hours = round(random.uniform(0, 12), 2)

    # Previous exam score (30 to 100)
    previous_score = random.randint(30, 100)

    # Technical skill level (1 to 10)
    technical_skill = random.randint(1, 10)

    # Assignments completed (0 to 10)
    assignments = random.randint(0, 10)

    # Attendance percentage (40% to 100%)
    attendance = random.randint(40, 100)

    # Calculate final score using a logical formula
    # (more study hours + good past performance + skills = higher marks)
    # More realistic score calculation
    final_score = (
        study_hours * 3
        + previous_score * 0.25
        + technical_skill * 2
        + assignments * 1.5
        + attendance * 0.15
        - random.randint(0, 25)   # performance loss / exam difficulty
    )


    # Keep score between 0 and 100
    final_score = max(0, min(100, round(final_score, 2)))

    # Pass/Fail condition
    # Student passes if marks >= 40
    passed = 1 if final_score >= 40 else 0

    # Add this student's data to list
    data.append([
        study_hours,
        previous_score,
        technical_skill,
        assignments,
        attendance,
        final_score,
        passed
    ])

# Create pandas DataFrame (table format)
df = pd.DataFrame(data, columns=[
    "study_hours",
    "previous_score",
    "technical_skill",
    "assignments_completed",
    "attendance",
    "final_score",
    "pass"
])

# Create correct file path to save dataset
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data", "student_data.csv")

# Save dataset as CSV file
df.to_csv(data_path, index=False)

print("Dataset created successfully!")
