import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'student-mat.csv'  # Update the path if necessary
data = pd.read_csv(file_path, delimiter=';')

# Display the first few rows
print("First 5 rows of the dataset:")
print(data.head())

# --- Data Exploration ---
print("\n--- Data Exploration ---")
# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_values)

# Check data types
print("\nData Types of Each Column:")
print(data.dtypes)

# Dataset size
print("\nDataset Size:")
print(data.shape)

# --- Data Cleaning ---
print("\n--- Data Cleaning ---")
# Remove duplicates
initial_rows = data.shape[0]
data = data.drop_duplicates()
duplicates_removed = initial_rows - data.shape[0]
print(f"Number of duplicate rows removed: {duplicates_removed}")

# --- Data Analysis ---
print("\n--- Data Analysis ---")
# 1. Average score in math (G3)
average_score = data['G3'].mean()
print(f"1. Average final grade (G3): {average_score:.2f}")

# 2. Number of students scoring above 15 in G3
students_above_15 = (data['G3'] > 15).sum()
print(f"2. Number of students scoring above 15 in G3: {students_above_15}")

# 3. Correlation between study time and G3
correlation = data['studytime'].corr(data['G3'])
print(f"3. Correlation between study time and G3: {correlation:.2f}")

# 4. Gender with a higher average G3
average_scores_by_gender = data.groupby('sex')['G3'].mean()
print("4. Average G3 by gender:")
print(average_scores_by_gender)

# --- Data Visualization ---
print("\n--- Data Visualization ---")

# 1. Histogram of final grades (G3)
plt.figure(figsize=(8, 6))
plt.hist(data['G3'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Final Grades (G3)')
plt.xlabel('Final Grade (G3)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 2. Scatter plot between study time and final grade (G3)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='studytime', y='G3', data=data, hue='sex', palette='Set2')
plt.title('Study Time vs Final Grade (G3)')
plt.xlabel('Study Time (hours per week)')
plt.ylabel('Final Grade (G3)')
plt.grid(alpha=0.5)
plt.show()

# 3. Bar chart comparing average scores of male and female students
plt.figure(figsize=(8, 6))
average_scores_by_gender.plot(kind='bar', color=['blue', 'pink'])
plt.title('Average Final Grade (G3) by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Final Grade (G3)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


