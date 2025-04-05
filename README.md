# Asteroid Detection Using Artificial Intelligence
# Author: Ayush
# Class 12 | Aspiring NASA Scientist

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------
# 1. Load Dataset
# -----------------------------------------------
file_url = "https://raw.githubusercontent.com/Ayushchand-spec/Asteroid-Detection/main/asteroid_dataset.csv"

try:
    data = pd.read_csv(file_url)
    print("Dataset loaded successfully!\n")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# -----------------------------------------------
# 2. Dataset Overview
# -----------------------------------------------
print("First 5 rows of the dataset:")
print(data.head(), "\n")

if "value" not in data.columns:
    print("Error: Required column 'value' not found in the dataset!")
    exit()

# -----------------------------------------------
# 3. Data Preprocessing & Normalization
# -----------------------------------------------
data.dropna(inplace=True)
data["normalized"] = (data["value"] - data["value"].min()) / (data["value"].max() - data["value"].min())
print("Data normalization complete.\n")

# -----------------------------------------------
# 4. Visual Analysis
# -----------------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(x=data.index, y=data["normalized"], marker="o", color='purple', label="Normalized Asteroid Data")
plt.title("Asteroid Detection - Normalized Value Visualization", fontsize=15)
plt.xlabel("Observation Index")
plt.ylabel("Normalized Value")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 5. Statistical Summary
# -----------------------------------------------
print("Descriptive Statistics for Normalized Data:")
print(data["normalized"].describe(), "\n")

# -----------------------------------------------
# 6. Conclusion
# -----------------------------------------------
print("Conclusion:")
print("- This AI-based approach normalizes asteroid datasets for improved pattern recognition.")
print("- Visualization shows how values fluctuate, assisting in anomaly and trend detection.")
print("- This foundational model can support real-time asteroid monitoring systems.\n")

# -----------------------------------------------
# 7. Future Scope
# -----------------------------------------------
print("Future Scope:")
print("- Integrate real-time APIs from NASA for dynamic data updates.")
print("- Use advanced AI (CNNs, RNNs) for image and time-series based detection.")
print("- Build an interactive dashboard for space research organizations.")
print("- Enable predictive modeling for asteroid threat estimation.\n")
