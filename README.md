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
file_url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/YOUR_FILE.csv"  # Replace with actual link

try:
    data = pd.read_csv(file_url)
    print("Dataset loaded successfully!\n")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# -----------------------------------------------
# 2. Data Overview
# -----------------------------------------------
print("First 5 rows of the dataset:")
print(data.head(), "\n")

if "value" not in data.columns:
    print("Error: Required column 'value' not found in the dataset!")
    exit()

# -----------------------------------------------
# 3. Data Normalization
# -----------------------------------------------
data["normalized"] = (data["value"] - data["value"].min()) / (data["value"].max() - data["value"].min())
print("Normalized data added successfully!\n")

# -----------------------------------------------
# 4. Visualization
# -----------------------------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(x=data.index, y=data["normalized"], marker="o", label="Normalized Asteroid Value")
plt.title("Asteroid Detection - Normalized Value Visualization", fontsize=14)
plt.xlabel("Index")
plt.ylabel("Normalized Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------------------------
# 5. Summary Stats
# -----------------------------------------------
print("Statistical Summary:")
print(data["normalized"].describe(), "\n")

# -----------------------------------------------
# 6. Conclusion
# -----------------------------------------------
print("Conclusion:")
print("The graph visualizes how AI can normalize asteroid data for better pattern analysis.")
print("This approach enables anomaly detection, trend visualization, and future AI-based classification.\n")

# -----------------------------------------------
# 7. Future Scope
# -----------------------------------------------
print("Future Scope:")
print("- Integrate real-time NASA APIs for live asteroid tracking")
print("- Apply deep learning (CNNs) for image-based detection")
print("- Create a web dashboard to track asteroid movement")
