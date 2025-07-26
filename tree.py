# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("path/to/your/dataset.csv")  # Replace with actual path

# Preprocessing
X = df.drop('species', axis=1)  # Replace 'species' with your target column name
y = LabelEncoder().fit_transform(df['species'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
