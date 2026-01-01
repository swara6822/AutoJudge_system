import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import joblib
import os

# Load dataset

DATA_PATH = "data/problems.csv"
df = pd.read_csv(DATA_PATH)
# Drop rows with missing critical values
df = df.dropna(subset=["Solved", "Problem Rating"])

print("Dataset loaded successfully")
print(df.head())

# Basic preprocessing

# Fill missing tag values with empty string
tag_cols = ["Problem_tag_1", "Problem_tag_2", "Problem_tag_3", "Problem_tag_4"]
for col in tag_cols:
    df[col] = df[col].fillna("")

# Combine tags into single text feature
df["tags_text"] = (
    df["Problem_tag_1"] + " " +
    df["Problem_tag_2"] + " " +
    df["Problem_tag_3"] + " " +
    df["Problem_tag_4"]
)

# Convert Solved column to numeric (remove 'x')
df["Solved"] = (
    df["Solved"]
    .astype(str)
    .str.replace("x", "", regex=False)
    .astype(int)
)

# Create difficulty labels (classification)

def get_difficulty(rating):
    if rating <= 1200:
        return "Easy"
    elif rating <= 1800:
        return "Medium"
    else:
        return "Hard"

df["difficulty"] = df["Problem Rating"].apply(get_difficulty)

# Feature extraction (TF-IDF)

tfidf = TfidfVectorizer()
X_tags = tfidf.fit_transform(df["tags_text"])

# Add solved count as numeric feature
X = np.hstack((X_tags.toarray(), df[["Solved"]].values))

# Targets

# Classification target
label_encoder = LabelEncoder()
y_class = label_encoder.fit_transform(df["difficulty"])

# Regression target
y_reg = df["Problem Rating"]

# Train-test split

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# Train models

# Classification model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_class_train)

# Regression model
reg = LinearRegression()
reg.fit(X_train, y_reg_train)

# Evaluation

# Classification
y_class_pred = clf.predict(X_test)
acc = accuracy_score(y_class_test, y_class_pred)

# Regression
y_reg_pred = reg.predict(X_test)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)


print(f"Classification Accuracy: {acc:.4f}")
print(f"Regression MAE: {mae:.2f}")
print(f"Regression RMSE: {rmse:.2f}")

# Save models

os.makedirs("models", exist_ok=True)

joblib.dump(clf, "models/classifier.pkl")
joblib.dump(reg, "models/regressor.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("Models saved successfully in /models")
