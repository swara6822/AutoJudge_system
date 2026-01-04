import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import joblib
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset

DATA_PATH = "data/problems_data.jsonl"
df = pd.read_json(DATA_PATH, lines=True)

print("Dataset loaded successfully")
print(df.head())

# Basic preprocessing
# Combine all text fields into one and handles missing values
df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

# Keyword frequency feature (algorithmic cues from text)
keywords = [
    "graph", "tree", "dp", "dynamic", "recursion",
    "greedy", "math", "string", "array", "sorting"
]

def keyword_frequency(text):
    text = text.lower()
    return [text.count(k) for k in keywords]

keyword_features = np.array(
    df["text"].apply(keyword_frequency).tolist()
)

# Create difficulty labels (classification)

def get_difficulty(score):
    if score < 4:
        return "Easy"
    elif score < 7:
        return "Medium"
    else:
        return "Hard"

df["difficulty"] = df["problem_score"].apply(get_difficulty)

# Feature extraction (TF-IDF)

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)
X_tfidf = tfidf.fit_transform(df["text"])
X = np.hstack([X_tfidf.toarray(), keyword_features])


# Targets

# Classification target
label_encoder = LabelEncoder()
y_class = label_encoder.fit_transform(df["difficulty"])

# Regression target
y_reg = df["problem_score"]

# Train-test split

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)


# 7. Train models (Random Forest)

# Classification model
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_class_train)

# Regression model
reg = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
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

cm = confusion_matrix(y_class_test, y_class_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Difficulty Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print(f"Regression MAE: {mae:.2f}")
print(f"Regression RMSE: {rmse:.2f}")


# Save models

os.makedirs("models", exist_ok=True)

joblib.dump(clf, "models/classifier.pkl")
joblib.dump(reg, "models/regressor.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(keywords, "models/keywords.pkl")


print("Models saved successfully in /models")
