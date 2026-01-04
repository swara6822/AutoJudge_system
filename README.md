# AutoJudge – Programming Problem Difficulty Prediction

##  Project Overview
AutoJudge is a machine learning–based system designed to predict the difficulty of programming problems.
The project performs two tasks:
- Classification of problems into Easy, Medium, or Hard
- Regression to estimate a numerical difficulty rating

Predictions are based on problem metadata such as problem tags and the number of users who solved the problem.

---

##  Dataset Used
- Source: Codeforces (publicly available programming problem dataset)
- Format: CSV file
- Features Used:
  - Problem tags
  - Number of users who solved the problem
- Target Variables:
  - Difficulty class (Easy / Medium / Hard)
  - Problem rating (numerical)

The dataset is included in the `data/` directory of this repository.

---

##  Approach and Models Used

### Data Preprocessing
- Removed rows with missing or invalid values
- Combined multiple problem tags into a single text feature
- Converted solve count into numerical format

### Feature Engineering
- Applied TF-IDF vectorization on problem tags
- Used solve count as an additional numerical feature

### Models Used
- Classification Model: Logistic Regression
- Regression Model: Random Forest Regressor

A hybrid approach was adopted after experimentation to achieve better overall performance.

---

##  Evaluation Metrics

### Classification
- Accuracy: 87.5%

### Regression
- Mean Absolute Error (MAE): 167.94
- Root Mean Squared Error (RMSE): 228.15

These metrics indicate reliable performance for both difficulty classification and rating prediction.

---

##  Web Interface
A web interface was built using Streamlit that allows users to:
- Enter problem tags
- Enter the number of users who solved the problem
- Instantly view the predicted difficulty level and difficulty rating

The application runs locally and uses the trained models for inference.

---

##  Steps to Run the Project Locally

```bash
# 1. Clone the repository
git clone https://github.com/swara6822/AutoJudge_system.git
cd AutoJudge_system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the models
python3 train.py

# 4. Run the web application
streamlit run app.py

# Demo Video: https://drive.google.com/file/d/1-COia1tOTDnNwZ0drxwDWRCInpChGeMd/view?usp=sharing

# Report: The detailed project report is available in the /report folder.

# Name: Swara Vikram Chalikwar
# Year:2nd
# Enrollment No: 24115153 