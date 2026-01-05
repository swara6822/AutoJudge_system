# AutoJudge – Programming Problem Difficulty Prediction

## Project Overview
AutoJudge is a machine learning–based system designed to predict the difficulty of programming problems using only textual information. The system performs two tasks:
- Classification of problems into Easy, Medium, or Hard
- Regression to estimate a numerical difficulty score on a 1–10 scale

Predictions are based solely on the textual description of programming problems.

---

## Dataset Used
The dataset used in this project was provided as part of the project guidelines. It consists of programming problems with the following fields:
- Problem title
- Problem description
- Additional textual fields (input/output descriptions) 
- sample I/O
- Difficulty scores (`problem_score`, `problem_class`)
- Problem URL

In this implementation, only the problem title and main description were used as input features. Other non-textual metadata were excluded.

The dataset is stored in JSON Lines (`.jsonl`) format and is available in the `data/` directory.

---

## Approach and Models Used

### Data Preprocessing
- Handled missing textual values by replacing them with empty strings
- Combined problem title and description into a single unified text field

### Feature Engineering
- Applied TF-IDF vectorization to the combined text
- Engineered keyword frequency features using algorithm-related terms such as *graph*, *dp*, and *recursion*

### Models Used
Multiple models were evaluated during experimentation, including Logistic Regression, Linear Regression, Support Vector Machines, and Gradient Boosting.  
Based on empirical results, the following models were selected as final:
- **Classification:** Random Forest Classifier
- **Regression:** Random Forest Regressor

---

## Evaluation Metrics

### Classification
- Accuracy: 0.5115

### Regression
- Mean Absolute Error (MAE): 1.73
- Root Mean Squared Error (RMSE): 2.07

These metrics reflect realistic performance for text-based difficulty prediction.

---

## Web Interface
A web interface was built using Streamlit that allows users to:
- Enter the full textual description of a programming problem
- View the predicted difficulty class and difficulty score

The application performs inference using the trained machine learning models.

---

## Steps to Run the Project Locally

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

# Demo Video: https://drive.google.com/file/d/1L_5fhGDA9U3fOSu4E8xOIigImyyXQQbp/view?usp=sharing

# Report: The detailed project report is available in the /report folder.

# Name: Swara Vikram Chalikwar
# Year:2nd
# Enrollment No: 24115153 