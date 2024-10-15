# Logistic Regression on Diabetes Dataset

This project demonstrates two implementations of logistic regression for predicting whether a patient's diabetes metric is above or below the median using the diabetes dataset from `sklearn`.

## Project Structure

The project consists of two Python scripts:

1. **question1.py**: 
   - A manual implementation of logistic regression using NumPy.
   - The model uses gradient descent to adjust weights and bias and classifies the data after several epochs.
   
2. **question2.py**:
   - A solution using Scikit-learn's `LogisticRegression` class.
   - The data is standardized using `StandardScaler`, and logistic regression is trained and evaluated with the help of built-in functions like `accuracy_score`, `confusion_matrix`, and `classification_report`.

## Dataset

The project uses the **Diabetes dataset** from the Scikit-learn library. The target is converted into a binary classification problem where:
- `1` denotes that the target value is above the median.
- `0` denotes that the target value is below the median.

## Requirements

Make sure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `scikit-learn`

You can install the required libraries using pip:
```bash
pip install numpy pandas scikit-learn
