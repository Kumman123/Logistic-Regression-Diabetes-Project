# Logistic Regression on Diabetes Dataset - Full Project

## Overview

This project demonstrates logistic regression using two different approaches: a **manual implementation** of logistic regression using NumPy and a **standard implementation** using Scikit-learn. The dataset used for the classification task is the **Diabetes dataset** from Scikit-learn, with the goal of predicting whether a patient's diabetes metric is above or below the median value.

## Project Structure

- `question1.py`: A manual implementation of logistic regression using gradient descent.
- `question2.py`: A Scikit-learn-based implementation using `LogisticRegression` and `StandardScaler`.
- `README.md`: Provides detailed information about the project, including instructions for running the code.
- `LICENSE`: MIT license for open source distribution.

---

## Questions

### Question 1: Manual Logistic Regression Implementation

#### Objective:
Implement logistic regression **without using any libraries** like Scikit-learn. Use **gradient descent** to update the model weights and biases, and use the sigmoid function for prediction. Train the model and evaluate its performance.

#### Approach:
In `question1.py`, logistic regression is implemented using **NumPy**. The model performs gradient descent over multiple epochs to adjust the weights and bias. Predictions are made using the **sigmoid function**. The loss is calculated using the **logistic loss function** (binary cross-entropy).

- **Input Data**: Manually defined arrays for features and target values.
- **Training**: Using gradient descent, updating weights and biases at each iteration.
- **Evaluation**: Accuracy, Precision, Recall, F1 Score, and Confusion Matrix are computed.

### Question 2: Logistic Regression Using Scikit-learn

#### Objective:
Use Scikit-learn’s `LogisticRegression` to solve the binary classification problem. The dataset is preprocessed using **StandardScaler**, and performance metrics such as accuracy, precision, recall, F1 score, and the confusion matrix are calculated.

#### Approach:
In `question2.py`, we:
- Load the **Diabetes dataset** from Scikit-learn.
- Use `StandardScaler` to normalize the features.
- Train a **LogisticRegression** model on the training set and evaluate the model on the test set.
- Compute evaluation metrics like Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.

---

## Dataset

The **Diabetes dataset** is used, which contains 10 baseline variables (features) including age, sex, BMI, blood pressure, and others, gathered from patients with diabetes. The target variable is converted into a **binary classification** based on whether the diabetes measure is above or below the median value.

---

## Requirements

Ensure that you have Python and the following libraries installed:
- `numpy`
- `pandas`
- `scikit-learn`

You can install these libraries via pip:

```bash
pip install numpy pandas scikit-learn
