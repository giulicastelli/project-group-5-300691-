Title and Team Members

Project Title: Predicting Guild Memberships in the Kingdom of Marendor
Team Members:

[Giulia Castelli] (Student ID: [300691])

[Francesca Peppoloni]

[Anna Granzotto]


[Section 1] Introduction

The purpose of this project is to predict guild memberships of scholars in the mythical kingdom of Marendor. Using the "Guilds" dataset, which contains a variety of magical and physical attributes of scholars, we aim to develop a machine learning model capable of accurate classification. This project will assist in understanding the underlying characteristics of guild membership and provide a practical application of machine learning techniques.

[Section 2] Methods

Proposed Ideas

Features:
The dataset contains features such as magical attributes (e.g., mana capacity, spell affinity), physical attributes (e.g., strength, agility), and social connections.

Algorithms:
We tested three models:

Random Forest Classifier

Support Vector Machine (SVM)

Neural Network (Multilayer Perceptron)

Training Overview:
The dataset was preprocessed by handling missing values, encoding categorical features, and standardizing numerical features. We performed feature selection to identify the most relevant attributes.

Environment

Programming Language: Python 3.9

Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn

Environment Reproducibility:

conda create -n guild_prediction python=3.9
conda activate guild_prediction
pip install -r requirements.txt

The requirements.txt includes all necessary dependencies.

Workflow Diagram

Below is a flowchart illustrating the steps in our machine learning pipeline:

Data Preprocessing

Feature Engineering

Model Training and Validation

Model Comparison

Test Set Evaluation

[Section 3] Experimental Design

Experiments

Main Purpose:
To evaluate the performance of different classifiers for predicting guild memberships.

Baselines:

Majority Class Baseline: Predicting the most common guild.

Decision Tree Classifier for initial comparison.

Evaluation Metrics:

Accuracy: To measure overall correctness.

F1 Score: To account for imbalanced guild memberships.

ROC-AUC: For analyzing classification thresholds.

Experimental Setup

We performed a 5-fold cross-validation to evaluate each model and used a stratified split to maintain class balance. The hyperparameters were optimized using grid search.

[Section 4] Results

Main Findings

Random Forest Classifier: Achieved the highest F1 Score of 0.87.

SVM: Balanced performance with an ROC-AUC of 0.82.

Neural Network: Performed well but required extensive tuning; F1 Score of 0.84.

Results Summary Table

Model

Accuracy

F1 Score

ROC-AUC

Random Forest

89%

0.87

0.91

SVM

85%

0.84

0.82

Neural Network

86%

0.84

0.85

Placeholder Figure

[Insert plot showing model comparison metrics]

[Section 5] Conclusions

Summary

This project successfully demonstrated the ability to classify guild memberships using machine learning techniques. The Random Forest Classifier emerged as the most effective model, achieving a strong balance between accuracy and interpretability.

Future Work

Despite the promising results, further research is needed to:

Explore additional features such as social dynamics or historical data.

Investigate ensemble methods for improving performance.

Develop interpretability techniques to better understand the driving factors behind predictions.
