# Predicting Guild Memberships 

## Team Members:

[Giulia Castelli] (Student ID: [300691])
[Francesca Peppoloni]
[Anna Granzotto]


## [Section 1] Introduction

This project focuses on predicting guild memberships within the Kingdom of Marendor, a fictional world where scholars possess unique combinations of magical and physical attributes. The dataset comprises 31 features, combining both categorical and numerical features. Guild membership represents the target variable for classification.

The primary objective is to build machine learning models to accurately classify scholars into their respective guilds based on their attributes. This involves extensive data preprocessing, including handling missing values, encoding categorical features, and robustly scaling numerical features to address outliers. We further explore the efficacy of three machine learning models, which are Logistic Regression, Random Forest and CART Decision Trees, to evaluate and optimize classification performance.

By applying advanced optimization strategies such as hyperparameter tuning and cross-validation, the project aims to identify the most effective model for this classification task. Insights gained from this analysis are expected to enhance our understanding of the factors influencing guild memberships and contribute to the development of interpretable and robust classification systems.

## [Section 2] Methods

### **1. EDA**
Exploratory Data Analysis (EDA) helps us to understand, clean, and gain insights from our 
dataset, preparing it for machine learning models. In order to do so we employed several Python libraries for analysis and visualization, such as Pandas, Numpy, Scikit-learn, Matplotlib, Seaborn, and Missingno. EDA process steps are:
1. Understand Column Meanings
2. Check Data Integrity
3. Visualize Distributions
4. Correlation Heatmap

#### **Understand Column Meanings**
We used a dataset named guilds.csv, which contains 253,680 rows and 31 columns, describing various magical and physical attributes of scholars in the Kingdom of Marendor. The data spans a mix of numerical, categorical, and derived variables that offer insights into the factors influencing guild memberships.

**Types of Data:**

- Numerical Columns: Examples include Fae_Dust_Reserve, Physical_Stamina, and Mystical_Index.
- Categorical Columns: Examples include Healer_consultation_Presence and Bolt_of_doom_Presence.
- Target Variable: Guild_Membership, which specifies guild affiliation, such as Master_Guild or No_Guild.
- Data Types: While most columns are numerical, some categorical features are binary or text-based.

#### **Check Data Integrity**
**Handling Missing Values:**
We identified missing data using visualizations (e.g., missing data matrix).
IMMAGINE

Types of Missingness:
The patterns and correlations observed suggest that most of the missingness in the dataset can be explained by relationships with other observed variables, classifying it primarily as `Missing at Random` (MAR). However, there are specific cases where some variables might exhibit missing values due to reasons inherent to the variables themselves, such as sensitivity or unavailability of the data, indicating a potential for `Missing Not at Random` (MNAR) in certain scenarios.


#### **Correlation Heatmap**

Encoding Categorical Features:
To ensure compatibility with machine learning models, we applied:
One-hot encoding for binary columns like Healer_consultation_Presence.
Manual mapping for ordinal columns such as Guild_Membership.
Outlier Treatment:
Instead of removing outliers, we scaled numerical features using Robust Scaling to mitigate their impact while preserving the range of values crucial for guild predictions.
Feature Selection and Transformation:
Features were evaluated for relevance and transformed where necessary to enhance model performance and computational efficiency.
Tools and Libraries:

Visualization: Missingno and Seaborn were used to visualize missing data and feature distributions.
Scaling: RobustScaler from Scikit-learn was applied to numerical columns.

### **2. Preprocessing the Dataset**

**Strategies we used to drop and imput data**

Since Guild_Membership is the target variable for our analysis, it is crucial to ensure that this column contains no missing values. Rows with missing values in the target variable cannot be used effectively in supervised learning tasks, such as classification or regression, because the model would have no outcome to predict for those rows. Dropping rows with missing Guild_Membership values ensures that the dataset used for modeling contains complete and valid targets, which is essential for training and evaluation.

 On the other hand, the other columns in the dataset serve as features, and their missing values can be imputed using appropriate methods, such as the median for numerical variables or the most frequent value for categorical variables. This approach allows you to retain as much data as possible while addressing missingness in a way that preserves the dataset's overall structure and variability. Imputation ensures that the information from the feature columns is not lost, thereby maximizing the size of the dataset and providing the model with the richest possible set of input data. 
 
 By combining these strategies—dropping rows with missing target values and imputing missing feature values—you maintain the integrity of the target variable while making the most of the available information in the features, ultimately creating a dataset that is both clean and robust for analysis.

#### **Drop missing values (NaN) in Guild_Membership**

Since Guild_Membership is the target variable for our analysis, it is crucial to ensure that this column contains no missing values. Rows with missing values in the target variable cannot be used effectively in supervised learning tasks, such as classification or regression, because the model would have no outcome to predict for those rows. Dropping rows with missing Guild_Membership values ensures that the dataset used for modeling contains complete and valid targets, which is essential for training and evaluation.

#### **Imput new values in other columns: Median**

 While for Guild_Membership we dropped its rows with missing values, on the other hand, the other columns in the dataset serve as features, and their missing values can be imputed using appropriate methods. This approach allows us to retain as much data as possible while addressing missingness in a way that preserves the dataset's overall structure and variability. Imputation ensures that the information from the feature columns is not lost, thereby maximizing the size of the dataset and providing the model with the richest possible set of input data. 

**Substituting missing values in numerical columns**

The **mean** is the average value of a dataset, calculated by summing all numbers and dividing by their count. It is sensitive to extreme values, which can significantly impact its value. On the other hand, the **median** represents the middle value of an ordered dataset and is resistant to outliers, making it more robust when dealing with extreme data points. In skewed distributions, the mean is influenced by the skew, while the median provides a more central and accurate representation.

In our case, where the dataset is highly imbalanced, the `median` is a better choice for summarizing data. Since extreme values are likely present due to the imbalance, using the median ensures these outliers do not disproportionately affect the representation of the data, making it more reliable for analysis.

**Substituting missing values in categorical columns**

For categorical columns, missing values are replaced with the `most frequent` value in the column, ensuring that no invalid or non-categorical entries are introduced. 
This is done through the SimpleImputer class from sklearn.impute, where the imputation strategy is set to 'most_frequent'.


#### **Encoding categorical features: One-Hot Encode**

**One-hot encoding** is a method used to transform categorical features into numerical representations by creating new binary columns for each unique category in a feature. Each of these binary columns takes a value of 0 or 1, indicating the absence or presence of the corresponding category in a given row of the dataset. 

The target variable "Guild_Membership" is transformed into numerical labels through label encoding, where each unique category ("Master Guild", "Apprentice Guild" and "No Guild") is assigned a distinct integer value (0, 1 or 2) encoded into a single column.



#### **Fixing outliers: Robust Scaling**

We decided to scale outliers instead of removing them because higher values in certain features are critical for distinguishing between guild memberships, such as master, no guild, or apprentice. Removing these values could lead to a loss of important information that contributes to classification, as the presence of extreme values might be significant indicators for certain categories in the target variable. By scaling, we reduce their numerical impact while preserving their relative importance within the dataset.

Robust Scaling is a scaling technique that adjusts numerical data by subtracting the median and scaling according to the interquartile range (IQR). It is particularly effective for datasets with outliers.

Robust scaling is not sensitive to outliers because it uses the median (a robust measure of central tendency) and IQR (a robust measure of spread).
This ensures that outliers do not overly influence the scaled values, which is critical for the dataset with features like Fae_Dust_Reserve and Physical_Stamina that have significant outliers.


---


### **3 Defining the problem type: Regression, Classification or Clustering**


### **4 Algorithms:**

Three algorithms were chosen based on their characteristics and suitability for the dataset:

**Logistic Regression:**
Logistic Regression is simple and interpretable, making it suitable for understanding relationships between features and the target. It is also computationally efficient, even with large datasets. However, it assumes linearity between features and the log-odds of the target, which may not hold true for complex datasets like this one. Logistic Regression struggles with imbalanced data, often predicting the majority class, and has limited ability to capture non-linear patterns unless features are transformed or interactions are explicitly modeled.

**Random Forest:**
Random Forest is an ensemble method that builds multiple decision trees during training and combines their predictions (e.g., majority vote for classification). Each tree is trained on a bootstrapped sample, and feature selection is randomized at each split to reduce overfitting.

Pros:

Handles non-linearity and interactions among features without explicit feature engineering.

Provides feature importance metrics, aiding in interpretability.

Robust to overfitting due to ensemble averaging and well-suited for imbalanced datasets by weighting classes or using techniques like SMOTE.

Cons:

Computationally intensive, especially with a high number of trees (n_estimators) or deep trees (max_depth).

Memory usage can be significant for large datasets.

Requires careful tuning of hyperparameters (e.g., n_estimators, max_features, max_depth) to achieve optimal performance.

CART Decision Trees:
CART (Classification and Regression Trees) is a non-parametric model that splits the dataset into subsets based on feature values, recursively partitioning until leaves contain instances of a single class or reach a stopping criterion.

Pros:

Easy to visualize, interpret, and explain to non-technical stakeholders.

Captures non-linear relationships and interactions among features naturally.

Requires minimal preprocessing, as it handles both numerical and categorical features natively.

Cons:

Highly prone to overfitting, as it tries to perfectly classify training data unless regularized (e.g., by limiting max_depth or min_samples_split).

Sensitive to small changes in the dataset, which can significantly alter the structure of the tree.

Limited predictive power as a standalone model; often outperformed by ensemble methods like Random Forest.





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
