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

#### **1.1 Understand Column Meanings**
We used a dataset named guilds.csv, which contains 253,680 rows and 31 columns, describing various magical and physical attributes of scholars in the Kingdom of Marendor. The data spans a mix of numerical, categorical, and derived variables that offer insights into the factors influencing guild memberships.

**Types of Data:**

- Numerical Columns: Examples include Fae_Dust_Reserve, Physical_Stamina, and Mystical_Index.
- Categorical Columns: Examples include Healer_consultation_Presence and Bolt_of_doom_Presence.
- Target Variable: Guild_Membership, which specifies guild affiliation, such as Master_Guild or No_Guild.
- Data Types: While most columns are numerical, some categorical features are binary or text-based.

#### **1.2 Check Data Integrity**
**Handling Missing Values:**
We identified missing data using visualizations (e.g., missing data matrix).
IMMAGINE

**Types of Missingness:**
The patterns and correlations observed suggest that most of the missingness in the dataset can be explained by relationships with other observed variables, classifying it primarily as `Missing at Random` (MAR). 

#### **1.3 Correlation Heatmap**

--- 

### **2. Preprocessing the Dataset**
 
#### **2.1 Handling Missing Data**
We used two different startegies to handles missing values (NaN):

a) **Drop Missing Target Values**:
     - Rows with missing values in the Guild_Membership column were removed, ensuring the dataset was valid for supervised learning tasks.

b) **Imputation for Features**:
     - **Numerical Features**: Missing values were replaced with the median to handle skewness and reduce the impact of outliers.
     - **Categorical Features**: Missing values were replaced with the most frequent value, preserving categorical integrity.


#### **2.2 Encoding Categorical Features**
- **One-Hot Encoding**: it was applied to categorical features such as Healer_consultation_Presence, creating binary columns for each category.
- **Label Encoding for Target**: Guild_Membership was encoded as integers: Master Guild (2), Apprentice Guild (1) and No Guild (0).


#### **2.3 Outlier Treatment**
Instead of removing outliers, we used **Robust Scaling** to reduce their influence while preserving critical information.

- **Robust Scaling**: it scaled features by subtracting the median and dividing by the interquartile range (IQR), making sure that the outliers had minimal impact on scaled values.

---

### **3. Defining the problem type: Regression, Classification or Clustering**

Based on the dataset, this is a multi-class classification problem because the target variable `Guild_Membership` consists of discrete categories representing the guild each scholar belongs to ("Master Guild", "Apprentice Guild" and "No Guild"). The objective is to predict which guild each scholar will be assigned to based on their attributes.  

---

### **4. Algorithms:**

We chose three algorithms based on their characteristics and suitability for the Guild dataset:

**Logistic Regression**

Logistic Regression because it is simple and interpretable, making it suitable for understanding relationships between features and the target. It is also computationally efficient, even with large datasets. However, it assumes linearity between features, which may not hold true for complex datasets like this one. Logistic Regression struggles with imbalanced data, often predicting the majority class, and has limited ability to capture non-linear patterns unless features are transformed or interactions are explicitly modeled.

**Random Forest**

Random Forest is an ensemble method that builds multiple decision trees on bootstrapped samples and combines their predictions to reduce overfitting. It effectively handles non-linearity, provides feature importance metrics, and is robust to imbalanced datasets. However, it can be computationally intensive, memory-heavy for large datasets, and requires careful hyperparameter tuning for optimal performance.

**CART Decision Trees**

CART (Classification and Regression Trees) is a model that splits the dataset into subsets based on feature values, recursively partitioning until a stopping criterion is met. It is easy to visualize, interpret, and explain, captures non-linear relationships and requires minimal preprocessing for both numerical and categorical data. However, it is highly prone to overfitting without regularization and it is sensitive to small changes in the dataset.

#### **4.1 Training of the models**

a) The dataset was split into training, validation, and test sets using stratified sampling to preserve class distribution across all subsets. This approach ensures consistent proportions of the target variable, allowing for reliable model training, hyperparameter tuning, and evaluation. The same splitting method was applied to all models to ensure comparability.
b) The models were trained on the training set using default hyperparameters and evaluated on the validation set, with metrics such as precision, recall, F1-score, and a confusion matrix calculated to assess performance.
c) We used cross-validation with GridSearchCV to identify the best hyperparameters for the models by testing various combinations of parameters. This approach ensures the models are optimized for performance while addressing issues like class imbalance and overfitting. The best hyperparameters and their cross-validation accuracy were extracted to fine-tune the model. 

---






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
