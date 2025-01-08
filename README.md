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

**Features:**
The dataset includes a mix of numerical and categorical features representing the magical and physical traits of scholars. 

**Numerical Features:**

- **Fae_Dust_Reserve**: Represents the subject’s reserve of mystical dust indicating magical potential.

- **Physical_Stamina**: Indicates the subject's overall physical endurance and health.

- **Mystical_Index**: Numeric representation of the subject's mystical power and well-being.

- **Mystic_Energy_Level**: The level of mystical energy possessed by the subject.
  
- **Age_of_Wisdom**: The subject's age, indicating life experience.

- **Mental_Wizardry**: Represents the subject's mental health and wizardry capacity.

- **Potion_Power_Level**: Represents the power or effectiveness of the potions used by the subject.

- **Gold_Pouches_Per_Year**: The subject’s annual income represented as gold pouches.

- **Wizardry_Skill**: The subject’s proficiency in magical skills.

- **Spell_Mastering_Days**: Number of days the subject has dedicated to mastering spells.

- **Level_of_Academic_Wisdom**: The highest level of knowledge or wisdom achieved by the subject.

- **General_Health_Condition**: An overall assessment of the subject's health status.

- **Dragon_Sight_Sharpness**: Measures the subject’s visual acuity or ability to see mystical beings like dragons.
    
- **Enchanted_Coin_Count**: The number of enchanted coins the subject possesses.

- **Celestial_Alignment**: Represents the alignment of the subject with celestial forces.

- **Knightly_Valor**: The bravery and valor displayed by the subject.

- **Rune_Power**: Represents the power derived from magical runes by the subject.

 
**Categorical Features:**

- **Healer_consultation_Presence**: 
  - **Explanation**: Indicates whether the subject has consulted a healer recently.
  - **Values**: Categorical, "Present" or "Absent".

- **Elixir_veggies_consumption_Presence**: 
  - **Explanation**: Indicates whether the subject has consumed enchanted vegetables.
  - **Values**: Categorical, "Present" or "Absent".

- **Bolt_of_doom_Presence**: 
  - **Explanation**: Indicates whether the subject experienced a thunderstrike.
  - **Values**: Categorical, "Present" or "Absent".

- **High_willingness_Presence**: 
  - **Explanation**: Indicates whether the subject has a high willingness to engage in advanced magical practices.
  - **Values**: Categorical, "Present" or "Absent".

- **Defense_spell_difficulty_Presence**: 
  - **Explanation**: Indicates if the subject has difficulty casting defense spells.
  - **Values**: Categorical, "Present" or "Absent".

- **Doc_availability_challenge_Presence**: 
  - **Explanation**: Shows if there were any barriers preventing access to healers.
  - **Values**: Categorical, "Present" or "Absent".

- **Dexterity_check_Presence**: 
  - **Explanation**: Indicates the subject's high dexterity.
  - **Values**: Categorical, "Present" or "Absent".

- **Fruits_of_eden_consumption_Presence**: 
  - **Explanation**: Indicates whether the subject consumes fruits from Eden.
  - **Values**: Categorical, "Present" or "Absent".

- **Knight_physical_training_Presence**: 
  - **Explanation**: Indicates if the subject underwent knight-like physical training.
  - **Values**: Categorical, "Present" or "Absent".

- **Royal_family_pressure_Presence**: 
  - **Explanation**: Indicates if the subject faces pressure from the royal family.
  - **Values**: Categorical, "Present" or "Absent".

- **Guild_Membership**: 
  - **Explanation**: The guild or magical faction the subject will belong to.
  - **Values**: Categorical, "Master_Guild", "No_Guild", "Apprentice_Guild" -> **target variable**

- **Heavy_elixir_consumption_Presence**: 
  - **Explanation**: Indicates if the subject consumes heavy magical elixirs.
  - **Values**: Categorical, "Present" or "Absent".

- **Stigmata_of_the_cursed_Presence**: 
  - **Explanation**: Indicates if the subject experienced a crisis or damage from magical or dark powers.
  - **Values**: Categorical, "Present" or "Absent".

- **Dragon_status_Presence**: 
  - **Explanation**: Denotes whether the subject has the sign of the dragon.
  - **Values**: Categorical, "Present" or "Absent".







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
