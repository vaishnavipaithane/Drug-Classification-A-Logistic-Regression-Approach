# Drug Classification: A Multinominal Logistic Repression Approach

In this analysis, we aimed to classify five different types of drugs (Drug A, Drug B, Drug C, Drug Y, and Drug X) using a multinomial logistic regression model. Approach involved a comprehensive set of steps to preprocess the data, explore its characteristics, and train a robust classification model. Below is a summary of our process and findings.

## Summary of the analysis process

**1. Loading Libraries and Data:**

- Imported the necessary libraries (e.g., Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn).

- Dataset (drug_sample.csv) was provided by NyBerman Bioinformatics and it was loaded to jupyter notebook. The dataset consisted of 200 records with features such as Age, Sex, BP, Cholesterol, and Na_to_K, with the target variable being the drug type.

**2. Data Preprocessing:**
   
- Separated the features and target variable from the dataset.

- Categorical variables (Sex, BP, Cholesterol, Drug) were identified and encoded using OneHotEncoder. Target variable was label-encoded using LabelEncoder.

**3. Exploratory Data Analysis (EDA):**

A bar plot was used to analyze the distribution of drug types, providing insights into the prevalence of each drug type in the dataset.

**4. Model Training and Evaluation**

- The dataset was split into training (70%) and testing (30%) sets. A stratified split was performed to ensure the class distribution is maintained in both sets.

- A logistic regression model was defined and tuned using GridSearchCV for hyperparameter tuning to identify the best regularization parameter (C) with L2 regularization. The model was trained on the training set, and the best parameters identified were {'C': 1.0, 'penalty': 'l2'}.

- Evaluated the model’s accuracy on both training and test datasets. Generated and analyzed confusion matrices for both training and test sets to assess the model's classification performance.

- Accuracy: The model achieved an accuracy of 100% on both the training and testing datasets, indicating a perfect fit.

- Confusion Matrix: The confusion matrices for both the training and testing sets showed that the model correctly classified all instances without any misclassifications.

## Insights and Interpretation

The evaluation metrics, including accuracy and confusion matrix, showed that the model performed well. It achieved 100% accuracy on both the training and test data for all drug types, with no misclassifications. This suggests that the logistic regression model effectively distinguishes between different drug types.

## Conclusions

- The multinomial logistic regression model has proven to be an effective tool for classifying different types of drugs based on the given features. The model's performance metrics show its potential for practical applications, such as assisting healthcare professionals in drug classification and prescription systems.

- However, achieving perfect accuracy in both training and testing sets often requires a note of caution, as it may indicate overfitting, particularly if the dataset is relatively small. 

## Future Work 

Suggest possible improvements or future work, such as exploring other models, feature engineering, croos-validation or collecting more data to enhance model performance.

## Thank you

