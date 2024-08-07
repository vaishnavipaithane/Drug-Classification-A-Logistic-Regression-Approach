# Drug Classification: A Multinominal Logistic Repression Approach

In this analysis, we aimed to classify five different types of drugs (Drug Y, Drug C, Drug X, Drug A, and Drug B) using a multinomial logistic regression model. The approach involved a comprehensive set of steps to preprocess the data, explore its characteristics, and train a robust classification model. Below is a summary of our process and findings.

## Summary of the analysis process

**1. Loading Libraries and Data:**

- Imported the necessary libraries (e.g., Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn).

- Dataset ([drug_sample.csv](https://github.com/vaishnavipaithane/Drug-Classification-A-Multinomial-Logistic-Regression-Approach/blob/main/drug_sample.csv)) was provided by NyBerman Bioinformatics and it was loaded to jupyter notebook. The dataset consisted of 200 records with features and the target variable.

**2. Data Preprocessing:**
   
- Separated the features (Age, Sex, BP, Cholesterol, Na_to_K) and target variable (drug) from the dataset.

- **Label Encoding for Target:** The target variable (Drug) was label-encoded into numerical values.

- **One-Hot Encoding for Categorical Features:** Categorical features (Sex, BP, Cholesterol) were identified and transformed into a one-hot encoded matrix.

- **Combining Features:** Non-categorical features (Age, Na_to_K) were combined with the one-hot encoded categorical features into a final feature matrix.

**3. Exploratory Data Analysis (EDA):**

A [bar plot](https://github.com/vaishnavipaithane/Drug-Classification-A-Multinomial-Logistic-Regression-Approach/blob/main/Drug_Type_Distribution.pdf) was used to analyze the distribution of drug types, providing insights into the prevalence of each drug type in the dataset.

**4. Model Training and Evaluation**

- **Train-Test Split:** The dataset was split into training (70%) and testing (30%) sets. A stratified split was performed to ensure the class distribution is maintained in both sets.

- **Logistic Regression Model:**

**Standardization:** Features were standardized using StandardScaler.

**Pipeline and Grid Search:** A pipeline was created for scaling and model training, and GridSearchCV was used for hyperparameter tuning to identify the best regularization parameter (C) with L2 regularization. The model was trained on the training set, and the best parameters identified were  {'logisticregression__C': 100.0, 'logisticregression__penalty': 'l2'}.

Evaluated the model’s accuracy on both training and test datasets. Generated and analyzed confusion matrices for both training and test sets to assess the model's classification performance.

**Accuracy:** 

Test Accuracy: 95% 

Train Accuracy: 100%

**Confusion Matrix:**

Test Set: Shows good performance with a few misclassifications, particularly for classes drugC and drugX.

Train Set: The model showed perfect classification, which is expected given the 100% training accuracy.

## Conclusions

- The multinomial logistic regression model with optimized hyperparameters achieved high accuracy on both the test and training datasets, indicating the model's effective learning from the data.
- The confusion matrix for the test set shows that the model performs well with minimal misclassifications, though there are specific areas (e.g., distinguishing drug types C and X) where improvements could be made.
- This model has proven to be an efficient tool for classifying different types of drugs based on the given features.
- The model's performance metrics show its potential for practical applications, such as assisting healthcare professionals in drug classification and prescription systems. 

## Future Work 

Suggest possible improvements or future work, such as exploring additional model evaluation metrics, experimenting with other models, feature engineering, cross-validation or collecting more data to enhance model performance.

This concludes the analysis and classification of the drug dataset using multinomial logistic regression. The results indicate a highly accurate model, though further validation is suggested to ensure robustness.

Click [here](https://github.com/vaishnavipaithane/Drug-Classification-A-Multinomial-Logistic-Regression-Approach/blob/main/Project.ipynb) to view the script.

## Thank you

