#!/usr/bin/env python
# coding: utf-8

# In[51]:


# OPAN-6607-200 Final Project (Fa2024)
## Ashley Daniels
## 12/12/2024


# ---

# In[53]:


#### Q1: Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

import pandas as pd 
import numpy as np
import altair as alt
import sklearn as sk

df = pd.read_csv("social_media_usage.csv")
s = pd.DataFrame(df)
print(s)


# ---

# In[55]:


#### Q2:Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

def clean_sm(x):
    return np.where(x == 1, 1, 0)


toy_df = pd.DataFrame({'one': [1, 0, 1, 5], 'two': [1, 4,3, 0]})
clean_df = toy_df.map(clean_sm)
print("Cleaned toy dataframe:\n", clean_df)


# ---

# In[57]:


#### Q3: Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

ss = pd.DataFrame(s[['income', 'educ2', 'par', 'marital', 'gender','age']])

print(ss)

ss['sm_li'] = clean_sm(s['web1a'])

print("'ss'DataFrame :\n", ss)

ss['income'] = np.where(ss['income'] > 9, np.nan, ss['income'])
ss['educ2'] = np.where(ss['educ2'] > 8, np.nan, ss['educ2'])
ss['age'] = np.where(ss['age'] > 98, np.nan, ss['age'])
ss['par'] = clean_sm(ss['par'])
ss['marital'] = clean_sm(ss['marital'])
ss['gender'] = np.where(ss['gender'] > 2, np.nan, ss['gender'])

ss = ss.dropna()

print("Correlation matrix:\n", ss.corr())
print("Value counts for sm_li:\n", ss['sm_li'].value_counts())


# ---

# In[59]:


#### Q4: Create a target vector (y) and feature set (X)

x = ss.drop(columns='sm_li')
y = ss['sm_li']


# ---

# In[61]:


#### Q5: Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)

print(f"Training Set: {len(x_train)} samples")
print(f"Testing Set: {len(x_test)} samples")


# Splitting the data into test and training sets allows for us to use the training samples on our models. We can then keep an undisturbed postion fo the sample data to used for validation once we find the best fit model. 

# ---

# In[64]:


#### Q6: Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data

from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(class_weight = 'balanced', random_state = 100)

model1.fit(x_train, y_train)


# ---

# In[66]:


#### Q7: Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

y_predictions = model1.predict(x_test)
model_accuracy = model1.score(x_test, y_test)
conf_matrix = sk.metrics.confusion_matrix(y_test, y_predictions)

print("Model Accuracy:", model_accuracy)
print("Model Confusion Matrix:\n", conf_matrix)


# The accuracy of this model is around 70.9%. In the confusion matrix, the number 138 indicates that the model correctly predicted 138 instances where people did not use LinkedIn. The 59 represents false positives where the model incorrectly predicted non-users. 14 refers to the number of instances where the model detected a false negative. Lastly, 40 represents the instances where the model correctly predicted users. 

# ---

# In[69]:


#### Q8: Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=["Actual Non-User", "Actual User"],
    columns=["Predicted Non-User", "Predicted User"]
)
print(conf_matrix_df)


# ---

# In[71]:


#### Q9: Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

true_pos = conf_matrix[1, 1]
true_neg = conf_matrix[0, 0]
false_pos = conf_matrix[0, 1]
false_neg = conf_matrix[1, 0]

precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

from sklearn.metrics import classification_report
report = classification_report(y_test, y_predictions)
print("Class Report:\n", report)


# The precision report for non-users is 0.91 compared to the 0.40 user prediction precision. So, it would be advantageous to use precision as an evaluation metric for identifying non-users. Recall would be the metric to identify real users, with a success rate of 0.74. The F1 Score is most advantageous when the dataset is balanced to identify real users.  

# ---

# In[74]:


#### Q10: Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

ss_model = pd.DataFrame({
    'income': [8, 8],
    'educ2': [7, 7],
    'par': [0, 0],
    'marital': [1, 1],
    'gender': [0, 0],
    'age': [42, 82]
})

predict_prob = model1.predict_proba(ss_model)[:, 1]
print(f"Probability of LinkedIn use (high earning and well-educated 42 year old married woman without children): {predict_prob[0]:.2f}")
print(f"Probability of LinkedIn use (high earning and well-educated 42 year old married woman without children):): {predict_prob[1]:.2f}")


# In[2]:


import jupytext as c
c.NotebookApp.contents_manager_class="jupytext.TextFileContentsManager"
c.ContentsManager.formats = "ipynb,py"

