#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, roc_curve
import scipy.stats as stats
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


# Load our data

# In[2]:


data = pd.read_csv("C:\\Users\\Faith\\Downloads\\Loan_Data.csv")
data.head(5)


# In[3]:


data.info()


# In[4]:


#Target variable is Loan_Status

# drop Index
data = data.drop(columns = ['Loan_ID'], inplace = False)


# In[5]:


data.nunique()


# In[6]:


data.plot(figsize=(18, 8))


# # Exploratory Data Analysis (EDA)

# In[7]:


data.describe().T


# Possibility of some outliers. However, most of the variables in real life will show some outliers so we should let the model learn to predict on outliers.

# Univariate Analysis

# In[8]:


sns.boxplot(x=data['ApplicantIncome'])


# In[9]:


import matplotlib.pyplot as plt
hist = plt.hist(x=data['ApplicantIncome'], density = True)
# Draw a vertical line in the histogram to visualize mean value of the numerical feature (NaNs will be ignored when calculating the mean)
plt.axvline(data['ApplicantIncome'].mean(), color = 'red', linestyle='--')
# Draw another vertical line in the histogram to visualize median value of the numerical feature (NaNs will be ignored when calculating the median)
plt.axvline(data['ApplicantIncome'].median(), color = 'black', linestyle='-')


# Observations:
# 
# 1. Positively skewed distribution
# 2. Outliers on right hand side, but expected with Income
# 3. Mean and Median are close

# In[10]:


sns.boxplot(x=data['CoapplicantIncome'])


# In[11]:


hist = plt.hist(x=data['CoapplicantIncome'], density = True)
# Draw a vertical line in the histogram to visualize mean value of the numerical feature (NaNs will be ignored when calculating the mean)
plt.axvline(data['CoapplicantIncome'].mean(), color = 'red', linestyle='--')
# Draw another vertical line in the histogram to visualize median value of the numerical feature (NaNs will be ignored when calculating the median)
plt.axvline(data['CoapplicantIncome'].median(), color = 'black', linestyle='-')


# In[12]:


sns.boxplot(x=data['LoanAmount'])


# In[13]:


hist = plt.hist(x=data['LoanAmount'], density = True)
# Draw a vertical line in the histogram to visualize mean value of the numerical feature (NaNs will be ignored when calculating the mean)
plt.axvline(data['LoanAmount'].mean(), color = 'red', linestyle='--')
# Draw another vertical line in the histogram to visualize median value of the numerical feature (NaNs will be ignored when calculating the median)
plt.axvline(data['LoanAmount'].median(), color = 'black', linestyle='-')


# Observations:
# 1. Slightly positively skewed.

# In[14]:


data['Dependents'].value_counts(normalize = True).plot.bar()
plt.xticks(rotation = 0)


# In[15]:


data['Gender'].value_counts(normalize = True).plot.bar()
plt.xticks(rotation = 0)


# In[16]:


data['Married'].value_counts(normalize = True).plot.bar()
plt.xticks(rotation = 0)


# In[17]:


data['Education'].value_counts(normalize = True).plot.bar()
plt.xticks(rotation = 0)


# In[18]:


data['Self_Employed'].value_counts(normalize = True).plot.bar()
plt.xticks(rotation = 0)


# In[19]:


data['Credit_History'].value_counts(normalize = True).plot.bar()
plt.xticks(rotation = 0)


# In[20]:


data['Property_Area'].value_counts(normalize = True).plot.bar()
plt.xticks(rotation = 0)


# Bivariate Analysis

# In[21]:


sns.pairplot(data, diag_kind= 'kde', hue = 'Loan_Status')


# In[22]:


# Identify Correlation
data.corr()


# In[23]:


data.corr(method = 'spearman')


# In[24]:


# Plot the Correlation matrix
plt.figure(figsize=(15,10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, vmin = -1, vmax = 1, fmt = '.2g')


# In[25]:


plt.figure(figsize=(15,10))
sns.heatmap(data.corr(method = 'spearman'), annot=True, linewidths=.5, vmin = -1, vmax = 1, fmt = '.2g')


# There is a very high correlation between 'ApplicantIncome' and 'LoanAmount'. Therefore, one of them needs to be dropped to prevent independent variable multicollinearity.

# In[26]:


data.isnull().sum()


# Data Processing

# In[27]:


# Imputing null for categorical variables with mode
data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Married'] = data['Married'].fillna(data['Married'].mode()[0])
data['Education'] = data['Education'].fillna(data['Education'].mode()[0])
data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])


# In[28]:


# Imputing null for numerical variables with median - mean was not chosen because of outliers in the data
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace = True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median(), inplace = True)


# In[29]:


data.isnull().sum()


# Balancing Dataset

# In[30]:


X = data.drop(columns = ['Loan_Status'])
y = data['Loan_Status']


# In[31]:


import imblearn
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_resample(X, y)
ax = sns.countplot(x=y)


# In[32]:


df1=pd.concat([X,y],axis=1)
df1.head()


# # Data preparation for modelling

# In[33]:


# OneHotEncoding of object dtype columns
cat_cols = ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']
dummies = pd.get_dummies(df1[cat_cols])
dummies


# In[34]:


#Concat dummmies into data frame
df1 = pd.concat([df1, dummies], axis = 1)
df1.shape


# In[35]:


# Drop original categorical columns from data frame
df1 = df1.drop(cat_cols, axis = 1)


# In[36]:


df1.shape


# In[37]:


from sklearn.preprocessing import LabelEncoder
df1['Loan_Status']
le = LabelEncoder()
# Convert the target variable into binary format
y = le.fit_transform(df1['Loan_Status'])


# In[38]:


# segregate the target variable
X = df1.drop(columns = ['Loan_Status'])

# splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[39]:


def get_metrics_score(clf, flag = True):
    # defining an empty list to store train and test results
    score_list=[] 
    # predict on both the training and test sets
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    # calculate Accuracy
    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    # calculate Recall
    train_recall = recall_score(y_train, pred_train)
    test_recall = recall_score(y_test, pred_test)
    # calculate Precision
    train_precision = precision_score(y_train, pred_train)
    test_precision = precision_score(y_test, pred_test)
    # calculate F1 score
    F1_Score = f1_score(y_test, pred_test)
    # calculate ROC_AUC_score
    Roc_Auc_score = roc_auc_score(y_test, pred_test)    
    # add these scores to score_list
    score_list.extend((train_acc, test_acc, train_recall, test_recall, train_precision, test_precision, f1_score, roc_auc_score))
        # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True: 
        print("Accuracy on training set : ", train_acc)
        print("Accuracy on test set : ", test_acc)
        print("Recall on training set : ", train_recall)
        print("Recall on test set : ", test_recall)
        print("Precision on training set : ", train_precision)
        print("Precision on test set : ", test_precision)
        print("F1_Score : ", F1_Score)
        print("Roc_Auc_score : ", Roc_Auc_score)
    
    return score_list # returning the list with train and test scores


# In[40]:


logreg = LogisticRegression(random_state = 42, class_weight = None)
logreg.fit(X_train, y_train)
predictions_logreg = logreg.predict(X_test)
predictions_logreg.shape


# In[41]:


print(classification_report(y_test, predictions_logreg))


# In[42]:


logistic_regression_scores = get_metrics_score(logreg)


# In[43]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10, algorithm = 'kd_tree')
knn.fit(X_train, y_train)
predictions_knn = knn.predict(X_test)
predictions_knn.shape


# In[44]:


print(classification_report(y_test, predictions_knn))


# In[45]:


knn_scores = get_metrics_score(knn)


# In[46]:


rf = RandomForestClassifier(random_state = 42, class_weight = 'balanced')
rf.fit(X_train, y_train)
predictions_rf = rf.predict(X_test)
predictions_rf.shape


# In[47]:


print(classification_report(y_test, predictions_rf))


# In[48]:


rf_scores = get_metrics_score(rf)


# Choosing the best model

# In[49]:


pd.DataFrame(data = {'Logistic Regression':logistic_regression_scores, 'KNN': knn_scores, 'Random Forest': rf_scores}, index = ['Accuracy - Train', 'Accuracy - Test', 'Recall - Train', 'Recall - Test', 'Precision - Train', 'Precision - Test', 'F1', 'ROC'])


# Based on the above chart, Random Forest by far outperform all other models, with the former edging slightly ahead on Accuracy and on Precision_Test. Also, with a close enough Accuracy on both train and test data, the model does not appear to be overfitting.

# Cross Validation

# In[50]:


from sklearn.model_selection import KFold, cross_val_score
clf = RandomForestClassifier(random_state=42)
k_folds = KFold(n_splits = 5)
scores = cross_val_score(clf, X, y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))


# ROC Curve plot

# In[51]:


def plot_roc_curve(y_test, y_pred):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# In[52]:


plot_roc_curve(y_test, predictions_rf)
print(f'model 1 AUC score: {roc_auc_score(y_test, predictions_rf)}')


# In[ ]:




