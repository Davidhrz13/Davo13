#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:29:14 2024

@author: davidhernandez
"""


# KNN for Classification  - ABC Task

# Import Packages

import pandas as pd
import pickle 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import RFECV


# Import sample data

data_for_model = pd.read_pickle(open("Data/abc_classification_modelling.p", "rb"))

# Drop unecessary columns

data_for_model.drop("customer_id", axis = 1, inplace = True)


# Shuffle data

data_for_model = shuffle(data_for_model, random_state = 42)


# Identifying class balance for output variable

data_for_model["signup_flag"].value_counts(normalize = True) # normalize for frequencies 69% vs 31%


# Deal with Missing Values

data_for_model.isna().sum() # very few missing values, for distance_from_stor, gender and credit score
# Hence it makes sense to only drop these values

data_for_model.dropna(how = "any", inplace = True)

# Deal with Outliers

outlier_investigacion = data_for_model.describe()
# We can investigate distance_from_store, total_sales, and total_items

outlier_columns = ["distance_from_store", "total_sales", "total_items"]

for column in outlier_columns: 
    
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended 
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index # Return index values for outliers
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)
    

# Split input and output variables

X = data_for_model.drop(["signup_flag"], axis = 1)
y = data_for_model["signup_flag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y) #stratifue for having same proportion of 1 and 0 in training


# Deal with categorical variable - gender 

categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse_output = False, drop = "first") 

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])


encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1) #drop true is for aligning rows
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1) #drop true is for aligning rows
X_test.drop(categorical_vars, axis = 1, inplace = True)


# Feature Scaling - Very important for distance models - MinMax Scaler - Normalization

scale_norm = MinMaxScaler()
X_train = pd.DataFrame(scale_norm.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scale_norm.transform(X_test), columns = X_test.columns) # No fit in test data, just transform


# Feature Selection - Each dimension is one of the input variables 
# We will pass a RF instaed of Log Regression

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state = 42)
feature_selector = RFECV(clf)

fit = feature_selector.fit(X_train,y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]


plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()

#Num items and credit_score where removed 

# Model Training 

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)


# Model Assessment 

y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1] # [:,1] to only get values of probability of belonging to 1


# Confusion Matrix 

conf_matrix = confusion_matrix(y_test, y_pred_class)

#plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j,i, corr_value, ha = "center", va = "center", fontsize = 12)

plt.show()


# Accuracy (Number of correct classifications out of all attempted classifications)

accuracy_score(y_test,y_pred_class) # 93.6%

# Precision (Of all osbservations that were predicted as positive, how many were actually positive)

precision_score(y_test,y_pred_class) # 100%

# Recall (Of all positie observations, how many were predicted as positive)

recall_score(y_test,y_pred_class) # 76% 

# F1 Score (Harmonic mean of precision and recall)

f1_score(y_test,y_pred_class) # 86%


# Finding the optimal K value 

K_list = list(range(2,25))
accuracy_scores =[]

for k in K_list:
    
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_k_value = K_list[max_accuracy_idx]

# plot of k optimal values

plt.plot(K_list, accuracy_scores)
plt.scatter(optimal_k_value, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy (F1 Score) by k \n Optimal value for k: {optimal_k_value} (Accuracy): {round(max_accuracy,4)}")
plt.xlabel("K")
plt.ylabel("Accyracy (F1 Score)")
plt.tight_layout()
plt.show()

# default of 5 resulted in the optimal value for this type of model 













