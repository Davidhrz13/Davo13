#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:25:47 2024

@author: davidhernandez
"""

###########################################
# PCA Code Template
###########################################

# Random Forest for Classification with PCA

# Import Packages

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Import sample data

data_for_model = pd.read_csv("data/sample_data_pca.csv")

# Drop unecessary columns

data_for_model.drop("user_id", axis = 1, inplace = True)


# Shuffle data

data_for_model = shuffle(data_for_model, random_state = 42)


# Identifying class balance for output variable

data_for_model["purchased_album"].value_counts(normalize = True) 


# Deal with Missing Values

data_for_model.isna().sum().sum()
data_for_model.dropna(how = "any", inplace = True)


# Split input and output variables

X = data_for_model.drop(["purchased_album"], axis = 1)
y = data_for_model["purchased_album"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y) 


###########################################
# Feature Scaling
###########################################

scale_standard = StandardScaler()
X_train = scale_standard.fit_transform(X_train)
X_test = scale_standard.transform(X_test)


###########################################
# Apply PCA
###########################################

# Instantiate & Fit

pca = PCA(n_components = None, random_state = 42) # None = # of variables (100 in this case)

pca.fit(X_train)

# Extract the explained variance across components

eplained_variance = pca.explained_variance_ratio_
eplained_variance_cumulative = pca.explained_variance_ratio_.cumsum()


#################################################
# Plot tge exolained variance accross components
#################################################

# Create list for number of components

num_vars_list = list(range(1,101))
plt.figure(figsize =(15,10))

# plot the variance explained by each component

plt.subplot(2,1,1)
plt.bar(num_vars_list, eplained_variance)
plt.title("Variane across Principal Components")
plt.xlabel("Numer of Components")
plt.ylabel("% Variance")
plt.tight_layout()

# plot cumulative variance

plt.subplot(2,1,2)
plt.plot(num_vars_list, eplained_variance_cumulative)
plt.title("Cumulative Variane across Principal Components")
plt.xlabel("Numer of Components")
plt.ylabel("Cumulative % Variance")
plt.tight_layout()
plt.show()

# 75% of variance seem to come from approx 25 variables, we can start with this

#################################################
# Apply PCA with selected number of components
#################################################

pca = PCA(n_components = 0.75, random_state = 42) # 75% of explained variance

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

pca.n_components_


#################################################
# Apply PCA to classifier
#################################################

clf = RandomForestClassifier(random_state = 42)
clf.fit(X_train, y_train)


#################################################
# Assess Model Accuracy
#################################################

y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)

# Accuracy 0f 93% with only 24 total info by using PCA






