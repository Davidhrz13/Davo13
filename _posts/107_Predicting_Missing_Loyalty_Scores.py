#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 19:43:34 2024

@author: davidhernandez
"""

# Import packages

import pandas as pd
import pickle

# Import customers for scoring

to_be_scored = pickle.load(open("abc_regression_scoring.p", "rb"))

# Import model and model objects

regressor = pickle.load(open("Data/ramdom_forest_regression_model.p", "rb"))
one_hot_encoder = pickle.load(open("Data/ramdom_forest_regression_ohe.p", "rb"))

# Drop unused columns

to_be_scored.drop(["customer_id"], axis = 1, inplace = True)

# Drop missing values 

to_be_scored.dropna(how = "any", inplace = True)

# Apply One hot encoder to gender
# Changes to code to not fit again, just apply the estanciated object loaded with pickle
categorical_vars = ["gender"]
encoder_vars_array = one_hot_encoder.transform(to_be_scored[categorical_vars])
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)
encoder_vars_df = pd.DataFrame(encoder_vars_array, columns = encoder_feature_names)
to_be_scored = pd.concat([to_be_scored.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis = 1) #drop true is for aligning rows
to_be_scored.drop(categorical_vars, axis = 1, inplace = True)

# Make the predictions!

loyalty_predictions = regressor.predict(to_be_scored)

