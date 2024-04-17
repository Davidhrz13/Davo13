#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:33:14 2024

@author: davidhernandez
"""

#############################################
# Association Rule Learning (Apriori Model)
#############################################

pip install apyori
#############################################
# Import Packages
#############################################

from apyori import apriori
import pandas as pd

#############################################
# Import Data
#############################################

# Import

alcohol_transaction = pd.read_csv("data/sample_data_apriori.csv")

# Drop ID Column

alcohol_transaction.drop("transaction_id", axis = 1, inplace = True)

# Modify data for apriori algorithm
# We have to provide data as list of lists, need one master list as well

transactions_list = []

for index, row in alcohol_transaction.iterrows():
    transaction = list(row.dropna())
    transactions_list.append(transaction)


#############################################
# Apply the Apriori Algorithm
#############################################

apriori_rules = apriori(transactions_list,
                        min_support = 0.003,
                        min_confidence = 0.2,
                        min_lift = 3,
                        min_length = 2,
                        max_length = 2
                        )

# 4 variables that have to be specified in algorithm and length (individual item to item comparison - combinations)

apriori_rules = list(apriori_rules) # convert from generation type to list type

apriori_rules[0] #data for first rule of list

#############################################
# Convert output to DataFrame
#############################################

product1 = [list(rule[2][0][0])[0] for rule in apriori_rules]
product2 = [list(rule[2][0][1])[0] for rule in apriori_rules]
support = [rule[1] for rule in apriori_rules]
confidence = [rule[2][0][2] for rule in apriori_rules]
lift = [rule[2][0][3] for rule in apriori_rules]

apriori_rules_df = pd.DataFrame({"product1" : product1,
                                 "product2" : product2,
                                 "support" : support,
                                 "confidence" : confidence,
                                 "lift" : lift})

#############################################
# Sort rules by descending list
#############################################

apriori_rules_df.sort_values(by = "lift", ascending = False, inplace = True)

#############################################
# Search rules
#############################################

apriori_rules_df[apriori_rules_df["product1"].str.contains("New Zealand")]









