#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:11:49 2024

@author: davidhernandez
"""

#########################################
# Casual Impact Analysis
#########################################

pip install pycausalimpact

#########################################
# Import required packages
#########################################

from causalimpact import CausalImpact
import pandas as pd 

#########################################
# Import and Create Data
#########################################

# Import data tables

transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name = "transactions")
campaign_data = pd.read_excel("data/grocery_database.xlsx", sheet_name = "campaign_data")


# Aggregate transactions data to customer data level

customer_daily_sales = transactions.groupby(["customer_id", "transaction_date"])["sales_cost"].sum().reset_index()

# Merge on the signup flag

customer_daily_sales = pd.merge(customer_daily_sales, campaign_data, how = "inner", on = "customer_id")

# Pivot the data to aggregate daily sales by signup group

causal_impact_df = customer_daily_sales.pivot_table(index = "transaction_date",
                                                    columns = "signup_flag",
                                                    values = "sales_cost", 
                                                    aggfunc = "mean")


# Provide a frequency for our DataTimeIndex (avoids a warning message)
# daily dataframe 
causal_impact_df.index.freq = "D"

# For causal impact we need the impacted group in the first column

causal_impact_df = causal_impact_df[[1,0]] #flipping order of columns


# Rename columns for something more meaningful

causal_impact_df.columns = ["member", "non_member"]


#########################################
# Apply Causal Impact
#########################################

pre_period = ['2020-04-01','2020-06-30']
post_period = ['2020-07-01','2020-09-30']

ci = CausalImpact(causal_impact_df, pre_period, post_period)


#########################################
# Plot the impact
#########################################

ci.plot()

##########################################
# Extract the summary statistics & report
##########################################

print(ci.summary())

print(ci.summary('report'))





