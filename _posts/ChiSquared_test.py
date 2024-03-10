#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 21:00:44 2024

@author: davidhernandez
"""
#AB Testing -  Task for ABC Grocery Store
# Chi squared test for independence -  binary response for two mailing campaigns
 
import pandas as pd
from scipy.stats import chi2_contingency, chi2

# Import data

campaign_data = pd.read_excel("grocery_database.xlsx", sheet_name = "campaign_data")

# Filter our data 

campaign_data = campaign_data.loc[campaign_data["mailer_type"] != "Control"]

# Summarise to get our observed frequencies

observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"]).values # .values to return an array

mailer1_signup_rate = 123 / (252 + 123)
mailer2_signup_rate = 127 / (209 + 127)
print(mailer1_signup_rate, mailer2_signup_rate)

# State hypothesis & set acceptance rate

null_hypothesis = "Theres is no relationship between mailer type and sign up rate, they are independent"
alternative_hypothesis = "There is a relationship, they are not independet"
acceptance_criteria = 0.05

# Calculate expected frequencies & cbi squared statistic 

chi2_statistic, p_value, dof, expected_values = chi2_contingency(observed_values, correction = False )
print(chi2_statistic, p_value)


# Critical value for our test

critical_value = chi2.ppf(1 - acceptance_criteria, dof) #percentage point function -find the CV based on acceptance crtieria
print(critical_value)

# print results (logic)

if chi2_statistic >= critical_value:
    print(f"As our chi-squared statistic of {chi2_statistic} is higher than our critical value of {critical_value} - we reject null hypothesis, and conclude that: {alternate_hypothesis}")
else: 
    print(f"As our chi-squared statistic of {chi2_statistic} is lower than our critical value of {critical_value} - we retain null hypothesis, and conclude that: {null_hypothesis}")
    
# with the p-value of 0.16 (lower than ), we can also conclude that we can't rehect H0, lower than the acceptance criteria

if p_value <= acceptance_criteria:
    print(f"As our p_value of {p_value} is lower than our acceptance criteria of {acceptance_criteria} - we reject null hypothesis, and conclude that: {alternate_hypothesis}")
else: 
    print(f"As our p_value of {p_value} is higher than our acceptance criteria of {acceptance_criteria} - we retain null hypothesis, and conclude that: {null_hypothesis}")

        