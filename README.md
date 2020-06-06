# loans-ml

This project predict whether online peer-to-peer loans are fully or partially funded. It was completed and presented at the 2019 InnovateGCL Competition at CapitalOne Headquarters, where it won the $100 best pitch prize in the FinTech category. 

## loan.csv

The data is a random sample of the larger file on a [Kaggle page](https://www.kaggle.com/wendykan/lending-club-loan-data). The file contains all loans issued on Lending Club (https://www.lendingclub.com/) from 2007 to 2015, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information when available. The features include loan amount, credit scores, number of finance inquiries, and zip codes.

## fully_funded.csv

This file contains all the loans in `loan.csv` that were fully funded, i.e. `loan_amnt = funded_amnt`.

## partially_funded.csv

This file contains all the loans in `loan.csv` that were partially funded, i.e. `loan_amnt > funded_amnt`.

## LCDataDictionary.csv

This spreadsheet contains descriptions for the variables in the three files above. 

## X.csv

A file with the loans from `loan.csv` after pre-processing is completed (drop redundant columns, clean formatting of certain columns, convert strings to ints, & impute missing values)

## model.py

code to use 5-fold cross-validation to train and evaluate different machine learning models on the data
