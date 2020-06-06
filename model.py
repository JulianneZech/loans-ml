import warnings
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
import pickle
import math
import os
import time
import re
warnings.filterwarnings('ignore', category=Warning)


def prep_data():
    """
    create separate files with partially and fully funded loans

    :return: write partially_funded.csv & fully_funded.csv
    """

    # read full data set
    data = pd.read_csv("loan.csv")
    print(data.head())

    # create data frame with only partially funded loans
    partially_funded = data[data['loan_amnt'] - data['funded_amnt'] > 0]
    print(partially_funded.shape)

    # create data frame with only fully funded loans
    fully_funded = data[data['loan_amnt'] - data['funded_amnt'] == 0]
    # take random sample with same number of records as partially_funded
    fully_funded = fully_funded.sample(n=partially_funded.shape[0], random_state=1)

    print(partially_funded.shape)
    print(partially_funded.head())
    partially_funded.to_csv("partially_funded.csv", index=False)

    print(fully_funded.shape)
    print(fully_funded.head())
    fully_funded.to_csv("fully_funded.csv", index=False)


def preprocess():
    """
    drop redundant columns, clean formatting of certain columns, convert strings to ints, & impute missing values
    :return: X: feature matrix
             y: matrix of labels
    """

    fully_funded = pd.read_csv("fully_funded.csv")
    partially_funded = pd.read_csv("partially_funded.csv")

    # columns we want to drop: either not useful for prediction, all records have the same value, or there is
    # another column with an identical purpose that is in a cleaner format already
    drop_cols = ['emp_title', 'desc', 'last_pymnt_d', 'last_credit_pull_d', 'issue_d', 'zip_code', 'title',
                 'addr_state', 'earliest_cr_line', 'pymnt_plan', 'initial_list_status']

    # drop columns where more than half of the data is missing in either data frame
    for col in fully_funded.columns:
        print(col + ": " + str(fully_funded[col].isna().sum(axis=0)))
        print(col + ": " + str(partially_funded[col].isna().sum(axis=0)))
        print('\n')
        if fully_funded[col].isna().sum(axis=0) > (fully_funded.shape[0] / 2) \
                or partially_funded[col].isna().sum(axis=0) > (partially_funded.shape[0] / 2):
            drop_cols.append(col)

    print(drop_cols)
    print('\n')

    fully_funded = fully_funded.drop(drop_cols, axis=1)
    partially_funded = partially_funded.drop(drop_cols, axis=1)
    print(fully_funded.shape)
    print(partially_funded.shape)

    # term feature is in the form: 36 months, 60 months, etc., so isolate the number and convert to integer
    partially_funded['term'] = partially_funded['term'].str.replace("\D+", '').astype(int)
    fully_funded['term'] = fully_funded['term'].str.replace("\D+", '').astype(int)

    # convert letter grades to numeric
    grade_dict = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
    fully_funded['grade'] = fully_funded['grade'].map(grade_dict)
    partially_funded['grade'] = partially_funded['grade'].map(grade_dict)

    # convert subgrades to numeric
    subgrade_dict = {"A1": 29, "A2": 28, "A3": 27, "A4": 26, "A5": 25,
                     "B1": 24, "B2": 23, "B3": 22, "B4": 21, "B5": 20,
                     "C1": 19, "C2": 18, "C3": 17, "C4": 16, "C5": 15,
                     "D1": 14, "D2": 13, "D3": 12, "D4": 11, "D5": 10,
                     "E1": 9, "E2": 8, "E3": 7, "E4": 6, "E5": 5,
                     "F1": 4, "F2": 3, "F3": 2, "F4": 1, "F5": 0}
    fully_funded['sub_grade'] = fully_funded['sub_grade'].map(subgrade_dict)
    partially_funded['sub_grade'] = partially_funded['sub_grade'].map(subgrade_dict)

    # convert durations of work to numeric
    emplength_dict = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5,
                      "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10 years": 10, "10+ years": 11}
    fully_funded['emp_length'] = fully_funded['emp_length'].map(emplength_dict)
    partially_funded['emp_length'] = partially_funded['emp_length'].map(emplength_dict)

    # find unique values for home ownership status
    print(fully_funded['home_ownership'].unique())
    print(partially_funded['home_ownership'].unique())

    # convert to numeric
    home_ownership_dict = {"RENT": 0, "MORTGAGE": 1, "OWN": 2, "OTHER": 3}
    fully_funded['home_ownership'] = fully_funded['home_ownership'].map(home_ownership_dict)
    partially_funded['home_ownership'] = partially_funded['home_ownership'].map(home_ownership_dict)

    # find unique values for verification status
    print(fully_funded['verification_status'].unique())
    print(partially_funded['verification_status'].unique())

    # convert to numeric
    verification_status_dict = {"Not Verified": 0, "Verified": 1, "Source Verified": 2}
    fully_funded['verification_status'] = fully_funded['verification_status'].map(verification_status_dict)
    partially_funded['verification_status'] = partially_funded['verification_status'].map(verification_status_dict)

    # convert loan status to numeric
    status_dict = {"Fully Paid": 5, "Current": 4, "Charged Off": 3,
                   "Does not meet the credit policy. Status:Fully Paid": 2, "Late (31-120 days)": 1,
                   "In Grace Period": 0}
    fully_funded['loan_status'] = fully_funded['loan_status'].map(status_dict)
    partially_funded['loan_status'] = partially_funded['loan_status'].map(status_dict)

    # convert loan purpose to numeric
    purpose_dict = {"debt_consolidation": 0, "medical": 1, "credit_card": 2, "home_improvement": 3, "small_business": 4,
                    "major_purchase": 5, "car": 6, "wedding": 7, "vacation": 8, "house": 9, "renewable_energy": 10,
                    "education": 11, "moving": 12, "other": 13}
    fully_funded['purpose'] = fully_funded['purpose'].map(purpose_dict)
    partially_funded['purpose'] = partially_funded['purpose'].map(purpose_dict)

    # find unique values of initial list status
    print(fully_funded['initial_list_status'].unique())
    print(partially_funded['initial_list_status'].unique())

    # convert to numeric
    status_dict = {"w": 0, "f": 1}
    fully_funded['initial_list_status'] = fully_funded['initial_list_status'].map(status_dict)
    partially_funded['initial_list_status'] = partially_funded['initial_list_status'].map(status_dict)

    # find unique values of application type
    print(fully_funded['application_type'].unique())
    print(partially_funded['application_type'].unique())

    # convert to numeric
    apptype_dict = {"Individual": 0, "Joint App": 1}
    fully_funded['application_type'] = fully_funded['application_type'].map(apptype_dict)
    partially_funded['application_type'] = partially_funded['application_type'].map(apptype_dict)

    # find unique values for flags
    print(fully_funded['hardship_flag'].unique())
    print(partially_funded['hardship_flag'].unique())
    print(fully_funded['debt_settlement_flag'].unique())
    print(partially_funded['debt_settlement_flag'].unique())

    # convert to numeric
    flag_dict = {"N": 0, "Y": 1}
    fully_funded['hardship_flag'] = fully_funded['hardship_flag'].map(flag_dict)
    partially_funded['hardship_flag'] = partially_funded['hardship_flag'].map(flag_dict)
    fully_funded['debt_settlement_flag'] = fully_funded['debt_settlement_flag'].map(flag_dict)
    partially_funded['debt_settlement_flag'] = partially_funded['debt_settlement_flag'].map(flag_dict)

    # convert disbursement method to numeric
    pmt_dict = {"Cash": 0, "DirectPay": 1}
    fully_funded['disbursement_method'] = fully_funded['disbursement_method'].map(pmt_dict)
    partially_funded['disbursement_method'] = partially_funded['disbursement_method'].map(pmt_dict)

    # impute missing values with the mean of the column for each class (fully funded & partially funded)
    for col in fully_funded:
        fully_funded[col].fillna((fully_funded[col].mean()), inplace=True)
        partially_funded[col].fillna((partially_funded[col].mean()), inplace=True)

    # assign appropriate labels
    fully_funded["label"] = 1
    partially_funded["label"] = 0

    # concatenate the data frames and shuffle the data
    X = pd.concat([fully_funded, partially_funded])
    X = X.sample(frac=1).reset_index(drop=True)
    X.to_csv("X.csv")

    # y dataframe contains class labels, X contains feature values
    y = X["label"]
    X = X.drop("label", axis=1)

    return X, y


def fit_nb(X, y):
    """
    fit and return naive bayes model
    :param X: predictor frame
    :param y: targets
    :return: nb (naive bayes model object)
    """

    nb = GaussianNB(priors=None, var_smoothing=1e-09)
    nb.fit(X, y)

    return nb


def fit_lr(X, y):
    """
    fit and return logistic regression model
    :param X: predictor frame
    :param y: targets
    :return: lr (logistic regression model object)
    """

    # search for optimal parameters
    gridsearch = GridSearchCV(
        estimator=LogisticRegression(solver='saga'),
        param_grid={
            'penalty': ['l1', 'l2'],
        },
        cv=5, verbose=0, n_jobs=-1)

    # determine best parameters
    gridsearch_result = gridsearch.fit(X, y)
    best_params = gridsearch_result.best_params_

    # declare and fit best model
    lr = LogisticRegression(penalty=best_params["penalty"], solver='liblinear', max_iter=20000,
                            random_state=False, verbose=False)
    lr.fit(X, y)

    return [lr, best_params]


def fit_rf(X, y):
    """
    fit and return random forest model
    :param X: predictor frame
    :param y: targets
    :return: rf (random forest model object)
    """

    # search for optimal parameters
    gridsearch = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid={
            'max_depth': range(3, 7),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=5, verbose=0, n_jobs=-1)

    # determine best parameters
    gridsearch_result = gridsearch.fit(X, y)
    best_params = gridsearch_result.best_params_

    # declare and fit best model
    rf = RandomForestClassifier(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],
                                random_state=False, verbose=False)
    rf.fit(X, y)

    # store feature importances in a dataframe
    feature_importances = pd.DataFrame(rf.feature_importances_, index=list(X), columns=['importances']). \
        sort_values('importances', ascending=False)

    return [rf, best_params, feature_importances]
    

def fit_xrt(X, y):
    """
    fit and return extremely randomized tree model
    :param X: predictor frame
    :param y: targets
    :return: xrt (extremely randomized tree model object)
    """

    # search for optimal parameters
    gridsearch = GridSearchCV(
        estimator=ExtraTreesClassifier(),
        param_grid={
            'max_depth': range(3, 7),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=5, verbose=0, n_jobs=-1)

    # determine best parameters
    gridsearch_result = gridsearch.fit(X, y)
    best_params = gridsearch_result.best_params_

    # declare and fit best model
    xrt = ExtraTreesClassifier(n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"],
                               random_state=False, verbose=False)
    xrt.fit(X, y)

    return [xrt, best_params]


def fit_gbm(X, y):
    """
    fit and return gradient boosted tree model
    :param X: predictor frame
    :param y: targets
    :return: gbm (gradient boosted tree model object)
    """

    # search for optimal parameters
    gridsearch = GridSearchCV(
        estimator=GradientBoostingClassifier(),
        param_grid={
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': range(3, 10),
            'n_estimators': [10, 50, 100]
        },
        cv=5, verbose=0, n_jobs=-1)

    # determine best parameters
    gridsearch_result = gridsearch.fit(X, y)
    best_params = gridsearch_result.best_params_

    # declare and fit best model
    gbm = GradientBoostingClassifier(learning_rate=best_params["learning_rate"], max_depth=best_params['max_depth'],
                                     n_estimators=best_params["n_estimators"], verbose=False)

    gbm.fit(X.values, y)

    return [gbm, best_params]


def fit_ensemble(X, y, estimators):
    """
    fit and return ensemble model
    :param X: predictor frame
    :param y: targets
    :return: ensemble (ensemble model object)
    """

    # declare VotingClassifier with one model of each type as estimators
    ensemble = VotingClassifier(estimators, voting='soft')
    ensemble.fit(X, y)

    return ensemble


def predict(X, y, model, type, params, fold, importances, model_dict):
    """
    print evaluation metrics to console
    :param X: predictor frame
    :param y: targets
    :param model: model object
    :param type: type of model
    :param params: fitted model parameters
    :param fold: number of cross-validation fold
    :param importances: random forest feature importances
    :param model_dict: dictionary mapping informal names to formal names (i.e. nb: Naive Bayes)
    """

    # get predictions
    preds = model.predict(X)

    dash = '-' * 80

    if fold == 1:
        # insert headers for columns
        print(model_dict[type] + str(" Results"))
        print(dash)
        print('{:<12s}{:<12s}{:<14s}{:>12s}{:>12s}{:>12s}'
              .format("Models", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1"))
        print(dash)

    # print scoring metrics
    print('{:<10s}{:>9.5f}{:>17.5f}{:>15.5f}{:>14.5f}{:>14.5f}'
          .format("Model " + str(fold), metrics.accuracy_score(y, preds), metrics.balanced_accuracy_score(y, preds),
                  metrics.precision_score(y, preds), metrics.recall_score(y, preds), metrics.f1_score(y, preds)))

    if fold == 5 and type != "nb" and type != "ens":
        print("\n")
        for elem in params:
            # print best hyper-parameters
            print("Best " + str(model_dict[type]) + " Hyper-parameters: " + str(elem))
        print('\n')

    if fold == 5 and type == "rf":
        for df in importances:
            # print importances
            print("Random Forest Feature Importances")
            print(df)
            print('\n')


def main():
    """
    use 5-fold cross-validation to train and evaluate different models
    """

    # prepare and pre-process data
    prep_data()
    X, y = preprocess()

    # use 5-fold cross-validation
    kf = KFold(n_splits=5)

    models = ['nb', 'lr', 'rf', 'xrt', 'gbm', 'ens']

    # map model acronyms to their proper names for convenience
    model_dict = {"nb": "Naive Bayes", "lr": "Logistic Regression", "rf": "Random Forest",
                  "xrt": "Extremely Randomized Trees Classifier", "gbm": "Gradient Boosting Trees Classifier",
                  "ens": "Ensemble Classifier"}

    obj_list = []  # store model objects
    for model in models:

        fold = 1
        params_list = []  # store best parameters
        importances_list = []  # store feature importances

        # time the creation of each type of model
        start = time.time()

        # split into training and test sets
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # train naive bayes classifier
            if model == 'nb':
                obj = fit_nb(X_train, y_train)
                obj_list.append(obj)

            # train logistic regression classifier
            elif model == 'lr':
                [obj, params] = fit_lr(X_train, y_train)
                params_list.append(params)
                obj_list.append(obj)

            # train random forest classifier
            elif model == 'rf':
                [obj, params, importances] = fit_rf(X_train, y_train)
                params_list.append(params)
                importances_list.append(importances)
                obj_list.append(obj)

            # train extremely randomized trees classifier
            elif model == 'xrt':
                [obj, params] = fit_xrt(X_train, y_train)
                params_list.append(params)
                obj_list.append(obj)

            # train gradient boosted machine classifier
            elif model == 'gbm':
                [obj, params] = fit_gbm(X_train, y_train)
                params_list.append(params)
                obj_list.append(obj)

            # train ensemble classifier
            elif model == 'ens':
                estimators = [('nb', obj_list[0]), ('lr', obj_list[5]), ('rf', obj_list[10]), ('xrt', obj_list[15]),
                              ('gbm', obj_list[20])]
                obj = fit_ensemble(X_train, y_train, estimators)

            # predict classes on test set and print evaluation metrics
            predict(X_test, y_test, obj, model, params_list, fold, importances_list, model_dict)

            fold += 1

        # print time to build classifier
        end = time.time()
        print('\n')
        print("Time to Build " + model_dict[model] + ": " + str(end - start) + " seconds")
        print('\n')


main()
