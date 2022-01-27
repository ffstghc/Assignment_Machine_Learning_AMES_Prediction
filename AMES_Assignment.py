#####################
## IMPORT PACKAGES ##
#####################
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


#################
## USER INPUTS ##
#################
# Data Set Selection
print("Input dataset # (1) AZ descriptors, (2) ecfi1024, (3) oeselma, (4) wfp")
ds_ind = int(input()) - 1  # Dataset index

# Perform Gridsearch (yes / no)
print("Do you want to optimize the parameters with scikit-learn's Gridsearch?", "\n", "!! CAN TAKE VERY LONG (1 h +) !!", "\n", "(1) = no, (2) = yes")
gs = int(input())


#################################
## HYPERPARAMATER OPTIMIZATION ##
#################################
def perform_gridSearch(parameters, classifier):
    grid = GridSearchCV(estimator=classifier, param_grid=parameters, cv=3, n_jobs=-1, verbose=4)
    grid_results = grid.fit(X_train, y_train)
    print("Best: {0}, using {1}".format(grid_results.cv_results_['mean_test_score'], grid_results.best_params_))
    results_df = pd.DataFrame(grid_results.cv_results_)
    return results_df


##################
## IMPORTS DATA ##
##################
# Create filename lists
filenames_train = ['ames.AZ_descriptors_tr.txt',
                   'ames.ecfi1024_tr.txt',
                   'ames.oeselma_tr.txt',
                   'ames.wfp_tr.txt']

filenames_test = ['ames.AZ_descriptors_ts.txt',
                  'ames.ecfi1024_ts.txt',
                  'ames.oeselma_ts.txt',
                  'ames.wfp_ts.txt']


def import_data(file_names, mode):

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')  # Create Imputer

    imported = [pd.read_csv(data, delimiter="\t", dtype='str', na_values="miss") for data in file_names]

    if mode == 'train_val':

        data_validation_full = []
        data_train_full = []
        data_pre_imputed = []

        for i in range(0, len(imported)):

            data_pre_imputed.append(imported[i])

            for j in range(0, len(data_pre_imputed)):
                X = data_pre_imputed[j].drop(['ID', 'ames', 'set'], axis=1).astype(float)
                y = data_pre_imputed[j]['ames']

            X = imp.fit_transform(X)

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=321)

            train = [X_train, y_train]
            val = [X_val, y_val]

            data_train_full.append(train)
            data_validation_full.append(val)

        return data_train_full, data_validation_full

    else:

        data_test_full = [imported[i].drop(['ID', 'ames', 'set'], axis=1).astype(float) for i in range(0, len(imported))]

        # Impute missing values
        for i in range(0, len(imported)):
            data_test_full[i] = imp.fit_transform(data_test_full[i])

        return data_test_full


# Import full and data without NAs
dat_train, dat_val = import_data(filenames_train, mode='train_val')  # Train / validation data
dat_test = import_data(filenames_test, mode='test')  # Test data

# Create Data Lists and make choice dependet on input
X_train = [dat_train[file_input][0] for file_input in range(0, len(dat_train))][ds_ind]  # Training data
y_train = [dat_train[file_input][1] for file_input in range(0, len(dat_train))][ds_ind]  # Training data
X_test = [dat_val[file_input][0] for file_input in range(0, len(dat_train))][ds_ind]  # Validation data
y_test = [dat_val[file_input][1] for file_input in range(0, len(dat_train))][ds_ind]  # Validation data
X_test_real = [dat_test[file_input] for file_input in range(0, len(dat_test))][ds_ind]  # Test data for predictions


#################
## STANDARDIZE ##
#################
std_sclr = StandardScaler()
X_train_scaled = std_sclr.fit_transform(X_train)
X_test_scaled = std_sclr.fit_transform(X_test)


####################
## Visualize Data ##
####################
## PRINT SUMMARY OUTPUT
def create_summary(x_test_val, y_test_val, y_pred_val, input_type, model):
    cfm = confusion_matrix(y_test, y_pred)
    accur = round((cfm[1, 1] + cfm[0, 0]) / cfm[:].sum(), 4)
    print('\n', '\n', '\n', input_type)
    print('Classification Report:', '\n', classification_report(y_test_val, y_pred_val))  # Classification Report
    print('Confusion Matrix:', '\n', confusion_matrix(y_test_val, y_pred_val))  # Confusion Matrix
    print('Cross-Validation Score:', '\n', clf.score(x_test_val, y_test))  # Cross-Val Score
    print('Accuracy from confusion matrix:', '\n', accur)
    return accur


## HEATMAP FROM CONFUSION MATRIX
def create_htmp_from_cnf(y_test, y_pred, algo, accur):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    class_names = ["AMES-", "AMES+"]  # Names  of classes
    ax = plt.axes()
    sns.heatmap(pd.DataFrame(cnf_matrix), ax=ax, annot=True, cmap="YlGnBu", fmt='g')
    num_rows, num_cols = X_test.shape
    n_features = num_cols
    var = ' '.join([str(algo), "\n", "Accuracy:", str(accur), "Features:", str(n_features)])  # Join Algorithm name and Accuracy
    ax.set_title(var)  # Set title to joined string
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    tick_marks = [0.5, 1.5]
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.show()


#############################
## SUPPORT VECTOR MACHINES ##
#############################
if gs == 2:  # USE GRIDSEARCH RESULTS
    gridsearched = perform_gridSearch({'kernel': ('linear', 'rbf'), 'C': [0.1, 1]}, svm.SVC())
    kern = gridsearched[gridsearched['rank_test_score'] == 1].param_kernel.tolist()
    c = gridsearched[gridsearched['rank_test_score'] == 1].param_C.tolist()
    clf = svm.SVC(kernel=str(kern[0]), C=c[0])
else:
    clf = svm.SVC(kernel='rbf', C=1)


clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

accur = create_summary(X_test_scaled, y_test, y_pred, 'SUMMARY: SUPPORT VECTOR MACHINE', clf)

create_htmp_from_cnf(y_test, y_pred, 'Support Vector Machine', accur)


#########################
## K NEAREST NEIGHBORS ##
#########################
if gs == 2:  # USE GRIDSEARCH RESULTS
    gridsearched = perform_gridSearch({'n_neighbors': [1, 10, 50, 100], 'leaf_size': [1, 10, 100]}, KNeighborsClassifier())
    n_eighs = gridsearched[gridsearched['rank_test_score'] == 1].param_n_neighbors.tolist()
    l_size = gridsearched[gridsearched['rank_test_score'] == 1].param_leaf_size.tolist()
    neigh = KNeighborsClassifier(n_neighbors=int(n_eighs[0]), leaf_size=int(l_size[0]))
else:
    neigh = KNeighborsClassifier(n_neighbors=3, leaf_size=2)

neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)

accur = create_summary(X_test, y_test, y_pred, 'SUMMARY: K NEAREST NEIGHBORS', neigh)

create_htmp_from_cnf(y_test, y_pred, 'K Nearest Neighbors', accur)


######################
## GRADIENT BOOSTED ##
######################
if gs == 2:  # USE GRIDSEARCH RESULTS
    gridsearched = perform_gridSearch({'n_estimators': [1, 10, 100], 'learning_rate': [1, 10], 'max_depth': [0.1, 1, 10]}, GradientBoostingClassifier())
    n_ests = gridsearched[gridsearched['rank_test_score'] == 1].param_n_estimators.tolist()
    l_rate = gridsearched[gridsearched['rank_test_score'] == 1].param_learning_rate.tolist()
    max_d = gridsearched[gridsearched['rank_test_score'] == 1].param_max_depth.tolist()
    clf = GradientBoostingClassifier(n_estimators=int(n_ests[0]), learning_rate=int(l_rate[0]), max_depth=max_d[0])
else:
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=69)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accur = create_summary(X_test, y_test, y_pred, 'SUMMARY: GRADIENT BOOSTED', clf)

create_htmp_from_cnf(y_test, y_pred, 'Gradient Boosted', accur)


######################
## NEURONAL NETWORK ##
######################
if gs == 2:  # USE GRIDSEARCH RESULTS
    gridsearched = perform_gridSearch({'hidden_layer_sizes': [10, 100, 500, 1000], 'learning_rate_init': [0.1, 1, 10], 'max_iter': [1, 10, 100, 500]}, MLPClassifier())
    n_hdl = gridsearched[gridsearched['rank_test_score'] == 1].param_hidden_layer_sizes.tolist()
    l_rate = gridsearched[gridsearched['rank_test_score'] == 1].param_learning_rate_init.tolist()
    max_iter = gridsearched[gridsearched['rank_test_score'] == 1].param_max_iter.tolist()
    clf = MLPClassifier(hidden_layer_sizes=n_hdl[0], learning_rate_init=l_rate[0], random_state=1, max_iter=max_iter[0], early_stopping=True, verbose=5)
else:
    clf = MLPClassifier(hidden_layer_sizes=500, learning_rate_init=0.1, random_state=1, max_iter=300,
                        early_stopping=True, verbose=5)

clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

accur = create_summary(X_test_scaled, y_test, y_pred, 'SUMMARY: NEURAL NETWORK', clf)

create_htmp_from_cnf(y_test, y_pred, 'Neural Network', accur)


###################
## RANDOM FOREST ##
###################
if gs == 2:  # USE GRIDSEARCH RESULTS
    gridsearched = perform_gridSearch({'max_depth': [1, 10, 100, 200, 500, 1000], 'n_estimators': [1, 10, 100, 500, 1000, 1500]}, RandomForestClassifier())
    max_depth = gridsearched[gridsearched['rank_test_score'] == 1].param_max_depth.tolist()  # Get Gridsearch Results as parameters
    n_ests = gridsearched[gridsearched['rank_test_score'] == 1].param_n_estimators.tolist()
    clf = RandomForestClassifier(max_depth=int(max_depth[0]), n_estimators=int(n_ests[0]))
else:
    clf = RandomForestClassifier(max_depth=100, n_estimators=500, random_state=69)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accur = create_summary(X_test, y_test, y_pred, 'SUMMARY: RANDOM FOREST', clf)

create_htmp_from_cnf(y_test, y_pred, 'Random Forest', accur)

# Predict for unknown data and export txt
y_pred_real = clf.predict(X_test_real)
y_2 = pd.DataFrame(y_pred_real)
y_2.columns = ["ames"]
y_2.to_csv('ames_predictions_ecfi1024.txt', header=True, index=False, sep=',')
