# Functions to be used in grn prediction
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
from sklearn.svm import SVR
import pandas as pd
import numpy as np

########################################################################################################################################
# GRN PREPARATION METHODS
# #######################################################################################################################################

def prep_dataset(target_gene, tf_list, exp_df):
    '''
    Prepares training set and test set for target gene
    
    Args:
        - target_gene: target gene for the iteration (y)
        - exp_df: expression dataframe (already in pandas df format)
        - tf_list: transcription factors, which will be the predictors (X)
        
    Returns:
        - Training and Testing set to be used in model predictions
        - label for predictors, so we can subset this later
    '''
    # Get y (target) and predictor matrix (X)
    y = exp_df.loc[target_gene, :].values
    X = exp_df.loc[tf_list, :]
    
    if target_gene in tf_list.values:
        X = X.drop(index=target_gene)
    
    X_label = X.index # Predictor labels for X
    X = X.values.transpose()
    
    # Split 80:20 for test and train
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

    return X_train, X_test, y_train, y_test, X_label

# Generate gold dataset / gold truth vector
def generate_gold_dataset(gold_file_list):
    '''
    Generate gold dataset from different gold standard files,remove duplicate entries, and remove rows with regulator == target
    Args:
        - gold_file_list: list of paths to different gold standard files
    
    '''

    gold_df = pd.DataFrame()
    
    for gold_file in gold_file_list:
        gold_df = gold_df.append(pd.read_csv(gold_file, sep='\t', header=None, names = ['Regulator', 'Target']))
    
    gold_df = gold_df.drop_duplicates(subset=None, keep='first')
    
    gold_df = gold_df[~(gold_df.loc[:, 'Regulator'] == gold_df.loc[:, 'Target'])]

    return gold_df


# Generate all possible edges df
def generate_possible_edges(tf_list, exp_df):
    '''
    Generate all possible edges from a given dataset, the index will be '(TF)->(target)' and the columns will be 'actual', and 'predicted_score'
    
    Args:
        - exp_df: expression dataframe (already in pandas df format)
        - tf_list: transcription factors, which will be the predictors (X)
        
    predicted score being the score of an edge based on a prediction method (from 0 to 1)
    actual will be a binary vector, with entries either 0 or 1, depending if they are present in the gold truth file    
    '''
    
    pe_df = pd.DataFrame({"Actual": [], "Predicted score": []})
    
    pos_edges = []
    targets = []
    tfs = []
    
    for tf in tf_list:
        for target in exp_df.index:
            if tf != target:
                pos_edges.append(f'{tf}->{target}')
                targets.append(target)
                tfs.append(tf)
                
    pe_df.loc[:, 'Actual'] = np.zeros(len(pos_edges))
    pe_df.loc[:, 'Predicted score'] = np.zeros(len(pos_edges))
    pe_df.loc[:, 'TF'] = tfs
    pe_df.loc[:, 'Target'] = targets
    
    pe_df.index = pos_edges
    
    
    return pe_df
    
# Populate actual column
def populate_actual_column(pe_df, gold_df):
    '''
    Function to populate 'actual' column in pe_df, output from possible edges, 0 if there is no edge in the gold df, and 1 if there is any
    Args:
        - pe_df: possible edges df, output from generate_possible_edges function
        - gold_df: gold_df, output from generate_gold_dataset function
    
    Returns:
        - pe_df with 'actual' column populated with 1 if present in gold file, and 0 if not present
    
    '''
    
    truth_list = gold_df.loc[:, 'Regulator'] + '->' + gold_df.loc[:, 'Target']
    pe_df.loc[pe_df.index.isin(truth_list), 'Actual'] = 1

    return pe_df

def merge_dataset(df_list, merge_method, index_name=None):
    '''
    Merge a list of data frames and return a merged data frame
    
    Args:
        - df_list (required): list of data frames to merge
        - merge_method (required): a function that merges two data frames (see example below)
        - index_name (optional): index_name for the returned data frame
    
    Example Usage:
        merge_method_df = lambda x, y: x.merge(y, how='inner', left_index=True, right_index=True) # define merge_method
        df_all = merge_dataset([ko_df, nv_df, stress_df], merge_method=merge_method_df, index_name='Gene')
        
        merge_method_tf = lambda x, y: x.merge(y, how='inner', left_on='TF', right_on='TF')
        tf_all = merge_dataset([ko_tf, nv_tf, stress_tf], merge_method=merge_method_tf)
    '''

    if len(df_list) == 0: raise Exception("input list cannot be empty")

    df = df_list[0]
    for i in range(1, len(df_list)):
        df = merge_method(df, df_list[i])

    if index_name:
        df.index.name = index_name

    return df

########################################################################################################################################
# GRN PREDICTION METHODS
# #######################################################################################################################################

def grn_lasso(target_gene, tf_list, exp_df, **kwargs):
    '''
    GRN inference method using lasso regression
    
    Args:
        - target_gene: target gene for the iteration (y)
        - exp_df: expression dataframe (already in pandas df format)
        - tf_list: transcription factors, which will be the predictors (X)
        - kwargs: lassoCV arguments (alphas, cv)
        
    Returns:
        - Numpy array of type str, with list of non-zero weight predictors
    '''
    # Prep data
    X_train, X_test, y_train, y_test, X_label = prep_dataset(target_gene, tf_list, exp_df)
    
    # Use Lasso regression
    lasso_reg = LassoCV(alphas=kwargs['alphas'], cv=kwargs['cv'])
    lasso_reg.fit(X_train, y_train)
    
    # Get scores (R^2)
    # train_score = lasso_reg.score(X_train, y_train) # Note: R^2 not very good, maybe use other methods
    # test_score = lasso_reg.score(X_test, y_test)
    
    # Get weights of lasso, non zero weights are regulators
    edges = X_label + '->' + target_gene

    return edges, lasso_reg.coef_

def grn_regforest(target_gene, tf_list, exp_df, **kwargs):
    '''
    GRN inference method using regression forest. This method does not assume linearity of data.
    
    Args:
        - target_gene: target gene for the iteration (y)
        - exp_df: expression dataframe (already in pandas df format)
        - tf_list: transcription factors, which will be the predictors (X)
        - kwargs: random forest regressor arguments (n_estimators, max_depth, bootstrap, min_samples_leaf, n_jobs)
          example: n_estimators = 100, max_depth = 8, bootstrap = True, min_samples_leaf = 10, n_jobs=-1
        
    Returns:
        - Numpy array of type str, with list of non-zero weight predictors
        
    '''
    # Prep data
    X_train, X_test, y_train, y_test, X_label = prep_dataset(target_gene, tf_list, exp_df)
    
    # Use regerssion tree
    forest_reg = RandomForestRegressor(n_estimators=kwargs['n_estimators'], max_depth=kwargs['max_depth'], bootstrap=kwargs['bootstrap'],
     min_samples_leaf=kwargs['min_samples_leaf'], n_jobs=kwargs['n_jobs'])
    
    forest_reg.fit(X_train, y_train)
    
    # Get Scores (R^2)
    # train_score = forest_reg.score(X_train, y_train)
    # test_score = forest_reg.score(X_test, y_test)

    # Get feature importance and corresponding labels
    edges = X_label + '->' + target_gene
    return edges, forest_reg.feature_importances_

def grn_svr(target_gene, tf_list, exp_df, **kwargs):
    '''
    GRN inference method using Supprt Vector Regression.

    Args:
        - target_gene: target gene for the iteration (y)
        - exp_df: expression dataframe (already in pandas df format)
        - tf_list: transcription factors, which will be the predictors (X)
        - kwargs: SVR arguments (kernel)

    Returns:
       - Numpy array of type str, with list of non-zero weight predictors
    '''
    # Prep data
    X_train, X_test, y_train, y_test, X_label = prep_dataset(target_gene, tf_list, exp_df)

    # Use SVR
    SVR_reg = SVR(kernel=kwargs['kernel'])
    SVR_reg.fit(X_train, y_train)

    # Get Scores (R^2)
    # train_score = SVR_reg.score(X_train, y_train)
    # test_score = SVR_reg.score(X_test, y_test)

    # Get weights
    edges = X_label + '->' + target_gene

    # Coefs conditions
    coef = ''
    if kwargs['kernel'] == 'linear':
        coef = SVR_reg.coef_[0]
    else:
        coef = SVR_reg.dual_coef_

    return edges, coef


########################################################################################################################################
# EVALUATION METHODS
# #######################################################################################################################################

# Score using ROC Curve
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

def generate_auroc(actual_edges, pred_score, title):
    '''
    Evaluation metrics 2: Instead of using intersection over union, we can use ROC curve, since we have the scores for each edge and truth value for each edge
    
    Args:
        - final_main_df: df with actual and predicted score columns populated
    
    Returns:
        - Plots the roc curve and value of auroc
    '''
    
    # Gets the fpr and tpr
    fpr, tpr, ths = roc_curve(actual_edges, pred_score, pos_label = 1)
    
    # Gets auc score
    auc_score = auc(fpr, tpr)
    
    fig = plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = f'ROC Curve, AUC = {auc_score}')
    plt.plot([0, 1], [0, 1], color = 'navy', linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
    return auc_score

# Score using PR Curve
from sklearn.metrics import precision_recall_curve, average_precision_score
def generate_aupr(actual_edges, pred_score, title):
    precision, recall, ths = precision_recall_curve(actual_edges, pred_score, pos_label = 1)
    
    auc_score = auc(recall, precision)

    # ratio of the positive class (used to plot no-skill line)
    positive_ratio = float(sum(actual_edges)) / actual_edges.size

    fig = plt.figure(figsize=(10,10))
    plt.plot(recall, precision, color = 'darkorange', lw = 2, label = f'PR Curve, AUC = {auc_score}')
    plt.plot([0, 1], [positive_ratio, positive_ratio], color = 'navy', linestyle = '--', label = 'No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

    print(f'average precision score: {average_precision_score(actual_edges, pred_score)}')

    return auc_score, 

# Sound alert after code is finished!
from IPython.display import Audio, display
def FinallyItIsDone():
    display(Audio(url='../dataset/ragnarok-online-level-up-sound.mp3', autoplay=True))
