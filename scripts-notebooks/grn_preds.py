# Functions to be used in grn prediction
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVR

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
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.001)

    # return X_train, X_test, y_train, y_test, X_label
    return X, y, X_label

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
    
    pe_df = pd.DataFrame()
    
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
    pe_df.loc[:, 'Uniform score'] = np.repeat(0.5, len(pos_edges)) 
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
    X, y, X_label = prep_dataset(target_gene, tf_list, exp_df)
    
    # Use Lasso regression
    lasso_reg = LassoCV(alphas=kwargs['alphas'], cv=kwargs['cv'])
    lasso_reg.fit(X, y)
    
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
    X, y, X_label = prep_dataset(target_gene, tf_list, exp_df)
    
    # Use regerssion tree
    forest_reg = RandomForestRegressor(n_estimators=kwargs['n_estimators'], max_depth=kwargs['max_depth'], bootstrap=kwargs['bootstrap'],
     min_samples_leaf=kwargs['min_samples_leaf'], n_jobs=kwargs['n_jobs'])
    forest_reg.fit(X, y)

    # Get feature importance and corresponding labels
    edges = X_label + '->' + target_gene
    return edges, forest_reg.feature_importances_

def grn_linear_svr(target_gene, tf_list, exp_df, **kwargs):
    '''
    GRN inference method using Supprt Vector Regression, linear kernel.
    Hyperparameter to be optimized are C and epsilon

    Args:
        - target_gene: target gene for the iteration (y)
        - exp_df: expression dataframe (already in pandas df format)
        - tf_list: transcription factors, which will be the predictors (X)
        - kwargs: SVR arguments (kernel)

    Returns:
       - Numpy array of type str, with list of non-zero weight predictors
    '''
    
    # Prep data
    X, y, X_label =     prep_dataset(target_gene, tf_list, exp_df)
    
    # List candidates C and epsilon
    Cs = [0.01, 0.1, 1, 10, 100]
    epsilons = [0, 0.01, 0.1, 1, 10]
    
    # Get tuples of all possible combinations
    C_eps_list = []
    for C in Cs:
        for eps in epsilons:
            C_eps_list.append((C, eps))
    
    
    # Get cross validation scores
    C_eps_mean_score = []
    for C_eps_tup in C_eps_list:
        curr_C = C_eps_tup[0]
        curr_eps = C_eps_tup[1]
        
        svr_reg = LinearSVR(epsilon = curr_eps, C = curr_C)
        svr_reg_scores = cross_val_score(svr_reg, X, y, cv = 5)
        C_eps_mean_score.append(np.mean(svr_reg_scores))
    
    # Get argmax of validation score, and choose that as hyperparam, then get weights
    best_C_eps = C_eps_list[np.argmax(C_eps_mean_score)]
    svr_reg_best = LinearSVR(epsilon = best_C_eps[1], C = best_C_eps[0])
    svr_reg_best.fit(X, y)
    
    # Get weights and edges
    weights = svr_reg_best.coef_
    edges = X_label + '->' + target_gene
    
    return edges, weights


#########################################################################################################################################
# Standardize scores
# #######################################################################################################################################

from sklearn.preprocessing import StandardScaler

def standardize_scores(target_gene, scores_df, method_name):
    '''
    Function for standardizing weights per target gene. Ignore zero entries, normalize others
    
    Args:
        - target_gene: name of target gene (string)
        - scores_df: scores df
        - method_name: method of prediction
        
    Returns:
        - tuple with ([index names], [noramlized scores])
    '''
    
    # Filtered by target and specific method column
    filtered_df = scores_df[scores_df.loc[:, 'Target'] == target_gene].loc[:, [f'{method_name} scores']]
    
    # Get nonzero values only
    nonzero_df = filtered_df[filtered_df.loc[:, f'{method_name} scores'] != 0]
    
    if len(nonzero_df) == 0:
        return filtered_df.index, np.repeat(0, len(filtered_df))

    # Normalize these values, so it has mean of 0 and standard deviation of 1
    stand_scores = StandardScaler().fit_transform(nonzero_df.loc[:, f'{method_name} scores'].values.reshape(len(nonzero_df), 1))
    
    # Add the standardized column to filtered df
    filtered_df.loc[:, f'Standardized {method_name} scores'] = 0
    filtered_df.loc[nonzero_df.index, f'Standardized {method_name} scores'] = stand_scores
    
    # Target gene index, normalized score
    target_gene_index = filtered_df.loc[nonzero_df.index, f'Standardized {method_name} scores'].index
    stand_scores_values = filtered_df.loc[nonzero_df.index, f'Standardized {method_name} scores'].values

    return target_gene_index, stand_scores_values

#########################################################################################################################################
# EVALUATION METHODS
# #######################################################################################################################################

# Score using ROC Curve
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

def generate_auroc(actual_edges, pred_score, title='', show=True):
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

    if show:
        plt.show()
    
    return auc_score

# Score using PR Curve
from sklearn.metrics import precision_recall_curve, average_precision_score
def generate_auprc(actual_edges, pred_score, title='', show=True):
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

    if show:
        plt.show()
        print(f'average precision score: {average_precision_score(actual_edges, pred_score)}')

    return auc_score

# Sound alert after code is finished!
from IPython.display import Audio, display
def FinallyItIsDone():
    display(Audio(url='../dataset/ragnarok-online-level-up-sound.mp3', autoplay=True))
