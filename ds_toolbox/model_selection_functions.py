""" functions develop to support model development

    Author: Yi Zhang <uc.yizhang@gmail.com>
    Date: JAN/30/2016
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
import matplotlib.pyplot as plt 
from tqdm import tqdm


def classifier_evaluator(model, x, y):
    """ generate evaluation report on binary classifier
        ROCAUC, F1-score, Accuracy, Sensitivity, Specificity,
        Precision

        Parameters:
        ===========
        model: fitted classification model
        x: ndarray
        y: ndarray
    """
    y_true = y

    y_probas_ = model.predict_proba(x)[:, 1]
    y_pred = model.predict(x)

    fpr, tpr, thresholds = roc_curve(y_true, y_probas_)
    roc_auc = auc(fpr, tpr)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tn + fp + fn + tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tn / (tn + fp)
    f1_score_val = f1_score(y_true, y_pred)

    res = {"roc_auc": roc_auc, "f1_score": f1_score_val, "accuracy": accuracy,
           "sensitivity": sensitivity, "specificity": specificity,
           "precision": precision}
    return res
    

def model_selection_cv( models, x, y, k=5, eval_func=None, random_state=None):
    """ framework for model selection based on stratified 
        cross-validation
    
        Parameters:
        ----------
        * models: {dictionary}, key: model label, value: learner object
        * x: {np.array}, predictor data 
        * y: {np.array}, response variable data
        * k: {integer}, the number of folds in cross-validation
        * random_state: {integer}, the random state set for replication
        * eval_func: {function}, return evaulation score
    """
    # stratified cross_validation
    cv = StratifiedKFold( y, n_folds=k, shuffle=False, random_state=random_state)
    tot_models = len( models.keys() )
    tot_iter = tot_models * k 
    
    pbar = tqdm(total=tot_iter)

    train_reports, test_reports = [], []

    for jj, model_name in enumerate(models):
        model = models[model_name]
        # cross-validation evaluation containers
        train_scores = []
        test_scores = []
        # print( "--- model: {}'s cross-validation test ----".format( model_name ) )
        for ii, (train_idx, test_idx) in enumerate(cv):
            # retrieve data for relevant usage
            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]
            # training model 
            model.fit( x_train, y_train )
            # evaluation model
            train_score = eval_func( model, x_train, y_train )
            train_score["model_name"] = model_name
            test_score = eval_func( model, x_test, y_test )
            test_score["model_name"] = model_name
            
            train_reports.append( train_score )
            test_reports.append( test_score )
            
            pbar.update()
            
    pbar.close()

    # convert list of performance records into dataframe
    train_reports = DataFrame(train_reports)
    test_reports = DataFrame(train_reports)

    metrics_names = [feat for feat in train_reports.columns.tolist() if feat != "model_name"]

    train_reports.sort_values(by=["model_name"])
    train_reports = train_reports[["model_name"] + metrics_names]
    test_reports.sort_values(by=["model_name"])
    test_reports = test_reports[["model_name"] + metrics_names]

    return train_reports, test_reports
    
    
def eval_barchart(df, 
                  test_metrics_colnames, 
                  train_metrics_colnames,
                  label_colname = None, 
                  ylabel = 'Performance Score' ):
    """ plot model evaluation performance 
    
        Parameters:
        ----------
        * df: {pandas.DataFrame}
        * label_colname: {string}, the column name for colum keeping model name 
        * test_metrics_colnames: {list}, columns for test metric mean, columns for test metric std.
        * train_metrics_colnames: {list}, columns for train metric mean, columns for train metric std.
        * ylabel: {string} y-axis label 
    """    
    if ( label_colname==None ):
        label_colname = df.columns[0]
    
    labels = df[ label_colname ].tolist()
    
    test_mean = df[ test_metrics_colnames[0] ].tolist()
    test_std  = df[ test_metrics_colnames[1] ].tolist()
    
    train_mean = df[ train_metrics_colnames[0] ].tolist()
    train_std  = df[ train_metrics_colnames[1] ].tolist()
    
    # graphic parameter
    N = len(labels)
    ind = np.arange(N)
    width = 0.35 # the width of the bars
    
    fig, ax = plt.subplots()
    # plot 1st test error
    rects1 = ax.bar(ind, test_mean, width, color='g', yerr = test_std)
    # plot 2nd train error
    rects2 = ax.bar(ind + width, train_mean, width, color='b', yerr = train_std)

    # add some text for labels, title and axes ticks
    ax.set_ylim( 0, 1 )
    ax.set_ylabel( ylabel )
    ax.set_title( 'Evaluation Results Bar-Chart' )
    ax.set_xticks( ind + width )
    ax.set_xticklabels( labels, rotation=20 )
    ax.legend( ( rects1[0], rects2[0] ), ('Test', 'Train') )

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 
                    1.05*height,
                    '%d' % int(height),
                     ha='center', va='bottom')
    
    autolabel( rects1 )
    autolabel( rects2 )
    
    return fig 
