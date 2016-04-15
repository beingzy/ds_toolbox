""" auxiliary functions for object of sklearn.ensemble.GradientBoostingClassifier

    Author: Yi Zhang <beingzy@gmail.com>
    Date: 2016/04/15
"""
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc


def cal_staged_scores(estimator, x, y, scorer=None):
    """ retrieve staged training score and test scores

    Parameters:
    ==========

    """
    def get_auc(y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    if scorer is None:
    	scorer = get_auc
    
    n_estimators = estimator.n_estimators
    scores = np.zeros((n_estimators,), dtype=np.float64)

    for ii, y_pred in enumerate(estimator.staged_predict_proba(x):
        scores[ii] = scorer(y, y_pred[:, 1])

    return scores

def plot_staged_curve(estimator, x_train, y_train, x_test, y_test, 
                      fig_size=(12, 12), scorer=None, 
                      title="Staged Evolution Curve", 
                      xlab="Boosting Iteartions",
                      ylab="Score"):
    
    train_scores = cal_staged_scores(estimator, x_train, y_train, scorer)
    test_scores = cal_staged_scores(estimator, x_test, y_test, scorer)

    fig = plt.figure(figsize=fig_size)
    plt.subplot(1, 2, 1)
    plt.title('Area Under ROC (AUC)')
    plt.plot(np.arange(n_estimators) + 1, train_score, 'b-',
             label='Training Set')
    plt.plot(np.arange(n_estimators) + 1, test_score, 'r-',
             label='Test Set')
    plt.legend(loc='upper right')
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    return fig


