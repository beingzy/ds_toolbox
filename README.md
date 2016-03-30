### DS Toolbox (Data Scienec Toolbox)
The python library aims to provide high-level functions to help data scientist
develop prototype models quickly and easily. 


### How to use:
1. model selection (classification problem)
```{python}
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from ds_toolbox.model_selection_functions import classifier_evaluator
from ds_toolbox.model_selection_functions import model_selection_cv 
from ds_toolbox.model_selection_functions import eval_barchart

# define model objects
LogisticModel = LogisticRegression(penalty='l1', C=0.1, max_iter=1000)
Boosting = AdaBoostClassifier()
RF = RandomForestClassifier()
SVM = SVC( probability=True, random_state=_RANDOM_STATE )

# package classification algorithms in dictionary
classifiers = {"Logistic Regression (L1: Lasso)": LogisticModel, 
               "SVM (Kernel=RDF)": SVM,
               "Boosting Tree": Boosting, 
               "Random Forest": RF}
               
# evaluation model with stratified k-fold cross-validation 
model_slc_report = model_selection_cv(classifiers, xdata, ydata, \
                                      k=10, random_state=_RANDOM_STATE, \
                                      eval_func = classifier_evaluator )

## ##############################
## plot charts to visualize performance 
auc_barchart = eval_barchart( model_slc_report, 
                              test_metrics_colnames = ['test_auc_mean', 'test_auc_std'], 
                              train_metrics_colnames = ['train_auc_mean', 'train_auc_std'], 
                              label_colname = 'model_name' )
    
sensitivity_barchart = eval_barchart( model_slc_report, 
                              test_metrics_colnames = ['test_sensitivity_mean', 'test_sensitivity_std'], 
                              train_metrics_colnames = ['train_sensitivity_mean', 'train_sensitivity_std'], 
                              label_colname = 'model_name' )
                              
```