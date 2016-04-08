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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from ds_toolbox.model_selection_functions import model_selection_cv 
from ds_toolbox.model_selection_functions import classifier_evaluator
from ds_toolbox.model_selection_functions import eval_barchart

# define model objects
LogisticModel = LogisticRegression(penalty='l1', C=0.1, max_iter=1000)
Boosting = AdaBoostClassifier()
RF = RandomForestClassifier()
SVM = SVC( probability=True)

# package classification algorithms in dictionary
classifiers = {"Logistic Regression (L1: Lasso)": LogisticModel, 
               "SVM (Kernel=RDF)": SVM,
               "Boosting Tree": Boosting, 
               "Random Forest": RF}
               
# evaluation model with stratified k-fold cross-validation 
# cross-validation results were exported as pandas.DataFrames.
train_reports, test_reports = model_selection_cv(classifiers, xdata, ydata,
                                      k=10, random_state=_RANDOM_STATE, 
                                      eval_func = classifier_evaluator )
```