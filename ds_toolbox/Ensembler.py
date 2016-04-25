import numpy as np
from sklearn import preprocessing


class WeightingAverageEnsembler():

    def __init__(self):
        self._model = None
        self._model_weights = None

    def load_models(self, model_list, weights=None):
        self._models = model_list
        if weights is None:
            weights = [1] * len(model_list)
            weights = [w / len(model_list) for w in weights]
        self._model_weights = weights

    def predict_proba(self, x):
        n_models = len(self._models)
        n_obs = x.shape[0]
        pred_negative = np.empty((n_obs, n_models))
        pred_positive = np.empty((n_obs, n_models))

        for ii, model in enumerate(self._models):
            pred_probs = model.predict_proba(x)
            pred_negative[:, ii] = pred_probs[:, 0]
            pred_positive[:, ii] = pred_probs[:, 1]

        # standardize feature before weights
        final_negative = preprocessing.scale(pred_negative)
        final_positive = preprocessing.scale(pred_positive)

        # create final prediction prob by weighted averaging
        final_negative = (pred_negative * self._model_weights).sum(axis=1).tolist()
        final_positive = (pred_positive * self._model_weights).sum(axis=1).tolist()

        return np.array([final_negative, final_positive]).T

    def predict(self, x):
        def assign_lebel(x):
            if x[0] > x[1]:
                return 0
            else:
                return 1

        pred_prob = self.predict_proba(x)
        labels = [None] * pred_prob.shape[0]
        for ii in range(len(labels)):
            labels[ii] = assign_lebel(pred_prob[ii, :])

        return np.array(labels)
