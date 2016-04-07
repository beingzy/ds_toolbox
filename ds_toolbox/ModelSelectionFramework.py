""" objective introduction

Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/04/07
"""



class ModelSelectionFrameworkMixin(object):

    def __init__(self, name='', type=0, **kwargs):
        """ initiate instance of ModelSelectionFramework

        Parameters:
        ===========
        type: 0 or 1,
            0: regression problem
            1: classification problem
        """
        self._learners = None
        self._evaluation_matrices = None

        if "learners" in kwargs.keys():
            self.load_learner(kwargs['learners'])
        if "metrics" in kwargs.keys():
            self.load_evaluation_metrics(kwargs["metrics"])
        if "sampler" in kwargs.keys():
            self.load_sampler(kwargs["sampler"])

    def load_learners(self, learners):
        self._learners = learners

    def load_evaluation_metrics(self, metrics):
        self._matrices = metrics

    def load_sampler(self, sampler):
        self._sampler = sampler

    def run(self):
        pass
