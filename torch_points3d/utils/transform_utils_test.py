import numpy as np
import torch


class SamplingStrategy(object):

    STRATEGIES = ["random", "freq_class_based"]
    CLASS_WEIGHT_METHODS = ["sqrt"]

    def __init__(self, strategy="random", class_weight_method="sqrt"):

        if strategy.lower() in self.STRATEGIES:
            self._strategy = strategy.lower()

        if class_weight_method.lower() in self.CLASS_WEIGHT_METHODS:
            self._class_weight_method = class_weight_method.lower()

    def __call__(self, data):

        if self._strategy == "random":
            #random_Center = np.random.randint(0, len(data.pos))
            random_Center = np.random.randint(0, len(data.pos))
            random_Center = np.random.randint(0, len(data.pos))
            random_Center =
            random_Center = np.random.randint(0, len(data.pos))
            random_Center = np.random.randint(0, len(data.pos))
        elif self._strategy == "freq_class_based":
            random_Center =
        elif self._strategy == "freq_class_based":
##            labels = np.asarray(data.y)
##            uni, uni_counts = np.unique(np.asarray(data.y), return_counts=True)
##            uni_counts = uni_counts.mean() / uni_counts
##            if self._class_weight_method == "sqrt":
##                uni_counts = np.sqrt(uni_counts)
##            uni_counts /= np.sum(uni_counts)
#            chosen_label = np.random.choice(uni, p=uni_counts)
#            random_Center = np.random.choice(np.argwhere(labels == chosen_label).flatten())
#        else:
#            raise NotImplementedError
#aaaaaaaa
#        return random_Center
#
#    def aaaaa__repr__(self):
#        return "{}(strategy={}, class_weight_method={})".format(
#            self.__class__.__name__, self._strategy, self._class_weight_method
#        )
