import numpy as np
from collections import Counter as counter
import math

class NBC:
    def __init__(self, feature_types, num_classes):
        assert 0 < len(feature_types) and 0 < num_classes
        self.D = len(feature_types)
        self.C = num_classes
        self.feature_types = feature_types
        self.param = self._make_2d_array(self.C, self.D)
        self.pi = [None] * self.C
        self.enumToLabel = {}
        self.labelToEnum = {}

    def fit(self, X, y):
        '''
        Generate internal parameters given data matrix X and labels y
        '''
        assert X.shape[0] == y.shape[0] and X.shape[1] == self.D
        N = X.shape[0]
        self._reset_state()
        self._enumerate_labels(y)
        y = self._toEnumArr(y)

        for j in range(self.D):
            cnt = [[X[i][j] for i in range(N) if y[i] == c] for c in range(self.C)]
            for c in range(self.C):
                if self.feature_types[j] == "r":
                    self.param[c][j] = self._fit_gaussian(cnt[c])
                else:
                    raise RuntimeError("Unknown feature type")

        cnt = counter(y)
        for c, n in cnt.items():
            self.pi[c] = n / N

    def predict(self, X):
        assert self.pi[0] is not None

        res = []
        for i in range(X.shape[0]):
            res.append(self._predict_datapoint(X[i]))
        return res

    def _predict_datapoint(self, x):
        maxClass = 0
        maxLogProb = math.log(self.pi[0]) + math.log(self._calc_prob(x,0))
        for c in range(1,self.C):
            curLogProb = math.log(self.pi[c]) + math.log(self._calc_prob(x,c))
            if curLogProb > maxLogProb:
                maxClass = c
                maxLogProb = curLogProb
        return self._toLabel(maxClass)

    def _calc_prob(self, x,c):
        '''
        Returns p(xnew | y = c, theta)
        '''
        res = 1
        for j in range(len(x)):
            feature_type = self.feature_types[j]
            if feature_type == "r":
                res *= self._p_gaussian(x[j],self.param[c][j])
            else:
                raise RuntimeError("Unknown feature type")
        return res

    def _fit_gaussian(self, vals):
        '''
        Given an array of floats *vals*, return the parameters for MLE Gaussian in form [mu, sigma^2]
        '''
        if len(vals) == 0: return [-1000,1]

        mu = sum(vals) / len(vals)
        sigma2 = sum([(x-mu)**2 for x in vals]) / len(vals) + 1e-6
        return [mu, sigma2]

    def _p_gaussian(self, x, param):
        mu, sigma2 = param
        return 1 / ( (2*np.pi*sigma2) ** 0.5) * np.e ** -((x-mu)**2 / (2*sigma2)) + 1e-6

    def _fit_bernoulli(self, vals):
        pass

    def _p_bernoulli(self, x):
        pass

    def _make_2d_array(self, nrow, ncol):
        return [[None] * ncol for r in range(nrow)]

    def _reset_state(self):
        self.pi = [1e-6] * self.C
        self.enumToLabel = {}
        self.labelToEnum = {}

    def _enumerate_labels(self, y):
        labels = list(set(y))
        assert len(labels) <= self.C

        for i, v in enumerate(labels):
            self.labelToEnum[v] = i
            self.enumToLabel[i] = v

    def _toEnumArr(self, y):
        res = []
        for l in y:
            res.append(self._toEnum(l))
        return res

    def _toEnum(self, l):
        return self.labelToEnum[l]

    def _toLabel(self, c):
        return self.enumToLabel[c]
        

# Comments:
# As we are doing the calculations in log space, the probabilities can never be 0. So we need to add 1e-6 to self._calc_prob.
# We also need to initialize pi to 1e-6 when we fit a model if the data set is small and there are classes which didn't appear a single time, they will be given a non-zero, low probability. We also need to set the parameters of these classes to be in a way that deviates from all data. This way, they will never be selected (this avoids the issue from enumToLabel key error)
