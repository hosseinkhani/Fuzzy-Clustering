import matplotlib.patches as shapes
import numpy as np

from ... import FuzzyClassifierException
from ..BaseMultivariateFuzzyCluster import BaseMultivariateFuzzyCluster


class MCMeanCluster(BaseMultivariateFuzzyCluster):
    def __init__(self, high, dim):
        self.v = np.random.uniform(high, size=dim)

    def update(self, xs, us, j, m, ci):
        for j in range(self.v.shape[0]):
            self.v[j] = sum([us[:, ci, j][i]**m * xs[i][j] for i in range(len(xs))]) / sum([us[:, ci, j][i]**m for i in range(len(xs))])
        # cl = sum([sum(us[i, ci, :])**m * xs[i] for i in range(len(xs))]) / sum([sum(us[i, ci, :])**m for i in range(len(xs))])
        # self.v = self.v * 3.0 / 4.0 + cl * 1.0 / 4.0

    def distance(self, x, j):
        return (x[j]-self.v[j])**2 + (np.linalg.norm(x-self.v)**1.3)/len(x)
        # return (x[j]-self.v[j])**2 + (np.linalg.norm(x-self.v)**1.3)/len(x)

    def __repr__(self):
        return "CMeans cluster# v={0}".format(self.v)

    def draw(self):
        if self.v.shape[0] > 2:
            raise FuzzyClassifierException("draw works for 2d data not more!")
        res = shapes.Circle(xy=self.v, radius=100)
        return res

    def center(self):
        return self.v
