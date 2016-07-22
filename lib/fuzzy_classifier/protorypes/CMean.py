import matplotlib.patches as shapes
import numpy as np

from ... import FuzzyClassifierException
from ..BaseFuzzyCluster import BaseFuzzyCluster


class CMeanCluster(BaseFuzzyCluster):
    def __init__(self, high, dim):
        self.r = np.random.uniform(high/5)
        self.v = np.random.uniform(high, size=dim)

    def _update(self, xs, uis, m):
        self.v = sum([uis[i]**m * xs[i] for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])
        self.r = sum([uis[i]**m * np.linalg.norm(xs[i]-self.v) for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])
        # self.v = sum([uis[i] * xs[i] for i in range(len(xs))]) / sum(uis)
        # self.r = sum([uis[i] * np.linalg.norm(xs[i]-self.v) for i in range(len(xs))]) / sum(uis)

    def distance(self, x):
        return np.linalg.norm(x-self.v)**2

    def __repr__(self):
        return "CMeans cluster# v={0} r={1}".format(self.v, self.r)

    def draw(self):
        if self.v.shape[0] > 2:
            raise FuzzyClassifierException("draw works for 2d data not more!")
        res = shapes.Circle(xy=self.v, radius=self.r)
        return res

    def center(self):
        return self.v
