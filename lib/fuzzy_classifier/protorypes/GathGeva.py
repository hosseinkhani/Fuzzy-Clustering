import matplotlib.patches as shapes
import numpy as np

from ... import FuzzyClassifierException
from ..BaseFuzzyCluster import BaseFuzzyCluster


class GGCluster(BaseFuzzyCluster):
    def __init__(self, high, dim):
        self.p = 100 + np.random.uniform(high/5)
        self.v = np.random.uniform(high, size=dim)
        self.A = np.random.uniform(-30, 30, size=(2, 2))
        self.A = np.dot(self.A, self.A.T)

    def update(self, xs, us, m, ci):
        self.v = sum([us[:, ci][i]**m * xs[i] for i in range(len(xs))]) / sum([us[:, ci][i]**m for i in range(len(xs))])
        self.p = sum([us[:, ci][i]**m for i in range(len(xs))]) / \
                 sum([us[:, j][i]**m for i in range(len(xs)) for j in range(us.shape[1])])

        self.A = sum([us[:, ci][i]**m * (xs[i]-self.v).reshape((2, 1)).dot((xs[i]-self.v).reshape((2, 1)).T) for i in range(len(xs))]) / \
            sum([us[:, ci][i]**m for i in range(len(xs))])

    def distance(self, x):
        d = (x-self.v).reshape((2, 1)).T.dot(np.linalg.inv(self.A)).dot((x-self.v).reshape((2, 1)))[0][0]
        print (np.linalg.det(self.A))**.5 * 1.5**(.5 * d) / self.p
        return (np.linalg.det(self.A))**.5 * 1.5**(.5 * d) / self.p

    def __repr__(self):
        return "GathGeva cluster# v={0} A={1}, p={2}".format(self.v, self.A, self.p)

    def draw(self):
        if self.v.shape[0] > 2:
            raise FuzzyClassifierException("draw works for 2d data not more!")

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        vals, vecs = eigsorted(self.A)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        height, width = self.p * np.sqrt(vals)
        res = shapes.Ellipse(xy=self.v, width=width, height=height, angle=theta)
        return res

    def center(self):
        return self.v
