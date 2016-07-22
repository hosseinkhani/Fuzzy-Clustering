import matplotlib.patches as shapes
import numpy as np

from ... import FuzzyClassifierException
from ..BaseFuzzyCluster import BaseFuzzyCluster


class EllipticalCluster(BaseFuzzyCluster):
    def __init__(self, high, dim, det=1):
        self.r = np.random.uniform(high/5)
        self.v = np.random.uniform(high, size=dim)
        self.det = det
        self.A = np.array([[2, 0], [0, det/20.0]])

    def _update(self, xs, uis, m):
        self.v = sum([uis[i]**m * xs[i] for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])
        self.r = sum([uis[i]**m * np.linalg.norm(xs[i]-self.v) for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])

        S = sum([(uis[i]**m * self.distance(xs[i]) / ((xs[i]-self.v).T.dot(self.A)).dot((xs[i]-self.v))**.5) *
                 ((xs[i]-self.v).reshape((2, 1)).dot((xs[i]-self.v).reshape((2, 1)).T)) for i in range(len(xs))])
        self.A = (self.det * np.linalg.det(S))**(1.0/len(self.v)) * np.linalg.inv(S)

    def distance(self, x):
        return (((x-self.v).T.dot(self.A)).dot(x-self.v)**.5-self.r)**2 / \
               ((x-self.v).T.dot(x-self.v)/(x-self.v).T.dot(self.A).dot(x-self.v))

    def __repr__(self):
        return "Elliptical cluster# v={0} r={1}".format(self.v, self.r)

    def draw(self):
        if self.v.shape[0] > 2:
            raise FuzzyClassifierException("draw works for 2d data not more!")

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]

        vals, vecs = eigsorted(self.A)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        height, width = self.r * np.sqrt(vals)
        res = shapes.Ellipse(xy=self.v, width=width, height=height, angle=theta)
        res.set_fill(False)
        return res

    def center(self):
        return self.v

    def is_same(self, cluster):
        if type(cluster) != EllipticalCluster:
            return False
        # print np.linalg.norm(cluster.v-self.v) + (self.r-cluster.r)**2 + np.linalg.norm(cluster.A-self.A)**2, '&'*10
        return np.linalg.norm(cluster.v-self.v) + (self.r-cluster.r)**2 + np.linalg.norm(cluster.A-self.A)**2 < 50

    @staticmethod
    def valid_distance(noise):
        return 10 * noise**2


class EllipticalCluster2(BaseFuzzyCluster):
    def __init__(self, high, dim):
        self.r = np.random.uniform(high/5)
        self.v1 = np.random.uniform(high, size=dim)
        self.v2 = np.random.uniform(high, size=dim)

    def _update(self, xs, uis, m):
        self.v1 = sum([uis[i]**m * (xs[i] + (np.linalg.norm(xs[i]-self.v2)-self.r)*(xs[i]-self.v1)/(np.linalg.norm(xs[i]-self.v1)))
                       for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])
        self.v2 = sum([uis[i]**m * (xs[i] + (np.linalg.norm(xs[i]-self.v1)-self.r)*(xs[i]-self.v2)/(np.linalg.norm(xs[i]-self.v2)))
                       for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])
        self.r = sum([uis[i]**m * (np.linalg.norm(xs[i]-self.v1)+np.linalg.norm(xs[i]-self.v2))
                      for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])

    def distance(self, x):
        return (np.linalg.norm(x-self.v1) + np.linalg.norm(x-self.v2) - self.r)**2

    def __repr__(self):
        return "Elliptical cluster# v={0} r={1}".format(self.center(), self.r)

    def draw(self):
        if self.v1.shape[0] > 2:
            raise FuzzyClassifierException("draw works for 2d data not more!")

        c = np.linalg.norm(self.v2-self.v1)/2.0
        a = self.r + c
        b = (a**2 - c**2)**.5

        c2 = np.dot(self.v1, self.v2)/np.linalg.norm(self.v1)/np.linalg.norm(self.v2)
        angle = np.arccos(np.clip(c2, -1, 1))
        res = shapes.Ellipse(xy=(self.v1+self.v2)/2, width=a, height=b, angle=np.rad2deg(angle))
        res.set_fill(False)
        return res

    def center(self):
        return (self.v1+self.v2)/2

    def is_same(self, cluster):
        if type(cluster) != EllipticalCluster:
            return False
        # print np.linalg.norm(cluster.v-self.v) + (self.r-cluster.r)**2 + np.linalg.norm(cluster.A-self.A)**2, '&'*10
        return np.linalg.norm(cluster.v-self.v) + (self.r-cluster.r)**2 + np.linalg.norm(cluster.A-self.A)**2 < 50

    @staticmethod
    def valid_distance(noise):
        return 10 * noise**2
