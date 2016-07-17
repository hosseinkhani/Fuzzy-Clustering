import matplotlib.patches as shapes
import numpy as np

from ..BaseFuzzyCluster import BaseFuzzyCluster


class CircularCluster(BaseFuzzyCluster):
    def __init__(self, low, high, dim):
        self.r = np.random.uniform(low)
        self.v = np.random.uniform(low, high, size=dim)

    def update(self, xs, uis, m):
        self.v = sum([uis[i]**m * xs[i] for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])
        self.r = sum([uis[i]**m * np.linalg.norm(xs[i]-self.v) for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])
        # self.v = sum([uis[i] * xs[i] for i in range(len(xs))]) / sum(uis)
        # self.r = sum([uis[i] * np.linalg.norm(xs[i]-self.v) for i in range(len(xs))]) / sum(uis)

    def distance(self, x):
        return (np.linalg.norm(x-self.v)-self.r)**2

    def __repr__(self):
        return "Circular cluster# v={0} r={1}".format(self.v, self.r)

    def draw(self):
        return shapes.Circle(xy=self.v, radius=self.r)

    def center(self):
        return self.v

    def is_same(self, cluster):
        if type(cluster) != CircularCluster:
            return False
        # print np.linalg.norm(cluster.v-self.v) + (cluster.r-self.r)**2, '&'*10
        return np.linalg.norm(cluster.v-self.v) + (cluster.r-self.r)**2 < 20

    @staticmethod
    def valid_distance(noise):
        return 5 * noise**2

# print np.array([1, 2])