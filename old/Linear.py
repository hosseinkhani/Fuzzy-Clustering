import numpy as np
from BaseFuzzyCluster import BaseFuzzyCluster
from matplotlib.lines import Line2D


class LinearCluster(BaseFuzzyCluster):
    def __init__(self, low, high, dim):
        self.e = np.random.uniform(size=dim)
        self.e /= np.linalg.norm(self.e)
        self.v = np.random.uniform(low, high, size=dim)

    def update(self, xs, uis, m):
        self.v = sum([uis[i]**m * xs[i] for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])

        C = sum([uis[i]**m * (xs[i]-self.v).reshape((2, 1)).dot((xs[i]-self.v).reshape((2, 1)).T)
                 for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])
        w, v = np.linalg.eig(C)
        if w[0] < w[1]:
            self.e = v[:, 1]
        else:
            self.e = v[:, 0]

    def distance(self, x):
        return (np.linalg.norm(x-self.v)**2 - (x-self.v).T.dot(self.e)**2)

    def __repr__(self):
        return "Linear cluster# v={0} e={1}".format(self.v, self.e)

    def draw(self):
        k1 = self.v - 1000 * self.e
        k2 = self.v + 1000 * self.e
        return Line2D([k1[0], k2[0]], [k1[1], k2[1]], lw=3.)

    def is_same(self, cluster):
        if type(cluster) != LinearCluster:
            return False
        # print '&'*3, np.linalg.norm(cluster.v-self.v) + np.linalg.norm(cluster.e-self.e)**2 < 20, np.linalg.norm(cluster.v-self.v) + np.linalg.norm(cluster.e-self.e)**2
        return np.linalg.norm(cluster.v-self.v) + np.linalg.norm(cluster.e-self.e)**2 < 20

    @staticmethod
    def valid_distance(noise):
        return 1.5 * noise**2