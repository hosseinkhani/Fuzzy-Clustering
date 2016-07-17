import numpy as np
from BaseFuzzyCluster import BaseFuzzyCluster
from matplotlib.patches import Ellipse


class EllipticalCluster(BaseFuzzyCluster):
    def __init__(self, det, low, high, dim):
        self.r = np.random.uniform(low)
        self.v = np.random.uniform(low, high, size=dim)
        self.det = det
        self.A = np.array([[2, 0], [0, det/20.0]])

    def update(self, xs, uis, m):
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
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]

        vals, vecs = eigsorted(self.A)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        height, width = self.r * np.sqrt(vals)
        ellip = Ellipse(xy=self.v, width=width, height=height, angle=theta)
        return ellip

    def is_same(self, cluster):
        if type(cluster) != EllipticalCluster:
            return False
        # print np.linalg.norm(cluster.v-self.v) + (self.r-cluster.r)**2 + np.linalg.norm(cluster.A-self.A)**2, '&'*10
        return np.linalg.norm(cluster.v-self.v) + (self.r-cluster.r)**2 + np.linalg.norm(cluster.A-self.A)**2 < 50

    @staticmethod
    def valid_distance(noise):
        return 10 * noise**2

class EllipticalCluster2(BaseFuzzyCluster):
    def __init__(self, low, high, dim):
        self.r = np.random.uniform(low)
        self.v1 = np.random.uniform(low, high, size=dim)
        self.v2 = np.random.uniform(low, high, size=dim)

    def update(self, xs, uis, m):
        self.v1 = sum([uis[i]**m * (xs[i] + (np.linalg.norm(xs[i]-self.v2)-self.r)*(xs[i]-self.v1)/(np.linalg.norm(xs[i]-self.v1)))
                       for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])
        self.v2 = sum([uis[i]**m * (xs[i] + (np.linalg.norm(xs[i]-self.v1)-self.r)*(xs[i]-self.v2)/(np.linalg.norm(xs[i]-self.v2)))
                       for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])
        self.r = sum([uis[i]**m * (np.linalg.norm(xs[i]-self.v1)+np.linalg.norm(xs[i]-self.v2))
                      for i in range(len(xs))]) / sum([uis[i]**m for i in range(len(xs))])

    def distance(self, x):
        return (np.linalg.norm(x-self.v1) + np.linalg.norm(x-self.v2) - self.r)**2

    def __repr__(self):
        return "Elliptical cluster# v1={0} v2={1} r={2}".format(self.v1, self.v2, self.r)

    def draw(self):
        c = np.dot(self.v1, self.v2)/np.linalg.norm(self.v1)/np.linalg.norm(self.v2)
        angle = np.arccos(np.clip(c, -1, 1))
        ellip = Ellipse(xy=(self.v1+self.v2)/2, width=1.5*self.r, height=self.r, angle=np.rad2deg(angle))
        return ellip

    def is_same(self, cluster):
        if type(cluster) != EllipticalCluster:
            return False
        # print np.linalg.norm(cluster.v-self.v) + (self.r-cluster.r)**2 + np.linalg.norm(cluster.A-self.A)**2, '&'*10
        return np.linalg.norm(cluster.v-self.v) + (self.r-cluster.r)**2 + np.linalg.norm(cluster.A-self.A)**2 < 50

    def __getattr__(self, item):
        if item == 'v':
            return (self.v1+self.v2)/2
        else:
            BaseFuzzyCluster.__getattr__(self, item)

    @staticmethod
    def valid_distance(noise):
        return 10 * noise**2
