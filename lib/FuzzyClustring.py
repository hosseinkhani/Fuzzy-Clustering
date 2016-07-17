import numpy as np
import math
from matplotlib import pyplot as plt

from . import FuzzyClassifierException


class FuzzyClassifier(object):
    def __init__(self, data, clusters, m=2):
        self.X = data
        self.C = clusters
        self.U = np.empty(tuple([len(self.X), len(self.C)]))
        self.m = m

    def fit(self, delta=0.1, **kwargs):
        """
        fit clusters to data
        :param delta: stop critrion value
        :param m:
        :param plot_level: different plotting levels, values 0:nothing,1:show all together,2:show detailed plots
        """
        plot_level = kwargs.get('plot_level', 0)
        delta_increase_iteration = kwargs.get('increase_iteration', 100)
        first_delta = delta

        self.update_memberships()

        iteration = 0
        while 1:
            iteration += 1

            for i, c in enumerate(self.C):
                c.update(self.X, self.U[:, i], self.m)

            dif = self.update_memberships()

            print iteration, dif
            if dif < delta:
                if plot_level == 1:
                    self.show_plot()
                elif plot_level == 2:
                    self.show_detailed_plot()
                return

            if iteration % delta_increase_iteration == 0:
                delta += first_delta

    def update_memberships(self):
        max_dif = -1

        for j, x in enumerate(self.X):
            distances = [c.distance(x) for c in self.C]
            for i, c in enumerate(self.C):
                old = self.U[j][i]
                self.U[j][i] = 1.0 / sum([(distances[i]/distances[k]) ** (1.0/(self.m-1)) for k in range(len(self.C))])

                if max_dif < abs(old - self.U[j][i]):
                    max_dif = abs(old - self.U[j][i])

        return max_dif

    def show_detailed_plot(self):
        if len(self.C) == 1:
            row_num, col_num = 1, 1
        elif len(self.C) <= 2:
            row_num, col_num = 1, 2
        elif len(self.C) <= 4:
            row_num, col_num = 2, 2
        elif len(self.C) <= 6:
            row_num, col_num = 2, 3
        elif len(self.C) <= 9:
            row_num, col_num = 3, 3
        elif len(self.C) <= 12:
            row_num, col_num = 3, 4

        fig = plt.figure(1)
        for selected in range(len(self.C)):
            shapes = [c.draw() for c in self.C]
            ax = fig.add_subplot(row_num*100 + col_num*10 + selected+1, aspect='equal')

            for i, c in enumerate(self.C):
                try:
                    ax.scatter(c.center()[0], c.center()[1], color='black')
                    ax.add_artist(shapes[i])
                    shapes[i].set_alpha(.1)
                    if i == selected:
                        shapes[i].set_color('red')
                except FuzzyClassifierException as e:
                    print e, "No shape or center implemented!"

            rgba_colors = np.zeros((len(self.X), 4))
            rgba_colors[:, 0] = 1.0
            rgba_colors[:, 3] = self.U[:, selected]
            ax.scatter(self.X[:, 0], self.X[:, 1], color=rgba_colors)

            ax.set_xlim(0, 1000)
            ax.set_ylim(0, 1000)
        plt.show()

    def show_plot(self):
        shapes = [c.draw() for c in self.C]

        fig = plt.figure(1)
        ax = fig.add_subplot(111, aspect='equal')

        for i, c in enumerate(self.C):
            try:
                ax.scatter(c.center()[0], c.center()[1], color='black')
                ax.add_artist(shapes[i])
                shapes[i].set_alpha(.1)
            except FuzzyClassifierException as e:
                print e, "No shape or center implemented!"

        ax.scatter(self.X[:, 0], self.X[:, 1], lw=0)

        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        plt.show()

    def scatter_clusters_data(self):
        if self.X.shape[1] > 2:
            print "just 2d data can be scattered!"
            return

        clustered_data = [[] for i in range(len(self.C))]
        for j, x in enumerate(self.X):
            ci = np.argmax(self.U[j, :])
            clustered_data[ci].append(x)

        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
        for i, xs in enumerate(clustered_data):
            xs = np.array(xs)
            plt.scatter(xs[:, 0], xs[:, 1], color=colors[i], lw=0)
        plt.xlim(0, 1000)
        plt.ylim(0, 1000)
        plt.show()
