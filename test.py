import old.data_loader
import numpy as np
import matplotlib.pyplot as plt


def scatter_2d_data(clusters_data):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    for i, cluster in enumerate(clusters_data):
        nl = np.array(cluster).reshape((len(cluster), 2))
        plt.scatter(nl[:, 0], nl[:, 1], color=colors[i], lw=0)
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.show()

NOISE = 30
class_num = 3
q = 100

xs = old.data_loader.generate_2d_line_dataset(class_num, noise=NOISE, q=q)
# xs = np.vstack((xs, old.data_loader.generate_2d_ellipse_dataset(1, noise=NOISE, q=q/4)))
# xs = np.vstack((xs, old.data_loader.generate_2d_circle_dataset(1, noise=NOISE, q=q/4)))

scatter_2d_data([xs])

import lib.protorypes.Linear as lp
clusters = [lp.LinearCluster(0, 1000, 2) for k in range(class_num)]

from lib import FuzzyClustring
ss = FuzzyClustring.FuzzyClassifier(xs, clusters)
ss.fit(delta=.001, plot_level=1)
ss.scatter_clusters_data()
