import numpy as np
import matplotlib.pyplot as plt


from data import datagen_2d
from lib import FuzzyClustring
from lib.protorypes import Linear, Circular, Elliptical



def scatter_2d_data(data):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    plt.scatter(data[:, 0], data[:, 1], color=colors[4], lw=0)
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.show()

# xs = datagen_2d.generate_2d_line_dataset(class_num, noise=NOISE, q=q)
# xs = np.vstack((xs, datagen_2d.generate_2d_ellipse_dataset(1, noise=NOISE, q=q/4)))
# xs = np.vstack((xs, datagen_2d.generate_2d_circle_dataset(1, noise=NOISE, q=q/4)))

# xs = datagen_2d.generate_2d_circle_dataset(class_num, noise=NOISE, q=q)
# datagen_2d.shuffle(xs)

# scatter_2d_data(xs)

# clusters = [Linear.LinearCluster(0, 1000, 2) for k in range(class_num)]
# clusters = [Elliptical.EllipticalCluster(5, 1000/5, 4*1000/5, 2) for k in range(class_num)]
# clusters = [Elliptical.EllipticalCluster2(1000/5, 4*1000/5, 2) for k in range(class_num)]
# clusters = [Circular.CircularCluster(1000/5, 4*1000/5, 2) for k in range(class_num)]


def linear_test(clusters, noise=10):
    q = 50 * clusters
    xs = datagen_2d.generate_2d_line_dataset(clusters, noise=noise, q=q)
    scatter_2d_data(xs)

    clusters = [Linear.LinearCluster(1000, 2) for k in range(clusters)]
    fc = FuzzyClustring.FuzzyClassifier(xs, clusters, m=2)
    fc.fit(delta=.001, increase_iteration=20, increase_factor=1.2, plot_level=2, verbose_level=0, verbose_iteration=100)
    print fc.C
    fc.scatter_clusters_data()


def circular_test(clusters, noise=10):
    q = 50 * clusters
    xs = datagen_2d.generate_2d_circle_dataset(clusters, noise=noise, q=q)
    scatter_2d_data(xs)

    clusters = [Circular.CircularCluster(1000, 2) for k in range(clusters)]
    fc = FuzzyClustring.FuzzyClassifier(xs, clusters, m=2)
    fc.fit(delta=.001, increase_iteration=20, increase_factor=1.2, plot_level=2, verbose_level=0, verbose_iteration=100)
    print fc.C
    fc.scatter_clusters_data()


def elliptical_test(clusters, ellipce_type=2, noise=10):
    q = 50 * clusters
    xs = datagen_2d.generate_2d_ellipse_dataset(clusters, noise=noise, q=q)
    scatter_2d_data(xs)

    if ellipce_type == 1:
        clusters = [Elliptical.EllipticalCluster(5, 1000, 2) for k in range(clusters)]
    elif ellipce_type == 2:
        clusters = [Elliptical.EllipticalCluster2(1000, 2) for k in range(clusters)]

    fc = FuzzyClustring.FuzzyClassifier(xs, clusters, m=2)
    fc.fit(delta=.001, increase_iteration=30, increase_factor=1.2, plot_level=2, verbose_level=0, verbose_iteration=100)
    print fc.C
    fc.scatter_clusters_data()


if __name__ == '__main__':
    print "Linear Test..."
    linear_test(clusters=3, noise=20)
    print "Circular Test..."
    circular_test(clusters=3, noise=20)
    # print "Elliptical Test..."
    # elliptical_test(clusters=3, ellipce_type=2, noise=20)
