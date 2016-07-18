import matplotlib.pyplot as plt

from lib.protorypes import Circular, Elliptical, CMean, GustafsonKessel, GathGeva

from data import datagen_2d
from lib.FuzzyClassifier import FuzzyClustring
from lib.FuzzyClassifier.protorypes import Linear


def scatter_2d_data(data):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    plt.scatter(data[:, 0], data[:, 1], color=colors[4], lw=0)
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.show()


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


def cmean_test(clusters, noise=10):
    q = 100 * clusters
    xs = datagen_2d.generate_2d_cmeans_dataset(clusters, noise=noise, q=q)
    scatter_2d_data(xs)

    clusters = [CMean.CMeanCluster(1000, 2) for k in range(clusters)]
    fc = FuzzyClustring.FuzzyClassifier(xs, clusters, m=2)
    fc.fit(delta=.001, increase_iteration=20, increase_factor=1.2, plot_level=2, verbose_level=0, verbose_iteration=100)
    print fc.C
    fc.scatter_clusters_data()


def gustafsonkessel_test(clusters, noise=10):
    q = 100 * clusters
    xs = datagen_2d.generate_2d_gustafsonkessel_dataset(clusters, noise=noise, q=q)
    scatter_2d_data(xs)

    clusters = [GustafsonKessel.GKCluster(1000, 2) for k in range(clusters)]
    fc = FuzzyClustring.FuzzyClassifier(xs, clusters, m=2)
    fc.fit(delta=.001, increase_iteration=20, increase_factor=1.2, plot_level=2, verbose_level=0, verbose_iteration=100)
    print fc.C
    fc.scatter_clusters_data()


def gathgeva_test(clusters, noise=10):
    q = 100 * clusters
    xs = datagen_2d.generate_2d_gathgeva_dataset(clusters, noise=noise, q=q)
    scatter_2d_data(xs)

    clusters = [GathGeva.GGCluster(1000, 2) for k in range(clusters)]
    fc = FuzzyClustring.FuzzyClassifier(xs, clusters, m=2)
    fc.fit(delta=.001, increase_iteration=20, increase_factor=1.2, plot_level=2, verbose_level=0, verbose_iteration=100)
    print fc.C
    fc.scatter_clusters_data()


def elliptical_test(clusters, ellipce_type=2, noise=10):
    q = 50 * clusters
    xs = datagen_2d.generate_2d_ellipse_dataset(clusters, noise=noise, q=q)
    scatter_2d_data(xs)

    if ellipce_type == 1:
        clusters = [Elliptical.EllipticalCluster(1000, 2) for k in range(clusters)]
    elif ellipce_type == 2:
        clusters = [Elliptical.EllipticalCluster2(1000, 2) for k in range(clusters)]

    fc = FuzzyClustring.FuzzyClassifier(xs, clusters, m=2)
    fc.fit(delta=.001, increase_iteration=30, increase_factor=1.2, plot_level=2, verbose_level=0, verbose_iteration=100)
    print fc.C
    fc.scatter_clusters_data()


if __name__ == '__main__':
    # print "CMEAN Test..."
    # cmean_test(clusters=3, noise=20)

    print "GustafsonKessel Test ..."
    gustafsonkessel_test(clusters=3, noise=20)

    print "GathGeva Test ..."
    gathgeva_test(clusters=3, noise=20)

    # print "Linear Test..."
    # linear_test(clusters=3, noise=20)

    # print "Circular Test..."
    # circular_test(clusters=3, noise=20)

    # print "Elliptical Test..."
    # elliptical_test(clusters=3, ellipce_type=2, noise=20)
