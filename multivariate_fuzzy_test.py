import matplotlib.pyplot as plt

from lib.multivariate_fuzzy_classifier.prototypes import MCMean
from lib.multivariate_fuzzy_classifier import MultivariateFuzzyClustering
from lib.fuzzy_classifier.protorypes import CMean
from lib.fuzzy_classifier import FuzzyClustring

from data import datagen_2d


def scatter_2d_data(data):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    plt.scatter(data[:, 0], data[:, 1], color=colors[1], lw=0)
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.show()


def MFCM_test(clusters, noise=10, data=None):
    if data is None:
        q = 100 * clusters
        xs = datagen_2d.generate_2d_cmeans_dataset(clusters, noise=noise, q=q)
        scatter_2d_data(xs)
    else:
        xs = data

    clusters = [MCMean.MCMeanCluster(1000, 2) for k in range(clusters)]
    fc = MultivariateFuzzyClustering.MultivariateFuzzyClassifier(xs, clusters, m=7)
    fc.fit(delta=.0001, increase_iteration=30, increase_factor=1.1, plot_level=2, verbose_level=0, verbose_iteration=100)
    print fc.C
    fc.scatter_clusters_data()


def cmean_test(clusters, noise=10, data=None):
    if data is None:
        q = 100 * clusters
        xs = datagen_2d.generate_2d_cmeans_dataset(clusters, noise=noise, q=q)
        scatter_2d_data(xs)
    else:
        xs = data

    clusters = [CMean.CMeanCluster(1000, 2) for k in range(clusters)]
    fc = FuzzyClustring.FuzzyClassifier(xs, clusters, m=2)
    fc.fit(delta=.001, increase_iteration=20, increase_factor=1.2, plot_level=2, verbose_level=0, verbose_iteration=100)
    print fc.C
    fc.scatter_clusters_data()


if __name__ == '__main__':
    clusters_num = 3
    noise = 20

    # Data
    q = 300 * clusters_num
    xs = datagen_2d.generate_2d_porellipse_dataset(clusters_num, noise=noise, q=q)
    scatter_2d_data(xs)

    MFCM_test(clusters=3, noise=10, data=xs)
    cmean_test(clusters=3, noise=10, data=xs)
