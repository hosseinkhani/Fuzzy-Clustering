import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import data_loader as datagen
from lib.protorypes.Circular import CircularCluster
from lib.protorypes.Elliptical import EllipticalCluster, EllipticalCluster2

from lib.fuzzy_classifier.FuzzyClustring import fuzzy_clustring_algorithm
from lib.fuzzy_classifier.protorypes.Linear import LinearCluster


def scatter_data(clusters_data):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    for i, cluster in enumerate(clusters_data):
        nl = np.array(cluster).reshape((len(cluster), 2))
        plt.scatter(nl[:, 0], nl[:, 1], color=colors[i], lw=0)
    plt.xlim(0, 1000)
    plt.ylim(0, 1000)
    plt.show()


if __name__ == '__main__':
    class_num = 4
    q = 300

    NOISE = 10
    U_TRESH = .9
    T_TRESH = 5

    final_clusters = []
    final_clusters_distance = []

    cluster_types = [LinearCluster, CircularCluster, EllipticalCluster2]
    trial = 0

    # Data
    xs = datagen.generate_2d_line_dataset(2, noise=NOISE, q=q/2)
    xs = np.vstack((xs, datagen.generate_2d_ellipse_dataset(1, noise=NOISE, q=q/4)))
    xs = np.vstack((xs, datagen.generate_2d_circle_dataset(1, noise=NOISE, q=q/4)))

    scatter_data([xs])

    while len(final_clusters) < class_num:
        print 'clusters:', len(final_clusters), '@', trial

        cs = [deepcopy(final_clusters[i]) for i in range(len(final_clusters))]
        if trial/T_TRESH >= len(cluster_types):
            if 'old_final_clusters' in locals():
                'no answer :|'
                break
            print 'coudnt solve this data, delete the clustered ones'
            trial = 0
            old_final_clusters = deepcopy(final_clusters)
            final_clusters = []
            final_clusters_distance = []

            old_class_num = class_num
            class_num -= len(old_final_clusters)

            old_xs = deepcopy(xs)

            dead_indices = []
            for i, x in enumerate(xs):
                for cluster in old_final_clusters:
                    if cluster.distance(x) < cluster.valid_distance(NOISE):
                        dead_indices.append(i)
            xs = np.delete(xs, dead_indices, axis=0)

            scatter_data([xs])

            continue
        elif cluster_types[trial/T_TRESH] == LinearCluster:
            cs += [LinearCluster(1000/5, 1000*3/5, 2) for i in range(class_num-len(final_clusters))]
        elif cluster_types[trial/T_TRESH] == CircularCluster:
            cs += [CircularCluster(1000/5, 1000*3/5, 2) for i in range(class_num-len(final_clusters))]
        elif cluster_types[trial/T_TRESH] == EllipticalCluster:
            cs += [EllipticalCluster(5, 1000/5, 1000*3/5, 2) for i in range(class_num-len(final_clusters))]
        elif cluster_types[trial/T_TRESH] == EllipticalCluster2:
            cs += [EllipticalCluster2(1000/5, 1000*3/5, 2) for i in range(class_num-len(final_clusters))]

        us = fuzzy_clustring_algorithm(xs, cs, m=2, delta=1e-2, plot=False)

        clusters_data = [[] for i in range(class_num)]
        for i, x in enumerate(xs):
            if max(us[i]) > U_TRESH:
                uu = us[i].tolist()
                clusters_data[uu.index(max(us[i]))].append(x)
        # scatter_data(clusters_data)

        distances = []
        for i, data in enumerate(clusters_data):
            if len(data) > (3*len(xs))/(4*class_num) and (i < len(final_clusters) or sum([int(seen.is_same(cs[i])) for seen in final_clusters])==0):
                distances.append(sum([cs[i].distance(data[j]) for j in range(len(data))]) / len(data))
                # print i, distances[i], '/', cluster_types[trial/T_TRESH].valid_distance(NOISE)
            else:
                distances.append(np.inf)

        best_distance = np.inf
        best_index = None
        print 'distances', distances
        for i, d in enumerate(distances):
            if i < len(final_clusters):
                if final_clusters_distance[i] > d:
                    final_clusters_distance[i] = d
                    final_clusters[i] = cs[i]
            else:
                if d < cs[i].valid_distance(NOISE) and d < best_distance:
                    best_distance = d
                    best_index = i

        print 'final_clusters', final_clusters
        print 'final_dists', final_clusters_distance

        if best_index == None:
            trial += 1
            continue
        else:
            final_clusters_distance.append(distances[best_index])
            final_clusters.append(cs[best_index])
            # class_num -= 1
            trial = 0

        print '+++', distances[best_index], cs[best_index]

        final_clusters_data = [[] for i in range(len(final_clusters))]
        for i, x in enumerate(xs):
            best_distance = np.inf
            best_index = None
            for j, cluster in enumerate(final_clusters):
                if cluster.distance(x) < cluster.valid_distance(NOISE) and cluster.distance(x) < best_distance: #and not vals[i]:
                    best_distance = cluster.distance(x)
                    best_index = j
            if best_index is not None:
                final_clusters_data[best_index].append(x)
                # deleted_indices.append(i)
        # xs = np.delete(xs, deleted_indices, axis=0)

        # ourliers
        # deleted_indices = []
        # nearests = [np.inf] * len(xs)
        # for i in range(len(xs)):
        #     for j in range(len(xs)):
        #         if np.linalg.norm(xs[i]-xs[j]) < nearests[i]:
        #             nearests[i] = np.linalg.norm(xs[i]-xs[j])
        # mean = np.mean(nearests)
        # std = np.std(nearests)
        # for i in range(len(xs)):
        #     if nearests[i] - mean > 2 * std:
        #         deleted_indices.append(i)
        # xs = np.delete(xs, deleted_indices, axis=0)
        #
        # deleted_indices = []
        # mean = np.mean(xs)
        # std = np.mean(xs)
        # for i in range(len(xs)):
        #     if nearests[i] - mean > 2 * std:
        #         deleted_indices.append(i)
        # xs = np.delete(xs, deleted_indices, axis=0)

        scatter_data(final_clusters_data)
        # scatter_data([xs])

    if 'old_final_clusters' in locals():
        final_clusters += old_final_clusters
        class_num = old_class_num
        xs = old_xs

    final_clusters_data = [[] for i in range(len(final_clusters))]
    for i, x in enumerate(xs):
        best_distance = np.inf
        best_index = None
        for j, cluster in enumerate(final_clusters):
            if cluster.distance(x) < cluster.valid_distance(NOISE) and cluster.distance(x) < best_distance: #and not vals[i]:
                best_distance = cluster.distance(x)
                best_index = j
        if best_index is not None:
            final_clusters_data[best_index].append(x)
    scatter_data(final_clusters_data)
