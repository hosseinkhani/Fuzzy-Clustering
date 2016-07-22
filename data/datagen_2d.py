import numpy as np
import math
from random import shuffle


def generate_2d_gustafsonkessel_dataset(clusters, q=1000, noise=10, radiuse=500):
    res = np.empty((q, 2))

    centers = 1000/5 + np.random.uniform(size=(clusters, 2)) * 1000*3/5
    covs = np.random.uniform(-3, 3, size=(clusters, 2, 2))
    for k in range(clusters):
        covs[k] = np.dot(covs[k], covs[k].T)
        covs[k] += .9 * np.eye(2)
    covs *= radiuse

    for i in range(q):
        ind = np.random.randint(clusters)
        res[i] = np.random.multivariate_normal(centers[ind], covs[ind], 1)

    return res


def generate_2d_gathgeva_dataset(clusters, q=1000, noise=10):
    res = np.empty((q, 2))

    centers = 1000/5 + np.random.uniform(size=(clusters, 2)) * 1000*3/5
    radiuses = 50 + np.random.uniform(size=clusters) * 1000/5
    covs = np.random.uniform(-3, 3, size=(clusters, 2, 2))
    for k in range(clusters):
        covs[k] = np.dot(covs[k], covs[k].T)
        covs[k] += .9 * np.eye(2)
        covs[k] *= radiuses[k]

    for i in range(q):
        ind = np.random.randint(clusters)
        res[i] = np.random.multivariate_normal(centers[ind], covs[ind], 1)

    return res


def generate_2d_cmeans_dataset(clusters, q=1000, noise=10):
    res = np.empty((q, 2))

    centers = 1000/5 + np.random.uniform(size=(clusters, 2)) * 1000*3/5
    radiuses = 50 + np.random.uniform(size=clusters) * 1000/5
    # print centers, radiuses

    for i in range(q):
        ind = np.random.randint(clusters)
        alpha = np.random.uniform(high=2*math.pi)
        r = np.random.uniform(high=radiuses[ind])

        res[i] = centers[ind] + \
                 np.array([r * math.cos(alpha), r * math.sin(alpha)]) + \
                 np.array([np.random.randint(noise), np.random.randint(noise)])

    return res


# bad bad bad
def generate_2d_porellipse_dataset(ellipses, q=1000, noise=10):
    res = np.empty((q, 2))

    centers = 1000/5 + np.random.uniform(size=(ellipses, 2)) * 1000*3/5
    radiuses = 50 + np.random.uniform(size=(ellipses, 2)) * 1000/5

    for i in range(q):
        ind = np.random.randint(ellipses)
        alpha = np.random.uniform(high=2*math.pi)
        h = np.random.uniform(high=radiuses[ind][0])
        w = np.random.uniform(high=radiuses[ind][1])

        res[i] = centers[ind] + \
                 np.array([h * math.cos(alpha), w * math.sin(alpha)]) + \
                 np.array([np.random.randint(noise), np.random.randint(noise)])

    return res


def generate_2d_line_dataset(clusters, q=1000, noise=10):
    res = np.zeros((q, 2))

    centers = 1000/5 + np.random.uniform(size=(clusters, 2)) * 1000*3/5
    e = np.random.uniform(size=(clusters, 2))

    i = 0
    while i < q:
        ind = np.random.randint(clusters)
        point = centers[ind] + \
                e[ind].T.dot(10+np.random.randint(-500, 500)) + \
                np.array([np.random.randint(noise), np.random.randint(noise)])
        if 0 < point[0] < 1000 and 0 < point[1] < 1000:
            res[i] = point
            i += 1

    return res


def generate_2d_circle_dataset(clusters, q=1000, noise=10):
    res = np.empty((q, 2))

    centers = 1000/5 + np.random.uniform(size=(clusters, 2)) * 1000*3/5
    radiuses = 50 + np.random.uniform(size=clusters) * 1000/5
    # print centers, radiuses

    for i in range(q):
        ind = np.random.randint(clusters)
        alpha = np.random.uniform(high=2*math.pi)
        # print alpha

        res[i] = centers[ind] + \
                 np.array([radiuses[ind] * math.cos(alpha), radiuses[ind] * math.sin(alpha)]) + \
                 np.array([np.random.randint(noise), np.random.randint(noise)])

    return res


def generate_2d_ellipse_dataset(clusters, q=1000, noise=10):
    res = np.empty((q, 2))

    centers = 1000/5 + np.random.uniform(size=(clusters, 2)) * 1000*3/5
    radiuses = 50 + np.random.uniform(size=(clusters, 2)) * 1000/5

    for i in range(q):
        ind = np.random.randint(clusters)
        alpha = np.random.uniform(high=2*math.pi)

        res[i] = centers[ind] + \
                 np.array([radiuses[ind][0] * math.cos(alpha), radiuses[ind][1] * math.sin(alpha)]) + \
                 np.array([np.random.randint(noise), np.random.randint(noise)])

    return res


