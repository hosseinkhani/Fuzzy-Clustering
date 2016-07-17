import numpy as np
import math


def generate_2d_circle_dataset(circles, q=1000, noise=10):
    res = np.empty((q, 2))

    centers = 1000/5 + np.random.uniform(size=(circles, 2)) * 1000*3/5
    radiuses = 50 + np.random.uniform(size=circles) * 1000/5
    # print centers, radiuses

    for i in range(q):
        ind = np.random.randint(circles)
        alpha = np.random.uniform(high=2*math.pi)
        # print alpha

        res[i] = centers[ind] + \
                 np.array([radiuses[ind] * math.cos(alpha), radiuses[ind] * math.sin(alpha)]) + \
                 np.array([np.random.randint(noise), np.random.randint(noise)])

    return res


def generate_2d_ellipse_dataset(ellipses, q=1000, noise=10):
    res = np.empty((q, 2))

    centers = 1000/5 + np.random.uniform(size=(ellipses, 2)) * 1000*3/5
    radiuses = 50 + np.random.uniform(size=(ellipses, 2)) * 1000/5

    for i in range(q):
        ind = np.random.randint(ellipses)
        alpha = np.random.uniform(high=2*math.pi)

        res[i] = centers[ind] + \
                 np.array([radiuses[ind][0] * math.cos(alpha), radiuses[ind][1] * math.sin(alpha)]) + \
                 np.array([np.random.randint(noise), np.random.randint(noise)])

    return res


def generate_2d_line_dataset(lines, q=1000, noise=10):
    res = np.zeros((q, 2))

    centers = 1000/5 + np.random.uniform(size=(lines, 2)) * 1000*3/5
    e = np.random.uniform(size=(lines, 2))

    for i in range(q):
        ind = np.random.randint(lines)
        point = centers[ind] + \
                e[ind].T.dot(10+np.random.randint(-500, 500)) + \
                np.array([np.random.randint(noise), np.random.randint(noise)])
        if 0 < point[0] < 1000 and 0 < point[1] < 1000:
            res[i] = point
        else:
            i -= 1

    return res


def load_generated_data():
    res = []
    n = 0
    with file('data.txt', 'r') as infile:
        for line in infile.readlines():
            n += 1
            l = [float(x) for x in line.strip().split()]
            res.append(l)

    return np.array(res).reshape((n, 2))
