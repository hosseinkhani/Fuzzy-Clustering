import numpy as np
from matplotlib import pyplot as plt


def show_plot(clusters, xs, us, m, selection=None):
    fig = plt.figure(0)
    ax = fig.add_subplot(111, aspect='equal')

    shapes = [clusters[i].draw() for i in range(len(clusters))]

    for i in range(len(clusters)):
        ax.scatter(clusters[i].v[0], clusters[i].v[1], color='black')
        ax.add_artist(shapes[i])
        shapes[i].set_alpha(.1)
        if i == selection:
            shapes[i].set_color('red')

    if selection is None:
        ax.scatter(xs[:, 0], xs[:, 1], lw=0)
    else:
        rgba_colors = np.zeros((len(xs), 4))
        rgba_colors[:, 0] = 1.0
        rgba_colors[:, 3] = us[:, selection] ** m
        ax.scatter(xs[:, 0], xs[:, 1], color=rgba_colors)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    plt.show()


def fuzzy_clustring_algorithm(xs, clusters, m=2, delta=0.1, plot=False):
    us = np.empty(tuple([len(xs), len(clusters)]))
    for i in range(len(clusters)):
        for j in range(len(xs)):
            us[j][i] = 1.0 / sum([(clusters[i].distance(xs[j])/clusters[k].distance(xs[j])) ** (1/(m-1))
                                  for k in range(len(clusters))])

    t = 0
    while True:
        # if t % 20 == 0:
        #     print clusters
        #     for i in range(len(clusters)):
        #         show_plot(clusters, xs, us, m, i)
        t += 1

        for i in range(len(clusters)):
            clusters[i].update(xs, us[:, i], m)

        max_dif = -1
        us_new = np.zeros(us.shape)
        for i in range(len(clusters)):
            for j in range(len(xs)):
                us_new[j][i] = 1.0 / sum([(clusters[i].distance(xs[j])/clusters[k].distance(xs[j])) ** (1/(m-1))
                                          for k in range(len(clusters))])
                if max_dif < abs(us_new[j][i] - us[j][i]):
                    max_dif = abs(us_new[j][i] - us[j][i])

        if max_dif < delta:
            print clusters
            if plot:
                show_plot(clusters, xs, us, m)
            return us_new

        if t % 50 == 0:
            delta *= 2

        us = us_new
