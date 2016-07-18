import numpy as np
import matplotlib.pyplot as plt


def drawEllipse(x, y, r1, r2, noise=0.1):


    # X = np.arange(1,3,0.02)

    X = np.zeros(100)
    X[0] = 1
    X[-1] = 3
    d = 0.00001
    for i in range(1, 50):
        X[i] = 1 + d
        X[-1-i] = 3 - d
        # d = 1.08 * d + 0.002
        d = 1.1 * d + 0.001
    # print d
    # print X[45:55]
    X += y - 2

    Y1 = np.zeros(len(X))
    Y2 = np.zeros(len(X))
    for i in range(len(X)):
        Y1[i] = r1 * np.sqrt(1-(X[i] - y)**2) + x
        Y2[i] = -r1 * np.sqrt(1-(X[i] - y)**2) + x

    X = X + noise * np.random.randn(len(X))
    tetha = np.pi * .5

    X1 = np.zeros(len(X))
    X2 = np.zeros(len(X))
    for i in range(len(X)):
        X1[i] = np.cos(tetha)*X[i] + np.sin(tetha)*Y1[i]
        Y1[i] = r2 * np.sin(tetha)*X[i] + np.cos(tetha)*Y1[i]
        X2[i] = np.cos(tetha)*X[i] + np.sin(tetha)*Y2[i]
        Y2[i] = r2 * np.sin(tetha)*X[i] + np.cos(tetha)*Y2[i]

    Data = np.array([np.hstack((X1, X2)), np.hstack((Y1, Y2))]).T
    print Data.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(Data[:, 0], Data[:, 1])
    plt.show()

if __name__ == '__main__':
    drawEllipse(1, -1, 20, 10, noise=.1)


