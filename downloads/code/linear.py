import matplotlib.pyplot as plt
import numpy as np
from datareader import DataReader


class Solution:
    def __init__(self, path, nvar, iteration=1500, lr=0.01):
        """

        :param path:
        :param nvar: 变量的数目
        :param iteration:
        :param lr:
        """
        data = DataReader.read(path, nvar + 1)  # nvar + y
        self.y = np.array([data[-1]]).transpose()
        self.x = np.array([np.ones((len(self.y),))] \
                          + [np.array(data[i]) for i in range(nvar)]).transpose()
        self.theta = np.zeros((nvar + 1, 1))
        self.iteration = iteration
        self.lr = lr
        self.nvar = nvar

        self.mu = self.x.mean(0)
        self.s = self.x.max(0) - self.x.min(0)
        self.mu[0] = 0
        self.s[0] = 1  # for x_0: (1 - 0) / 1 = 1
        self.feature_normed = False

    def feature_norm(self):
        self.feature_normed = True
        for i in range(len(self.x)):
            self.x[i] = (self.x[i] - self.mu) / self.s

    def linear_regression(self):
        for i in range(self.iteration):
            yield self.cal_j_theta()
            self.__step()
        yield self.cal_j_theta()

    def cal_j_theta(self):
        deltay = self.predict(self.x) - self.y
        return (deltay * deltay).sum() / 2 / len(self.y)

    def __step(self):
        self.theta = self.theta - self.x.transpose().dot(self.predict(self.x) - self.y) * self.lr / len(self.y)

    def norm_equation(self):
        xT = self.x.transpose()
        self.theta = np.linalg.pinv(xT.dot(self.x)).dot(xT).dot(self.y)

    def predict(self, x):
        return x.dot(self.theta)


if __name__ == "__main__":
    s = Solution("ex1data2.txt", 2, lr=0.01)
    s.feature_norm()
    j_thetas = np.array([i for i in s.linear_regression()])
    print("J(theta) of linear regression = %f" % (j_thetas[-1]))
    x = np.linspace(0, len(j_thetas), len(j_thetas))
    plt.scatter(x, j_thetas)
    plt.show()
    # s.norm_equation()
    # print("J(theta) of normal equation = %f"%(s.cal_j_theta()))