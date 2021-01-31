import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


class Linear:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        self.theta = np.zeros((nout, nin + 1))
        self.x = np.array([])
        self.nx = 0
        self.y = np.array([])
        self.mu = []
        self.s = []
        self.reg_param = 0
        self.feature_normed = False

    def set_data(self, training_data, do_feature_norm=True, reg_param=0):
        self.x = training_data[0]
        self.nx = len(self.x[0])
        self.y = training_data[1]
        if do_feature_norm:
            self.__feature_norm()
        self.feature_normed = do_feature_norm
        self.reg_param = reg_param

    def learn_debug(self, iteration=1500, lr=0.01):
        for i in range(iteration):
            yield self.j_theta()
            self.__step(lr)
        yield self.j_theta()

    def j_theta(self):
        h_theta = self.__forward(self.x)
        j_theta = -(self.y * np.log(h_theta) + (1 - self.y) * np.log(1 - h_theta)).sum() / self.nx
        return j_theta + self.reg_param * (self.theta * self.theta)[1:].sum() / 2 / self.nx

    def predict(self, x):
        return self.__forward((x - self.mu) / self.s) if self.feature_normed \
                else self.__forward(x)

    def test_accuracy(self, test_data):
        count = 0
        total = 0
        for p, a in zip(np.argmax(self.predict(test_data[0]), axis=0), np.argmax(test_data[1], axis=0)):
            if p == a:
                count += 1
            total += 1
        return float(count) / total

    def __feature_norm(self):
        self.mu = self.x.mean(0)
        self.s = self.x.max(0) - self.x.min(0)
        for i in range(len(self.x)):
            self.x[i] = (self.x[i] - self.mu) / self.s

    def __forward(self, x):
        return Linear.sigmoid(self.theta.dot(np.insert(x, 0, 1, axis=0)))

    def __step(self, lr):
        for i in range(len(self.theta) - 1):
            self.theta[i + 1] = self.theta[i + 1] * (1 - lr * self.reg_param / self.nx)
        self.theta = self.theta - (self.__forward(self.x) - self.y) \
            .dot(np.insert(self.x, 0, 1, axis=0).transpose()) * lr / self.nx


if __name__ == "__main__":
    data = loadmat('ex3data1.mat')
    y = np.zeros((5000, 10), dtype=int)
    for i in range(len(y)):
        y[i][round(data['y'][i][0] - 1)] = 1
    training_data = (data['X'].transpose(), y.transpose())
    s = Linear(400, 10)
    s.set_data(training_data, do_feature_norm=False)
    iteration = 100
    lr = 0.01

    accuracy = [s.test_accuracy(training_data) * 100 for _ in s.learn_debug()]

    x = np.linspace(0, len(accuracy), len(accuracy))
    plt.scatter(x, accuracy)
    plt.title("iteration=%d, lr=%f, final=%f" % (iteration, lr, accuracy[-1]))
    plt.show()