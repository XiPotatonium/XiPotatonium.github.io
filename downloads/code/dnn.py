import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


class NeuralNetwork:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def __init__(self, shape):
        self.shape = shape
        self.thetas = []

    def predict(self, x):
        x = x.transpose()
        for theta in self.thetas:
            x = NeuralNetwork.sigmoid(theta.dot(np.insert(x, 0, 1, axis=0)))
        return x

    def test_accuracy(self, data):
        hit = 0
        total = 0
        for p, a in zip(np.argmax(self.predict(data[0]), axis=0), np.argmax(data[1].transpose(), axis=0)):
            if p == a:
                hit += 1
            total += 1
        return hit, total

    def j_theta(self, data, reg_param=0):
        x = data[0].transpose()
        y = data[1].transpose()
        m = len(data[0])
        for theta in self.thetas:
            x = NeuralNetwork.sigmoid(theta.dot(np.insert(x, 0, 1, axis=0)))
        j_theta = -(y * np.log(x) + (1 - y) * np.log(1 - x)).sum() / m
        for theta in self.thetas:
            j_theta = j_theta + reg_param * (theta * theta).transpose()[1:].sum() / 2 / m
        return j_theta

    def _gradient_checking(self, data, reg_param, res, m):
        ys = []
        epsilon = 1e-4
        for i in range(len(self.thetas)):
            for r in range(len(self.thetas[i])):
                for c in range(len(self.thetas[i][r])):
                    self.thetas[i][r][c] -= epsilon
                    t = self.j_theta(data, reg_param)
                    self.thetas[i][r][c] += 2 * epsilon
                    t -= self.j_theta(data, reg_param)
                    t /= (2 * epsilon)
                    self.thetas[i][r][c] -= epsilon
                    ys.append(t - res[i][r][c] / m)
        xs = np.linspace(0, len(ys), len(ys))
        plt.scatter(xs, ys)
        plt.show()

    def train(self, training_data, test_data=None, iteration=1500, lr=0.01, reg_param=0):
        self.thetas = [np.random.randn(y, x + 1) for x, y in zip(self.shape[:-1], self.shape[1:])]
        ret = []  # accuracy list
        m = len(training_data[0])
        x = training_data[0].transpose()
        y = training_data[1].transpose()
        for i in range(iteration):
            delta_theta = self._back_prop(x, y)
            # gradient checking if necessary
            # self.__gradient_checking(training_data, reg_param, delta_theta, m)
            # apply delta_theta
            for i in range(len(self.thetas)):
                self.thetas[i] = self.thetas[i] * (1 - lr / m * reg_param) - lr / m * delta_theta[i]
            ret.append(self.test_accuracy(test_data if test_data is not None else training_data))
        return ret

    def _back_prop(self, x, y):
        """

        :param x: transposed, x_0 not inserted
        :param y:
        :return:
        """
        a = [x]
        delta_theta = [np.zeros(t.shape) for t in self.thetas]
        # forward
        for theta in self.thetas:
            a.append(NeuralNetwork.sigmoid(theta.dot(np.insert(a[-1], 0, 1, axis=0))))
        # back prop
        delta = a[-1] - y
        for i in range(len(delta_theta)):
            delta_theta[-1 - i] = delta.dot(np.insert(a[-2 - i], 0, 1, axis=0).transpose())
            delta = (self.thetas[-1 - i].transpose().dot(delta)[1:] * (1 - a[-2 - i]) * a[-2 - i])
        return delta_theta


if __name__ == "__main__":
    data = loadmat('ex4data1.mat')
    y = np.zeros((len(data['y']), 10), dtype=int)
    for i in range(len(y)):
        y[i][round(data['y'][i][0] - 1)] = 1

    data = [(x, y) for x, y in zip(data['X'], y)]
    np.random.shuffle(data)

    x = np.array([data[i][0] for i in range(0, 4000)])
    y = np.array([data[i][1] for i in range(0, 4000)])
    training_data = (x, y)

    x = np.array([data[i][0] for i in range(4000, 5000)])
    y = np.array([data[i][1] for i in range(4000, 5000)])
    test_data = (x, y)

    s = NeuralNetwork((400, 25, 10))
    iteration = 2000
    lr = 3
    accuracy_data = s.train(training_data, test_data, iteration, lr)
    accuracy = []
    for hit, count in accuracy_data:
        accuracy.append(float(hit) / count)
    x = np.linspace(0, len(accuracy), len(accuracy))
    plt.scatter(x, accuracy)
    plt.title("iteration=%d, lr=%f, final=%f" % (iteration, lr, accuracy[-1]))
    plt.show()