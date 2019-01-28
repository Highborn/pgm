from random import randint, random
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from common.singleton_meta_class import Singleton
from config.runtime_config import RuntimeConfig
from expectation_maximization.expectation_maximization_dal import ExpectationMaximizationDal


class ExpectationMaximizationManager(metaclass=Singleton):
    DEFAULT_VARIANCE = 1.
    MAX_EPOCH = 5000
    MIN_UPDATE = 0.0001

    dal = None
    cluster_qty = None
    mean_list = None
    variance_list = None
    theta_list = None
    data_qty = None

    def __init__(self, cluster_qty: int, mean_list: List[float] = None, variance_list: List[float] = None,
                 theta_list: List[float] = None, data_path=None):
        self.dal = ExpectationMaximizationDal(data_path=data_path)
        self.cluster_qty = cluster_qty
        self.mean_list = mean_list
        self.variance_list = variance_list
        self.theta_list = theta_list

    def _populate_variables(self, data: np.ndarray):
        delta = data.max() - data.min()
        if self.mean_list is None:
            self.mean_list = [(random() * delta + data.min()) for _ in range(self.cluster_qty)]
        if self.variance_list is None:
            self.variance_list = [self.DEFAULT_VARIANCE] * self.cluster_qty
        if self.theta_list is None:
            self.theta_list = [1 / self.cluster_qty] * self.cluster_qty
        self.data_qty = data.shape[0]

        if RuntimeConfig.DEBUG_MODE:
            print('*' * 70)
            print('cluster qty:', self.cluster_qty)
            print('mean list:', self.mean_list)
            print('variance list:', self.variance_list)
            print('theta list:', self.theta_list)
            print('data qty:', self.data_qty)
            print('*' * 70)

    def compute_variance(self, mean_list, theta_list, data):
        for mean, theta in zip(mean_list, theta_list):
            pass

    def compute_responsibility(self, evidence, mean_list, variance_list):
        responsibility_list = []
        for mean, variance in zip(mean_list, variance_list):
            numerator = np.exp(-(evidence - mean) ** 2 / (2 * variance))
            denominator = np.sqrt(2 * np.pi * variance)
            responsibility_list.append(numerator / denominator)
        return responsibility_list

    def compute_membership(self, evidence, theta_list, mean_list, variance_list):
        responsibility_list = self.compute_responsibility(evidence, mean_list, variance_list)
        membership = list()
        for theta, responsibility in zip(theta_list, responsibility_list):
            membership.append(theta * responsibility)
        membership_sum = np.sum(membership)
        membership = [x / membership_sum for x in membership]
        return membership

    def run(self):
        data = self.dal.get_processed_data()
        self._populate_variables(data)
        theta_list = list(self.theta_list)
        mean_list = list(self.mean_list)
        variance_list = list(self.variance_list)
        membership_array = np.zeros((len(data), self.cluster_qty))
        for epoch in range(self.MAX_EPOCH):
            for i in range(self.data_qty):
                membership_list = self.compute_membership(data[i], theta_list, mean_list, variance_list)
                membership_array[i] = np.array(membership_list)
            mean_array = np.dot(np.array([data, ]), membership_array) / np.sum(membership_array, axis=0)
            new_mean_list = list(mean_array.ravel())
            for idx in range(self.cluster_qty):
                mst = (data - new_mean_list[idx]) ** 2
                variance_array = np.dot(np.array([mst, ]), membership_array[:, idx]) / np.sum(membership_array[:, idx])
                variance_list[idx] = variance_array[0]
            theta_list = np.sum(membership_array, axis=0) / self.data_qty
            if RuntimeConfig.DEBUG_MODE:
                print('epoch: ', epoch, '\t', '-' * 70)
                print('\tmean list:', new_mean_list)
                print('\tvariance list:', variance_list)
                print('\ttheta list:', theta_list)
                print()
            if np.sqrt(np.sum((np.array(new_mean_list) - np.array(mean_list)) ** 2)) < self.MIN_UPDATE:
                break
            else:
                mean_list = new_mean_list
        self.plot_result(mean_list, variance_list, data, membership_array)

    def plot_result(self, mean_list, variance_list, data, membership_array):
        args = np.argsort(data)
        plt.hist(data[args], 50, density=True, facecolor='#%06x' % randint(0, 0xFFFFFF), alpha=0.75)
        color_array = ['#%06x' % randint(0, 0xFFFFFF) for _ in range(self.cluster_qty)]
        for mean, variance, color, membership in zip(mean_list, variance_list, color_array, membership_array.T):
            x_values = np.linspace(data.min(), data.max(), 120)
            plt.plot(x_values, self.gaussian(x_values, mean, variance), color=color)
        plt.ylabel('Probability')
        plt.title('Histogram of data')
        plt.axis([data.min(), data.max(), 0, 1])
        plt.grid(True)
        plt.show()

    def contour_plot(self, data):
        theta_list = [.5, .5]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        u = np.linspace(-1, 4, 21)
        x, y = np.meshgrid(u, u)
        z = np.zeros((441,))
        mean_array = np.array([(i, j) for i, j in zip(x.ravel(), y.ravel())])
        for idx, mean_list in enumerate(mean_array):
            density = 0
            for mean, theta in zip(mean_list, theta_list):
                density += theta * np.exp(-((data - mean) ** 2) / 2) / np.sqrt(np.pi * 2)
            z[idx] = np.sum(np.log(density))
        z = z.reshape(21, 21)
        ax.contourf(x, y, z)
        plt.show()

    def gaussian(self, x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


if __name__ == '__main__':
    manager = ExpectationMaximizationManager(
        cluster_qty=2,
        mean_list=[1, 2],
        theta_list=[0.33, 0.67],
    )
    manager.run()
