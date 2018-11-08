import itertools
from collections import Counter
from random import shuffle, randint

import matplotlib.pyplot as plt
import numpy

from bayesian_network.bayesian_network_dal import BayesianNetworkDal
from common.singleton_meta_class import Singleton


class BayesianNetworkManager(metaclass=Singleton):
    MLE_KEY = 'mle'
    MAP_KEY = 'map'
    ALPHA_KEY = 'alpha'
    BETA_KEY = 'beta'
    ALPHA_DOMAIN = []
    BETA_DOMAIN = []

    def __init__(self, alpha_domain, beta_domain):
        self.dal = BayesianNetworkDal()

        self.prior = None
        self.alpha = None
        self.beta = None
        self.ALPHA_DOMAIN = alpha_domain
        self.BETA_DOMAIN = beta_domain

        # population variables ...
        self.evidence_qty = None
        self.feature_qty = None
        self.feature_domain = None
        self.label_domain = None
        self.label_count = None

    def set_prior(self, prior, **kwargs):
        if prior is not None:
            self.prior = prior
        if prior == self.MAP_KEY:
            self.alpha = kwargs.get(self.ALPHA_KEY)
            self.beta = kwargs.get(self.BETA_KEY)

    def _populate_variables(self, label_array: numpy.ndarray, feature_matrix: numpy.ndarray) -> None:
        self.evidence_qty = feature_matrix.shape[0]
        self.feature_qty = feature_matrix.shape[1]
        self.feature_domain = [sorted(set(x)) for x in feature_matrix.transpose()]
        self.label_domain = sorted(set(label_array))
        self.label_count = Counter(label_array)

    @staticmethod
    def create_frequency_table(labels, features):
        assert isinstance(labels, numpy.ndarray)
        assert isinstance(features, numpy.ndarray)

        frequency_table_list = list()
        features = features.transpose()
        for feature in features:
            label_factor = labels * 100
            labeled_feature = feature + label_factor.astype(int)
            feature_frequency_summary = Counter(labeled_feature)
            frequency_table_list.append(feature_frequency_summary)

        return frequency_table_list

    def compute_likelihood(self, frequency_table_list):
        likelihood_table_list = list(frequency_table_list)
        for feature_domain, likelihood_table in zip(self.feature_domain, likelihood_table_list):
            for key, value in likelihood_table.items():
                cls_key = key // 100
                laplace_denominator = self.label_count.get(cls_key) + len(feature_domain)
                likelihood_table[key] = (value + 1) / laplace_denominator
        return likelihood_table_list

    def compute_posterior(self, label_array, feature_matrix, likelihood_table_list):
        prior_translate_dict = {
            self.MLE_KEY: self.mle_prior,
            self.MAP_KEY: self.map_prior,
        }
        prior_dict = dict()
        for label in self.label_domain:
            prior_dict[label] = prior_translate_dict.get(self.prior)(label)

        prediction = numpy.zeros_like(label_array)
        for idx, evidence in enumerate(feature_matrix):
            best_posterior, best_label = None, None
            for label in self.label_domain:
                posterior = prior_dict.get(label)
                likelihood_list = list()
                for i in range(len(evidence)):
                    labeled_feature = evidence[i] + (100 * label)
                    likelihood = likelihood_table_list[i].get(labeled_feature, 0)
                    if likelihood == 0:
                        laplace_denominator = self.label_count.get(label) + len(self.feature_domain[i])
                        likelihood = 1 / laplace_denominator
                    likelihood_list.append(likelihood)
                posterior = numpy.prod(likelihood_list) * posterior
                if best_posterior is None or best_posterior < posterior:
                    best_posterior, best_label = posterior, label
            prediction[idx] = best_label
        return prediction

    def map_prior(self, label):
        if label < numpy.mean(self.label_domain):
            nominator = self.alpha - 1
        else:
            nominator = self.beta - 1
        nominator += self.label_count.get(label)
        denominator = self.evidence_qty + self.alpha + self.beta - 2
        prior = nominator / denominator
        print('prior:', prior)
        return prior

    def mle_prior(self, label):
        nominator = self.label_count.get(label)
        denominator = self.evidence_qty
        prior = nominator / denominator
        print('prior:', prior)
        return prior

    @staticmethod
    def accuracy(prediction, label_array):
        error: numpy.ndarray = (prediction - label_array)
        error_rate = numpy.count_nonzero(error) / len(error)
        print(error_rate)
        return error_rate

    def run_with_cross_validation(self):
        label_array, feature_matrix = self.dal.get_processed_data()
        evidence_qty = label_array.shape[0]
        indices = numpy.random.permutation(evidence_qty)
        fraction_index = evidence_qty * 8 // 10

        test_result = list()
        train_result = list()
        for i in range(20):
            shuffle(indices)
            shuffled_label_array = label_array[indices]
            shuffled_feature_matrix = feature_matrix[indices]

            train_labels = shuffled_label_array[:fraction_index]
            train_features = shuffled_feature_matrix[:fraction_index]
            self._populate_variables(train_labels, train_features)
            test_labels = shuffled_label_array[fraction_index:]
            test_features = shuffled_feature_matrix[fraction_index:]

            frequency_tables = self.create_frequency_table(train_labels, train_features)
            likelihood_tables = self.compute_likelihood(frequency_tables)

            run_label_list = [train_labels, test_labels]
            run_feature_list = [train_features, test_features]
            run_result_list = [train_result, test_result]
            for labels, features, result in zip(run_label_list, run_feature_list, run_result_list):
                accuracy_list = list()
                self.set_prior(BayesianNetworkManager.MLE_KEY)
                generic_prediction = self.compute_posterior(labels, features, likelihood_tables)
                generic_accuracy = self.accuracy(generic_prediction, labels)
                accuracy_list.append(generic_accuracy)
                for alpha in self.ALPHA_DOMAIN:
                    for beta in self.BETA_DOMAIN:
                        self.set_prior(BayesianNetworkManager.MAP_KEY, alpha=alpha, beta=beta)
                        beta_prediction = self.compute_posterior(labels, features, likelihood_tables)
                        beta_accuracy = self.accuracy(beta_prediction, labels)
                        accuracy_list.append(beta_accuracy)

                result.append(accuracy_list)
            print('-' * 20, i)

        print('=' * 20)
        train_result = numpy.array(train_result)
        test_result = numpy.array(test_result)
        print(self.meta_predictor(train_result))
        print(self.meta_predictor(test_result))
        self.plot_results(test_result)
        return test_result

    def plot_results(self, result_array):
        array_transpose = result_array.transpose()
        color_array = ['#%06x' % randint(0, 0xFFFFFF) for i in range(len(array_transpose))]
        color_detail = list()
        color_detail.append('generic')
        beta_details = ['alpha:%s beta:%s' % x for x in itertools.product(self.ALPHA_DOMAIN, self.BETA_DOMAIN)]
        color_detail.extend(beta_details)
        for result, color in zip(array_transpose, color_array):
            plt.plot(numpy.arange(1, 21), result, color)
        for i, result in enumerate(array_transpose):
            mean = numpy.array(result).mean(axis=0)
            color_detail[i] = str(mean) + ' - ' + color_detail[i]

        plt.legend(color_detail, loc='best', bbox_to_anchor=(1.05, 1))
        fig = plt.gcf()
        fig.set_size_inches(10, 5)
        plt.show()

    def meta_predictor(self, predictions):
        meta_prediction = numpy.array(predictions).mean(axis=0)
        return meta_prediction


if __name__ == '__main__':
    manager = BayesianNetworkManager([1, 2], [1, 2])

    manager.run_with_cross_validation()
