from typing import Tuple

import numpy
from config.runtime_config import RuntimeConfig


class BayesianNetworkDal:
    def __init__(self, split_char=','):
        self.split_char = split_char

    def read_data(self) -> numpy.ndarray:
        """Implement based on breastCancer unformatted data set:
        https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/unformatted-data
        https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names
        """
        data_path = RuntimeConfig.BASE_PROJECT_DIR + 'static/data/binary_breast_cancer.csv'
        data_list = list()
        with open(data_path, 'r') as data_file:
            for line in data_file:
                record = line.strip().split(self.split_char)
                record = [int(x) for x in record]
                data_list.append(numpy.array(record))
        data_array = numpy.array(data_list)
        return data_array

    def get_processed_data(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Implement based on breastCancer unformatted data set:
        https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/unformatted-data
        https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names
        """
        data_matrix = self.read_data()
        data_matrix = data_matrix.transpose()
        label_array = data_matrix[1]
        label_array.astype(int)
        feature_matrix = data_matrix[2:]
        feature_matrix = feature_matrix.transpose()
        return label_array, feature_matrix


if __name__ == '__main__':
    manager = BayesianNetworkDal()
    data = manager.get_processed_data()
