import numpy
from config.runtime_config import RuntimeConfig


class BayesianNetworkDal:
    def __init__(self, split_char=','):
        self.split_char = split_char

    def read_data(self) -> numpy.ndarray:
        data_path = RuntimeConfig.BASE_PROJECT_DIR + 'static/data/breast_cancer.csv'
        data_list = list()
        with open(data_path, 'r') as data_file:
            for line in data_file:
                record = line.strip().split(self.split_char)
                data_list.append(numpy.array(record))
        data_array = numpy.array(data_list)
        return data_array


if __name__ == '__main__':
    manager = BayesianNetworkDal()
    data = manager.read_data()
