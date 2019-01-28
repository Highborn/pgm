import numpy

from config.runtime_config import RuntimeConfig


class ExpectationMaximizationDal:
    data_path = None

    def __init__(self, split_char=',', data_path=None):
        self.split_char = split_char
        if data_path is None:
            self.data_path = RuntimeConfig.BASE_PROJECT_DIR + 'static/data/data.csv'
        else:
            self.data_path = data_path

    def read_data(self) -> numpy.ndarray:
        data_path = self.data_path
        data_list = list()
        with open(data_path, 'r') as data_file:
            for line in data_file:
                record = float(line.strip())
                data_list.append(numpy.array(record))
        data_array = numpy.array(data_list)
        return data_array

    def get_processed_data(self) -> numpy.ndarray:
        data_matrix = self.read_data()
        return data_matrix


if __name__ == '__main__':
    manager = ExpectationMaximizationDal()
    data = manager.get_processed_data()
