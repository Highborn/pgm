from typing import Tuple

import numpy

from common.discretization_utils import DiscretizationUtils
from config.runtime_config import RuntimeConfig


class TanStructureEstimationDal:
    def __init__(self, split_char=',', bin_qty=None):
        self.split_char = split_char
        if not bin_qty:
            bin_qty = RuntimeConfig.BIN_QTY
        self.bin_qty = bin_qty

    def read_data(self):
        """Implement based on Diabetes data set:
        https://www.kaggle.com/uciml/pima-indians-diabetes-database/version/1
        """
        data_path = RuntimeConfig.BASE_PROJECT_DIR + 'static/data/diabetes.csv'
        data_list = list()
        with open(data_path, 'r') as dataset:
            for line in dataset:
                record = line.strip().split(self.split_char)
                record = [float(x) for x in record]
                data_list.append(numpy.array(record))
            data_array = numpy.array(data_list)
        return data_array

    def get_processed_data(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        data_matrix = self.read_data()
        data_matrix = data_matrix.transpose()
        label_array = data_matrix[-1]
        label_array.astype(int)
        feature_matrix = data_matrix[:-1]
        for i in range(feature_matrix.shape[0]):
            feature_matrix[i] = DiscretizationUtils.equal_width(feature_matrix[i], bin_qty=self.bin_qty)
        feature_matrix = feature_matrix.transpose()
        return label_array, feature_matrix
