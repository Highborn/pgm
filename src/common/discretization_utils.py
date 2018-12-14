import math

import numpy as np

from config.runtime_config import RuntimeConfig


class DiscretizationUtils:
    @classmethod
    def descretize(cls, data, method, *args, **kwargs) -> np.ndarray:
        assert len(data.shape) is 1
        methods = [
            cls.orthogonal,
            cls.equal_width,
            cls.equal_frequency,
        ]
        for _method in methods:
            if method == _method.__name__:
                return _method(data, *args, **kwargs)
        else:
            raise ValueError('undefined method!')

    @classmethod
    def orthogonal(cls, data: np.ndarray, labels: np.ndarray, inertia_threshold=0):
        assert len(data.shape) is 1
        sorted_data = np.sort(data)

        bin_label = labels[0]
        cut_points = list()
        inertia = 0
        candidate = sorted_data[0]
        cut_points.append(candidate)

        for i in range(len(sorted_data)):
            if bin_label == labels[i]:
                candidate = sorted_data[i]
            else:
                inertia += 1

            if inertia > inertia_threshold:
                inertia = 0
                cut_points.append(candidate)
        for i in range(len(cut_points) - 1):
            data[(cut_points[i] <= data) & (data < cut_points[i + 1])] = i + 1
        return data

    @classmethod
    def equal_frequency(cls, data: np.ndarray, bin_qty: int) -> np.ndarray:
        assert len(data.shape) is 1

        data_slice = math.ceil(data.shape[0] / bin_qty)
        categorical_data = (data // data_slice) + 1
        if RuntimeConfig.DEBUG_MODE:
            print(categorical_data)
        return categorical_data

    @classmethod
    def equal_width(cls, data: np.ndarray, bin_qty: int) -> np.ndarray:
        assert len(data.shape) is 1

        data_min = data.min()
        data_max = data.max()
        data_max += data_max / (10**6)
        delta = (data_max - data_min) / bin_qty
        categorical_data = np.array(data)
        for i in range(categorical_data.shape[0]):
            categorical_data[i] = (categorical_data[i] - data_min) // delta + 1
        if RuntimeConfig.DEBUG_MODE:
            print(categorical_data)
        return categorical_data


if __name__ == '__main__':
    _data = np.array([3, 7, 1, 4, 2, 6, 5, 1, 1, 1])
    _label = np.array([1, 1, 2, 1, 2, 2, 1, ])
    _bin_qty = 3
    DiscretizationUtils.equal_frequency(_data, _bin_qty)
    DiscretizationUtils.equal_width(_data, _bin_qty)
