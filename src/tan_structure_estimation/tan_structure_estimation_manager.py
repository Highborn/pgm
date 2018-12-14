from collections import Counter
from datetime import datetime
from functools import reduce
from random import randint

import numpy as np
from graphviz import Digraph

from common.singleton_meta_class import Singleton
from config.runtime_config import RuntimeConfig
from tan_structure_estimation.tan_structure_estimation_dal import TanStructureEstimationDal


class TanStructureEstimationManager(metaclass=Singleton):
    def __init__(self):
        self.dal = TanStructureEstimationDal()
        self.mutual_information_dict = dict()
        self.evidence_qty = None
        self.feature_qty = None
        self.feature_domain = None
        self.label_domain = None
        self.label_count = None

    def _populate_variables(self, label_array: np.ndarray, feature_matrix: np.ndarray) -> None:
        self.evidence_qty = feature_matrix.shape[0]
        self.feature_qty = feature_matrix.shape[1]
        self.feature_domain = [sorted(set(x)) for x in feature_matrix.transpose()]
        self.label_domain = sorted(set(label_array))
        self.label_count = Counter(label_array)
        self.idx_matrix = self._create_idx_matrix(label_array, feature_matrix)

    def _create_idx_matrix(self, label_array, feature_matrix):
        idx_matrix = dict()
        for i in range(feature_matrix.T.shape[0]):
            for v in self.feature_domain[i]:
                key = i * 100 + v
                idx_matrix[key] = np.argwhere(feature_matrix.T[i] == v)
        for v in self.label_domain:
            key = -100 + v
            idx_matrix[key] = np.argwhere(label_array == v)
        return idx_matrix

    def compute_mutual_information(self, feature_matrix, label_array):
        for i in range(feature_matrix.shape[1]):
            for j in range(i):
                key = (i, j)
                self.mutual_information_dict[key] = self._compute_mutual_information(i, j)
                mirror_key = (j, i)
                self.mutual_information_dict[mirror_key] = self.mutual_information_dict[key]

    def _compute_mutual_information(self, i, j):
        mutual_info = 0
        for v in self.feature_domain[i]:
            i_key = i * 100 + v
            xi_args = self.idx_matrix[i_key]
            for u in self.feature_domain[j]:
                j_key = j * 100 + u
                xj_args = self.idx_matrix[j_key]
                for c in self.label_domain:
                    c_key = -1 * 100 + c
                    c_args = self.idx_matrix[c_key]
                    n = c_args.shape[0]
                    p_xi_xj_c = reduce(np.intersect1d, (xi_args, xj_args, c_args)).shape[0] / self.evidence_qty
                    p_xi_xj_lc = reduce(np.intersect1d, (xi_args, xj_args, c_args)).shape[0] / n
                    p_xi_lc = np.intersect1d(xi_args, c_args).shape[0] / n
                    p_xj_lc = np.intersect1d(xj_args, c_args).shape[0] / n
                    log_expression = p_xi_xj_lc / (p_xi_lc * p_xj_lc)
                    if log_expression == 0.:
                        log_expression = 1
                    if RuntimeConfig.DEBUG_MODE:
                        print(log_expression)
                    mutual_info += p_xi_xj_c * np.log(log_expression)
        print('mi:', mutual_info)
        return mutual_info

    def find_mst(self):
        mst_edges = list()
        jungle = [{x} for x in range(self.feature_qty)]
        while len(jungle) > 1:
            max_mutual_information = 0
            edge = None
            tree_indices = None
            for i in range(len(jungle)):
                for j in range(i):
                    for src_node in jungle[i]:
                        for dst_node in jungle[j]:
                            key = (src_node, dst_node)
                            mutual_information = self.mutual_information_dict[key]
                            if max_mutual_information < mutual_information:
                                max_mutual_information = mutual_information
                                tree_indices = (i, j)
                                edge = (src_node, dst_node)
            jungle[tree_indices[0]].update(jungle[tree_indices[-1]])
            jungle.pop(tree_indices[-1])
            mst_edges.append(edge)
        return mst_edges

    def create_dag(self, mst_edges):
        feature_quantity = self.feature_qty
        root = randint(0, feature_quantity - 1)
        edges_domain = list(mst_edges)
        node_domain = set(range(feature_quantity))
        node_domain.remove(root)
        dag_edges = list()
        self.find_child(root, edges_domain, node_domain, dag_edges)
        return dag_edges

    def find_child(self, root, edges_domain, node_domain, dag_edges):
        for i, edge in enumerate(edges_domain):
            if edge[1] == root:
                edge = tuple(reversed(edge))
            if edge[0] == root:
                new_root = edge[1]
                if new_root in node_domain and tuple(reversed(edge)) not in dag_edges:
                    dag_edges.append(edge)
                    node_domain.remove(new_root)
                    edges_domain = self.find_child(new_root, edges_domain, node_domain, dag_edges)
        return edges_domain

    def visualize_tan(self, mst_edges):
        feature_qty = self.feature_qty
        tan_edges = ['C' + str(x) for x in range(feature_qty)]
        tan_edges.extend([str(x[0]) + str(x[1]) for x in mst_edges])
        time_stamp = datetime.timestamp(datetime.now())
        file_name = RuntimeConfig.BASE_PROJECT_DIR + 'static/graph_' + str(time_stamp)

        dot = Digraph()
        dot.node('C', 'C')
        for i in range(feature_qty):
            dot.node(str(i), str(i))
        dot.edges(tan_edges)

        print(dot.source)
        dot.render(file_name, view=True)

    def learn(self, *args, **kwargs):
        label_array, feature_matrix = self.dal.get_processed_data()
        self._populate_variables(label_array, feature_matrix)
        # creating full pear to pear mutual information (G_f)
        self.compute_mutual_information(feature_matrix, label_array)
        mst_edges = self.find_mst()
        dag_edges = self.create_dag(mst_edges)
        self.visualize_tan(dag_edges)


if __name__ == '__main__':
    manager = TanStructureEstimationManager()
    manager.learn()
