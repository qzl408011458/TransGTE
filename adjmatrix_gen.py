import pickle

import numpy as np
import math

from networkx.classes import edges

from utils.haversine_distance import haversine
import torch
from utils.MyDataset import Trip_parDataset


def GenAdjmatrix(row=50, column=50):
    adjmatrix = [[0 for j in range(row * column)] for i in range(row * column)]
    for i in range(len(adjmatrix)):
        for j in range(len(adjmatrix[i])):
            # 4 corners: 0, column - 1, (row-1) * column, row * column - 1, 3 neighbors
            # 4 edges(expect the corners): first row and last row, first column and last column, 5 neighbors
            # others: 8 neighbors

            # assume each grid has 8 neighbors, some of which may be out of the boundary
            # left, right, up, down, up-left, up-right, down-left, down-right

            r = int(i / column)
            c = i - r * column
            for r_neigh, c_neigh in [(r-1, c), (r+1, c), (r, c-1), (r, c+1),
                                     (r-1, c-1), (r+1, c-1), (r-1, c+1), (r+1, c+1)]:
                if r_neigh >= 0 and r_neigh < row and c_neigh >= 0 and c_neigh < column:
                    adjmatrix[i][r_neigh * column + c_neigh] = 1

    return np.array(adjmatrix)

def GenAdjEdge(row=50, column=50, pad=False):
    # adj_edges = []
    adjmatrix = [[0 for j in range(row * column)] for i in range(row * column)]
    for i in range(len(adjmatrix)):
        for j in range(len(adjmatrix[i])):
            # 4 corners: 0, column - 1, (row-1) * column, row * column - 1, 3 neighbors
            # 4 edges(expect the corners): first row and last row, first column and last column, 5 neighbors
            # others: 8 neighbors

            # assume each grid has 8 neighbors, some of which may be out of the boundary
            # left, right, up, down, up-left, up-right, down-left, down-right

            r = int(i / column)
            c = i - r * column
            for r_neigh, c_neigh in [(r-1, c), (r+1, c), (r, c-1), (r, c+1),
                                     (r-1, c-1), (r+1, c-1), (r-1, c+1), (r+1, c+1)]:
                if r_neigh >= 0 and r_neigh < row and c_neigh >= 0 and c_neigh < column:
                    adjmatrix[i][r_neigh * column + c_neigh] = 1
    adj_edges = []
    if pad:
        for i in range(row * column):
            for j in range(row * column):
                if adjmatrix[i][j]:
                    adj_edges.append([i + 1, j + 1])
    else:
        for i in range(row * column):
            for j in range(row * column):
                if adjmatrix[i][j]:
                    adj_edges.append([i, j])
    return adj_edges

# def GenAdjmatrix2(gtrjdata, device='cuda', row=50, column=50):
#     adjmatrix_list = []
#     for i in range(len(gtrjdata)):
#         adjmatrix = [[0 for j in range(row * column)] for i in range(row * column)]
#         gtrj_i = gtrjdata[i]
#         if len(gtrjdata[i]) > 1:
#             for i_g in range(len(gtrj_i) - 1):
#                 adjmatrix[gtrj_i[i_g]][gtrj_i[i_g + 1]] = 1
#                 adjmatrix[gtrj_i[i_g + 1]][gtrj_i[i_g]] = 1
#             adjmatrix = torch.tensor(adjmatrix, dtype=torch.float, device=device)
#             adjmatrix = process_graph(adjmatrix)
#         else:
#             adjmatrix = torch.tensor(adjmatrix, dtype=torch.float, device=device)
#         adjmatrix_list.append(adjmatrix.unsqueeze(0))
#     return torch.cat(adjmatrix_list, dim=0)
#
#
# def process_graph(graph_data):
#     N = graph_data.size(0)
#     matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)
#     graph_data += matrix_i  # A~ [N, N]
#
#     degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]
#     degree_matrix = degree_matrix.pow(-1)
#     degree_matrix[degree_matrix == float("inf")] = 0.  # [N]
#
#     degree_matrix = torch.diag(degree_matrix)  # [N, N]
#
#     return torch.mm(degree_matrix, graph_data)  # D^(-1) * A = \hat(A)
#
# if __name__ == '__main__':
#     # porto
#     device = 'cpu'
#     for pr in [0.1, 0.3, 0.5, 0.7, 0.9]:
#         path1 = '/home/vincentqin/PycharmProjects/DeepAGS_GPS_partTraj/data/porto/porto_municipality_dataset_r{}_dr0.5.pkl'.format(pr)
#
#         # path1 = 'data/chengdu/chengdu_dataset_r{}_dr0.5_2.pkl'.format(args.pr)
#
#         with open(path1, 'rb') as fr:
#             dataset = pickle.load(fr)
#
#         print(int(len(dataset['trainset']['dest'])))
#         print(int(len(dataset['valset']['dest'])))
#         print(int(len(dataset['testset']['dest'])))
#
#         trainset = Trip_parDataset(dataset['trainset'])
#         valset = Trip_parDataset(dataset['valset'])
#         testset = Trip_parDataset(dataset['testset'])
#         del dataset
#         train_adj_matrices = GenAdjmatrix2(trainset.gtrj, device=device)
#         val_adj_matrices = GenAdjmatrix2(valset.gtrj, device=device)
#         test_adj_matrices = GenAdjmatrix2(testset.gtrj, device=device)
#         print('pr{} is finished~~')
#         with open('data/porto/porto_adjmatrices2_r{}_dr0.5.pkl'.format(pr)):
#             data = {'train_adj': train_adj_matrices,
#                     'val_adj': val_adj_matrices,
#                     'test_adj': test_adj_matrices}