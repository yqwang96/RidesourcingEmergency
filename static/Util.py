# @Time : 2023/5/17 15:16 
# @Author : Yinquan Wang<19114012@bjtu.edu.cn>
# @File : Util.py 
# @Function:
import numpy as np
from collections import deque

nearest_num = 100


def nearest_mask(matrix):
    idx = np.argsort(matrix, axis=0)
    mask = np.zeros_like(idx)
    # mask[np.arange(mask.shape[0])[:, None], idx[:, :nearest_num]] = 1
    mask[idx[:nearest_num, :], np.arange(mask.shape[1])] = 1
    out = np.where(mask, matrix, np.nan)
    return out


def nearest_mask_for_time_distance(time_matrix, distance_matrix):
    idx = np.argsort(time_matrix, axis=0)
    mask = np.zeros_like(idx)
    mask[idx[:nearest_num, :], np.arange(mask.shape[1])] = 1
    time_out = np.where(mask, time_matrix, np.nan)
    distance_out = np.where(mask, distance_matrix, np.nan)
    return time_out, distance_out


def bfs_split_graph(matrix, matrix_limitation):
    """
    Perform a breadth-first search (BFS) to divide the graph into subgraphs.

    Parameters:
        matrix (np.ndarray): A 2D numpy array representing the adjacency matrix of the graph.
        matrix_limitation (float): A threshold for determining if two nodes are connected.

    Returns:
        all_sub_graph (List[List[np.ndarray]]): A list of subgraphs. Each subgraph is a list
        containing two numpy arrays. The first array contains the indices of the passenger nodes
        in the subgraph, and the second array contains the indices of the driver nodes in the subgraph.
    """
    n_A, n_B = matrix.shape[0], matrix.shape[1]
    adjacency = np.where(matrix <= matrix_limitation, 1, 0)
    graph = np.zeros((n_A + n_B, n_A + n_B))
    graph[:n_B, n_B:] = adjacency.T
    graph[n_B:, :n_B] = adjacency

    n = n_A + n_B
    visited = np.zeros(n, dtype=bool)
    non_connect_node = np.where(np.all(graph == 0, axis=1))[0]  # 没有邻接的节点
    visited[non_connect_node] = True  # 将这些节点标记为true，默认不匹配
    divided = set(non_connect_node)  # 存储已经切分的节点
    all_sub_graph = []  # 存储所有划分后的子图
    while len(divided) < n:
        start = np.nonzero(~visited)[0][0]  # 获取当前没有访问的节点中的第一个节点
        visited[start] = True
        divided.add(start)
        node_sub_graph = {start}
        adjacent_node = set(graph[start].nonzero()[0])
        node_sub_graph.update(adjacent_node)
        adjacent_node = deque(adjacent_node)
        while adjacent_node:
            cur = adjacent_node.popleft()
            visited[cur] = True
            cur_adjacent_node = set(graph[cur].nonzero()[0])  # 获取子节点的邻接节点
            adjacent_node.extend(cur_adjacent_node - node_sub_graph)
            node_sub_graph.update(cur_adjacent_node)
        divided = divided.union(node_sub_graph)
        node_sub_graph = np.array(list(node_sub_graph))
        driver = node_sub_graph[node_sub_graph < n_B]
        passenger = node_sub_graph[~(node_sub_graph < n_B)] - n_B
        all_sub_graph.append([passenger, driver])
    return all_sub_graph


def reconstruct_matrix(split_graph, matrix):
    """
    Reconstruct the adjacency matrices of subgraphs from the original adjacency matrix.

    Parameters:
        split_graph (List[List[np.ndarray]]): A list of subgraphs. Each subgraph is a list
        containing two numpy arrays. The first array contains the indices of the passenger nodes
        in the subgraph, and the second array contains the indices of the driver nodes in the subgraph.
        matrix (np.ndarray): The original adjacency matrix.

    Returns:
        List[np.ndarray]: A list of adjacency matrices for the subgraphs.
    """
    return [matrix[:, graph[1]][graph[0], :] for graph in split_graph]