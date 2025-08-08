import numpy as np
import scann

def recall(actual_indices, expected_indices, k):
    """
    Calculate recall at k.

    Parameters:
    actual_indices (np.ndarray): Actual indices of the nearest neighbors. Shape (n_queries, m), for m>=k. Assume that
                                  actual_indices are sorted. I.e. closest neighbors are at the beginning (column-wise)
    expected_indices (np.ndarray): Expected indices of the nearest neighbors. Shape (n_queries, l), for l>=k. Assume that
                                   expected_indices are sorted. I.e. closest neighbors are at the beginning (column-wise)
    k (int): The number of nearest neighbors to consider.
    """
    assert actual_indices.shape[1] >= k, "actual_indices must have at least k columns"
    assert (
        expected_indices.shape[1] >= k
    ), "expected_indices must have at least k columns"

    actual_top_k = actual_indices[:, :k]
    expected_top_k = expected_indices[:, :k]

    n_queries = actual_top_k.shape[0]

    total = 0

    for i in range(n_queries):
        this_actual_top_k = actual_top_k[i]
        this_expected_top_k = expected_top_k[i]

        intersect = np.intersect1d(this_actual_top_k, this_expected_top_k)

        recall_value = len(intersect) / k

        total += recall_value

    return total / n_queries

def perform_exact_nns(X: np.ndarray, Q: np.ndarray, k: int):
    """
    Perform exact nearest neighbor search using ScaNN.

    Parameters:
    X (np.ndarray): Dataset to search in. Shape (n, d)
    Q (np.ndarray): Query points. Shape (n_queries, d)
    k (int): Number of nearest neighbors to find

    Returns:
    indices (np.ndarray): Indices of nearest neighbors. Shape (n_queries, k)
    distances (np.ndarray): Distances to nearest neighbors. Shape (n_queries, k)
    """
    searcher = scann.scann_ops_pybind.builder(X, k, "dot_product") \
        .score_brute_force(True) \
        .build()
    
    indices, distances = searcher.search_batched(Q)
    return np.array(indices), np.array(distances)