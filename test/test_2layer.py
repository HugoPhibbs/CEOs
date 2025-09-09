import utils
import numpy as np
import CEOs
import timeit

if __name__ == '__main__':

    path = "/shared/Dataset/ANNS/FalconnPP/"

    n = 1183514
    d = 200
    bin_file = path + 'Glove_X_1183514_200.bin'
    X = utils.mmap_bin(bin_file, n, d)

    q = 1000
    bin_file = path + "Glove_Q_1000_200.bin"
    Q = utils.mmap_bin(bin_file, q, d)

    ## Cosine
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X /= norms # X = np.array(X, copy=True)  # makes it writable
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    norms[norms == 0] = 1
    Q /= norms # Q = np.array(Q, copy=True)  # makes it writable

    #-------------------------------------------------------------------
    savePath = "/shared/Dataset/ANNS/CEOs/"
    n_threads = 32

    # k = 100
    # exact_kNN = utils.faissBF(X, Q, k, n_threads)
    # exact_kNN = exact_kNN.astype(np.int32)
    # np.save(path + "Glove200_Cosine_k_100_indices.npy", exact_kNN)    # shape: (n, k), dtype: int32

    exact_kNN = np.load(savePath + "Glove200_Cosine_k_100_indices.npy")  # shape: (n, k), dtype: int32

    k = 10
    exact_kNN = exact_kNN[: , :k]
    #-------------------------------------------------------------------
    # top_m = 1000
    # n_repeats = 2**6
    # probed_vectors = 80
    # n_cand = 20000
    #
    # D = 2**10
    # iProbe = -1
    # verbose = True
    # seed = -1
    # centering = True
    #
    # print("\ncoCEOs-est1")
    # utils.coceos_est1(exact_kNN, X, Q, k, D, top_m, iProbe, probed_vectors, n_cand, n_repeats, n_threads=n_threads, seed=seed, centering=centering)

    #-------------------------------------------------------------------

    # top_m = 1000
    # n_repeats = 2**6
    # probed_vectors = 80
    # n_cand = 30000
    #
    # D = 2**8
    # iProbe = 8
    # verbose = True
    # seed = -1
    # centering = False
    #
    # print("\ncoCEOs-Est2")
    # utils.coceos_est2(exact_kNN, X, Q, k, D, top_m, iProbe, probed_vectors, n_cand, n_repeats, n_threads=n_threads, seed=seed, centering=centering)

    #-------------------------------------------------------------------
    # top_m = 500
    # n_repeats = 2**8 # increase n_repeats will increase indexing time and space, but increase the accuracy given fixed top-m and probed_vectors
    # D = 2**10 # increase D will increase indexing time and space, but increase the accuracy given fixed top-m and probed_vectors
    # probed_vectors = 160
    # iProbe = 0 # not used in 1 layer case
    # verbose = True
    # seed = -1 # -1 means random
    # centering = False
    #
    # print("\nCEOs-hash")
    # probed_points = top_m # numDist = probed_points * probed_vectors
    # utils.ceos_hash1(exact_kNN, X, Q, k, D, top_m, iProbe, probed_vectors, probed_points, n_repeats, n_threads,centering=centering)

    #-------------------------------------------------------------------
    top_m = 50
    n_repeats = 2**8 # increase n_repeats will increase indexing time and space, but increase the accuracy given fixed top-m and probed_vectors
    D = 2**8 # increase D will increase indexing time and space, but increase the accuracy given fixed top-m and probed_vectors
    probed_vectors = n_repeats * 5
    iProbe = 4
    verbose = True
    seed = -1 # -1 means random
    centering = 1

    print("\nCEOs-hash2")
    probed_points = top_m # numDist = probed_points * probed_vectors
    utils.ceos_hash2(exact_kNN, X, Q, k, D, top_m, iProbe, probed_vectors, probed_points, n_repeats, n_threads,centering=centering)

    #-------------------------------------------------------------------
    # utils.streamCEOs_test(exact_kNN, X, Q, k, top_m, probed_vectors, n_cand, n_repeats, n_threads)
    # print("\nStreamCEOs-est")
    #
    # top_m = 500
    # probed_vectors = 40
    # n_cand = 100
    # n_repeats = 2**1
    # D = 2**9
    # utils.streamCEOs_est(X, Q, k, top_m, probed_vectors, n_cand, n_repeats, n_threads)
    #
    # print("\nStreamCEOs-hash")
    # top_m = 50
    # probed_vectors = 40
    # n_repeats = 2**1
    # D = 2**9
    # utils.streamCEOs_hash(X, Q, k, top_m, probed_vectors, n_repeats, n_threads)

    #-------------------------------------------------------------------
    # print("\nFaiss-IVF")
    # utils.faissIVF(exact_kNN, X, Q, k, n_list=1000, n_probe=20, n_threads=n_threads, dist='ip')

    # #-------------------------------------------------------------------
    # print("\nHnswlib")
    # utils.hnswMIPS(exact_kNN, X, Q, k, efSearch=100, n_threads=n_threads)






