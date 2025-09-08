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
    # norms = np.linalg.norm(X, axis=1, keepdims=True)
    # norms[norms == 0] = 1
    # X /= norms # X = np.array(X, copy=True)  # makes it writable
    # norms = np.linalg.norm(Q, axis=1, keepdims=True)
    # norms[norms == 0] = 1
    # Q /= norms # Q = np.array(Q, copy=True)  # makes it writable

    #-------------------------------------------------------------------
    savePath = "/shared/Dataset/ANNS/CEOs/"
    n_threads = 32

    # k = 100
    # exact_kNN = utils.faissBF(X, Q, k, n_threads)
    # exact_kNN = exact_kNN.astype(np.int32)
    # np.save(path + "Glove200_Cosine_k_100_indices.npy", exact_kNN)    # shape: (n, k), dtype: int32

    exact_kNN = np.load(savePath + "Glove200_Cosine_k_100_indices.npy")  # shape: (n, k), dtype: int32
    # exact_kNN = np.load(path + "Gist_Dot_k_100_indices.npy")  # shape: (n, k), dtype: int32

    k = 10
    exact_kNN = exact_kNN[: , :k]

    #-------------------------------------------------------------------
    # probed_vectors = 20
    # n_cand = 50
    # n_repeats = 2**1
    # D = 2**10
    # print("\nCEOs")
    # verbose=True
    #
    # utils.ceos_est(exact_kNN, X, Q, k, D, probed_vectors, n_cand, n_repeats, n_threads, verbose=verbose)

    #-------------------------------------------------------------------
    # coCEOs-Est 1 layer
    top_m = 1000
    n_repeats = 2**8
    probed_vectors = 80
    n_cand = 10000

    D = 2**10
    iProbe = 3 # not used in 1 layer
    verbose = True
    seed = -1
    centering = True

    print("\ncoCEOs-est")

    t1 = timeit.default_timer()
    index = CEOs.CEOs(n, d)
    index.setIndexParam(D, n_repeats, top_m, iProbe, n_threads, seed)
    index.centering = True
    index.build_coCEOs_Est(X)  # X must have n x d
    t2 = timeit.default_timer()
    print('coCEOs-Est index time: {}'.format(t2 - t1))

    index.n_cand = n_cand
    index.n_probed_vectors = probed_vectors
    t1 = timeit.default_timer()
    approx_kNN, approx_Dist = index.search_coCEOs_Est(Q, k, verbose)  # search
    print("\tcoCEOs-Est query time: {}".format(timeit.default_timer() - t1))
    print("\tcoCEOs-Est accuracy: ", utils.getAcc(exact_kNN, approx_kNN))

    # utils.coceos_est(exact_kNN, X, Q, k, D, top_m, probed_vectors, n_cand, n_repeats, n_threads=n_threads, seed=seed, centering=centering)

    #-------------------------------------------------------------------
    # CEOs-Hash 1 layer
    top_m = 1000
    n_repeats = 2**8
    probed_vectors = 80
    probed_points = top_m

    D = 2**10
    iProbe = 10 # not used for 1 layer
    verbose = True
    seed = -1
    centering = True

    print("\nCEOs-hash")
    utils.ceos_hash(exact_kNN, X, Q, k, D, top_m, iProbe, probed_vectors, probed_points, n_repeats, n_threads,centering=centering)

    #-------------------------------------------------------------------
    # coCEOs-Est (2 layers)
    top_m = 1000
    n_repeats = 2**8
    probed_vectors = 80
    n_cand = 10000

    D = 2**8
    iProbe = 8
    verbose = True
    seed = 42
    centering = False

    print("\ncoCEOs-Est2")
    t1 = timeit.default_timer()
    index = CEOs.CEOs(n, d)
    index.setIndexParam(D, n_repeats, top_m, iProbe, n_threads, seed)
    index.centering = False
    index.build_coCEOs_Est2(X)  # X must have n x d
    t2 = timeit.default_timer()
    print('coCEOs-Est2 index time: {}'.format(t2 - t1))

    index.n_cand = n_cand
    index.n_probed_vectors = probed_vectors
    index.n_probed_points = top_m
    t1 = timeit.default_timer()
    approx_kNN, approx_Dist = index.search_coCEOs_Est2(Q, k, verbose)  # search
    print("\tcoCEOs-Est2 query time: {}".format(timeit.default_timer() - t1))
    print("\tcoCEOs-Est2 accuracy: ", utils.getAcc(exact_kNN, approx_kNN))

    #-------------------------------------------------------------------------------

    # CEOs-Hash 2 layers
    top_m = 20
    n_repeats = 2**8
    probed_vectors = n_repeats * 10

    D = 2**8
    iProbe = 4
    verbose = True
    seed = 42
    centering = False

    print("\nCEOs-Hash2")
    t1 = timeit.default_timer()
    index = CEOs.CEOs(n, d)
    index.setIndexParam(D, n_repeats, top_m, iProbe, n_threads, seed)
    index.centering = False
    index.build_CEOs_Hash2(X)  # X must have n x d
    t2 = timeit.default_timer()
    print('CEOs-Hash2 index time: {}'.format(t2 - t1))

    index.n_probed_vectors = probed_vectors
    index.n_probed_points = top_m
    t1 = timeit.default_timer()
    approx_kNN, approx_Dist = index.search_CEOs_Hash2(Q, k, verbose)  # search
    print("\tCEOs-Hash2 query time: {}".format(timeit.default_timer() - t1))
    print("\tCEOs-Hash2 accuracy: ", utils.getAcc(exact_kNN, approx_kNN))














    # #-------------------------------------------------------------------
    # print("\nFaiss-IVF")
    # utils.faissIVF(exact_kNN, X, Q, k, n_list=1000, n_probe=20, n_threads=n_threads)
    #
    # #-------------------------------------------------------------------
    # print("\nHnswlib")
    # utils.hnswMIPS(exact_kNN, X, Q, k, efSearch=100, n_threads=n_threads)


    #-------------------------------------------------------------------
    # Streaming test
    # print("\nStreamCEOs-est")
    #
    # top_m = 50
    # probed_vectors = 40
    # n_cand = 100
    # n_repeats = 2**1
    # D = 2**10
    # utils.streamCEOs_est(X, Q, k, top_m, probed_vectors, n_cand, n_repeats, n_threads)
    #
    # print("\nStreamCEOs-hash")
    # top_m = 10
    # probed_vectors = 40
    # probed_points = top_m
    # n_repeats = 2**1
    # D = 2**10
    # utils.streamCEOs_hash(X, Q, k, top_m, probed_vectors, n_repeats, n_threads)
    #-------------------------------------------------------------------