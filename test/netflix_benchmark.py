import utils
import numpy as np
import CEOs
import timeit

if __name__ == '__main__':

    path = "/shared/Dataset/ANNS/CEOs/"

    n = 17770
    d = 300
    bin_file = path + 'Netflix_X_17770_300.bin'
    X = utils.mmap_bin(bin_file, n, d)


    q = 999
    bin_file = path + "Netflix_Q_999_300.bin"
    Q = utils.mmap_bin(bin_file, q, d)

    # n_threads = 32
    # k = 100
    # exact_kNN = utils.faissBF(X, Q, k, n_threads, dist="l2")
    # exact_kNN = exact_kNN.astype(np.int32)
    # np.save(path + "Netflix_L2_k_100_indices.npy", exact_kNN)    # shape: (n, k), dtype: int32


    dist = "cosine"

    ## Cosine
    if dist == "cosine":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X /= norms # X = np.array(X, copy=True)  # makes it writable
        norms = np.linalg.norm(Q, axis=1, keepdims=True)
        norms[norms == 0] = 1
        Q /= norms # Q = np.array(Q, copy=True)  # makes it writable
    elif dist == "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = np.hstack((2*X, norms)) # X = np.array(X, copy=True)  # makes it writable
        Q = np.hstack((Q, np.ones((Q.shape[0], 1)))) # Q = np.array(Q, copy=True)  # makes it writable

    mu = X.mean(axis=0, keepdims=True)
    # X = X - mu
    # 2) draw rotation
    R = utils.srht_rotation(X.shape[1])
    # 3) rotate
    # X = X @ R
    # inverse (to map back): R_inv = R.T
    # Q = Q - mu
    # Q = Q @ R

    #-------------------------------------------------------------------
    n_threads = 32
    # k = 100
    # exact_kNN = utils.faissBF(X, Q, k, n_threads)
    # exact_kNN = exact_kNN.astype(np.int32)
    # np.save(path + "Netflix_Cosine_k_100_indices.npy", exact_kNN)    # shape: (n, k), dtype: int32

    if dist == 'ip':
        exact_kNN = np.load(path + "Netflix_Dot_k_100_indices.npy")  # shape: (n, k), dtype: int32
    elif dist == 'cosine':
        exact_kNN = np.load(path + "Netflix_Cosine_k_100_indices.npy")  # shape: (n, k), dtype: int32
    elif dist == 'l2':
        exact_kNN = np.load(path + "Netflix_L2_k_100_indices.npy")  # shape: (n, k), dtype: int32

    k = 10
    exact_kNN = exact_kNN[: , :k]
    seed = 42

    #-------------------------------------------------------------------
    # probed_vectors = 20
    # n_cand = 100
    # n_repeats = 2**1
    # D = 2**9
    # verbose = False
    # print("\nCEOs")
    # utils.ceos_est(exact_kNN, X, Q, k, D, probed_vectors, n_cand, n_repeats, n_threads=n_threads, seed=seed,verbose=verbose)

    #-------------------------------------------------------------------
    top_m = 500
    probed_vectors = 40
    n_cand = 100
    n_repeats = 2**1
    D = 2**6
    iProbe = 10
    verbose = True
    seed = -1

    print("\ncoCEOs-est")

    t1 = timeit.default_timer()
    index = CEOs.CEOs(n, d)
    index.setIndexParam(D, n_repeats, top_m, iProbe, n_threads, seed)
    index.centering = False
    index.build_coCEOs_Est(X)  # X must have n x d
    t2 = timeit.default_timer()
    print('coCEOs-Est index time: {}'.format(t2 - t1))

    index.n_cand = n_cand
    index.n_probed_vectors = probed_vectors
    t1 = timeit.default_timer()
    approx_kNN, approx_Dist = index.search_coCEOs_Est(Q, k, verbose)  # search
    print("\tcoCEOs-Est query time: {}".format(timeit.default_timer() - t1))
    print("\tcoCEOs-Est accuracy: ", utils.getAcc(exact_kNN, approx_kNN))

    utils.coceos_est(exact_kNN, X, Q, k, D, top_m, iProbe, probed_vectors, n_cand, n_repeats, n_threads=n_threads, seed=seed, centering=False)

    #-------------------------------------------------------------------
    top_m = 20
    probed_vectors = 100
    probed_points = top_m
    n_repeats = 2**3
    D = 2**9
    seed = 42
    print("\nCEOs-hash")
    utils.ceos_hash(exact_kNN, X, Q, k, D, top_m, iProbe, probed_vectors, probed_points, n_repeats, n_threads, seed=seed)

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






