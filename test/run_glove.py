import analysis_utils as au
import numpy as np
import CEOs as ceos

d = 200

X = np.loadtxt("/shared/Dataset/ANNS/FalconnPP/Glove_X_1183514_200.txt", dtype=np.float32, delimiter=None)
Q = np.loadtxt("/shared/Dataset/ANNS/FalconnPP/Glove_Q_1000_200.txt", dtype=np.float32, delimiter=None)

X = X.reshape((-1, d))
Q = Q.reshape((-1, d))

X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-10)

X_t = np.transpose(X)
Q_t = np.transpose(Q)

n_proj = 1024
repeats = 1
numThreads = 20
top_points = 100
seed = 1
k = 10

index = ceos.coCEOs(d)
index.setIndexParam(n_proj=n_proj, n_repeats=repeats, n_threads=numThreads, random_seed=seed, top_points=top_points)
index.build(X_t)

index.n_probedVectors = 50
index.n_cand = 500

kNN, _ = index.estimate_search(Q_t, k, True) # query has d x

exact_kNN, _ = au.perform_exact_nns(X, Q, k)

recall = au.recall(kNN, exact_kNN, k)

print("Recall: ", recall)