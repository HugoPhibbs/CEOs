import analysis_utils as au
import numpy as np
import CEOs as ceos

# dataset_path = "/home/hphi344/Documents/CEOs/test/datasets/glove-100_dataset.bin"
# queries_path = "/home/hphi344/Documents/CEOs/test/datasets/glove-100_queries.bin"

# X = np.fromfile(dataset_path, dtype=np.float32)
# Q = np.fromfile(queries_path, dtype=np.float32)

# X = X.reshape((1_183_514, 100))
# Q = Q.reshape((10_000, 100))

X = np.loadtxt("/shared/Dataset/ANNS/CosineKNN/Yahoo_X_624961_300.txt", dtype=np.float32, delimiter=None)
Q = np.loadtxt("/shared/Dataset/ANNS/CosineKNN/Yahoo_Q_1000_300.txt", dtype=np.float32, delimiter=None)

# X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
# Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-10)

Q = Q[:1000, :] 

X_t = np.transpose(X)
Q_t = np.transpose(Q)

n_features = X_t.shape[0]

n_proj = 1024
repeats = 1
numThreads = 20
top_points = 100
seed = 1
k = 10

index = ceos.coCEOs(n_features)
index.setIndexParam(n_proj=n_proj, n_repeats=repeats, n_threads=numThreads, random_seed=seed, top_points=top_points)
index.build(X_t)

index.n_probedVectors = 50
index.n_cand = 500

kNN, _ = index.estimate_search(Q_t, k, True) # query has d x

exact_kNN, _ = au.perform_exact_nns(X, Q, k)

recall = au.recall(kNN, exact_kNN, k)

print("Recall: ", recall)