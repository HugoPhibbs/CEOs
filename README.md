## CEOs - A novel dimensionality reduction method with applications for ANNS

CEOs is a novel dimensionality reduction method that leverages the behavior of concomintants of extreme order statistics.
Different from the forklore random projection, CEOs uses a significantly larger number of random vectors `n_proj`.
The projection values on a few closest/furthest vectors to the query are enough to estimate inner product between data points and the query.
Building on the theory of CEOs, we propose several algorithmic ANNS solvers, including
* CEOs and coCEOs-Est for estimating `n` inner product values to answer ANNS.
* CEOs-Hash is a locality-sensitive hashing scheme for ANNS (with inner product and cosine), that uses much smaller indexing space, achieves competitive accuracy-speed tradeoffs, and supports streaming indexing update.

There are two versions of CEOs-Hash, including:
* CEOs-Hash1 uses `n_proj` number of random vectors, corresponding to `2 * n_proj` buckets
* CEOs-Hash2 uses a tensor strick with `2 * n_proj` number of random vectors, corresponding to `4 * n_proj * n_proj` buckets.

Each group of `n_proj` random vectors is simulated by the [FFHT](https://github.com/FALCONN-LIB/FFHT) to speed up the projection time.
CEOs variants support multi-threading for both indexing and querying by adding only ```#pragma omp parallel for```.
streamCEOs supports delete/insert new points into the index in the streaming fashion.

We use [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) that supports SIMD dot product computation.
We use [TSL](https://github.com/Tessil/robin-map) for the hash map and hash set using linear robin hood hashing for coCEOs-Est.
## Prerequisites

* A compiler with C++17 support
* CMake >= 3.27 (test on Ubuntu 20.04 and Python3)
* Ninja >= 1.10 
* Eigen >= 3.3
* Boost >= 1.71
* Pybinding11 (https://pypi.org/project/pybind11/) 

## Installation

Just clone this repository and run

```bash
python3 setup.py install
```

or 

```bash
mkdir build && cd build && cmake .. && make
```


## Test call

Data and query must be n x d matrices.

```
import CEOs

# Static CEOs-Hash2
top_m = 50
n_repeats = 2**8 # increase n_repeats will increase indexing time and space, but increase the accuracy given fixed top-m and probed_vectors
D = 2**8 # increase D will increase indexing time and space, but increase the accuracy given fixed top-m and probed_vectors
probed_vectors = n_repeats * 5
iProbe = 4
verbose = True
seed = -1 # -1 means random
centering = 1 # default 0

n, d = np.shape(X)
index = CEOs.CEOs(n, d)
index.setIndexParam(D, n_repeats, top_m, iProbe, n_threads, seed)
# index.centering = centering
index.build_CEOs_Hash2(X)  # X must have d x n
print('CEOs-Hash2 index time (s): {}'.format(timeit.default_timer() - t1))

index.n_probed_vectors = n_repeats * 5        
t1 = timeit.default_timer()
approx_kNN, approx_Dist = index.search_CEOs_Hash2(Q, k, verbose)  # search
print("\tCEOs-Hash2 query time (s): {}".format(timeit.default_timer() - t1))
print("\tCEOs-Hash2 accuracy: ", getAcc(exact_kNN, approx_kNN))

```

See details in test/netflix_benchmark.py

## Authors

It is developed by Ninh Pham.
If you want to cite CEOs in a publication, please use

```
@inproceedings{DBLP:conf/kdd/Pham21,
author       = {Ninh Pham},
title        = {Simple Yet Efficient Algorithms for Maximum Inner Product Search via
Extreme Order Statistics},
booktitle    = {{KDD} '21: The 27th {ACM} {SIGKDD} Conference on Knowledge Discovery
and Data Mining, Virtual Event, Singapore, August 14-18, 2021},
pages        = {1339--1347},
publisher    = {{ACM}},
year         = {2021},
url          = {https://doi.org/10.1145/3447548.3467345},
doi          = {10.1145/3447548.3467345},
}
```





