
#include "CEOs.h"
#include "Header.h"
#include "Utilities.h"
#include <stdexcept>

// __builtin_popcount function
// #include <bits/stdc++.h>

/**
 * Build index of CEOs for estimating inner product
 *
 * We have D random vectors, n_repeats times, so in total we have D * n_repeats random vectors
 * We execute FHT n_repeats times, each time with 3 random rotations
 * We store the projection values of all data points to these D * n_repeats random vectors in the column-wise Matrix_P of size N x (D * n_repeats)
 *
 * Algorithm:
 * - We parallel on the point Xi
 * - For each point Xi, we execute n_repeats times of FHT, storing its projection values in 1 x (D * n_repeats) row vector of Matrix_P
 *
 * @param matX: Note that CEOs::matrix_X has not been initialized, so we need to send param matX
 * We can avoid duplicated memory by loading data directly from the filename if dataset is big.
 *
 * Passing reference Eigen: https://stackoverflow.com/questions/21132538/correct-usage-of-the-eigenref-class
 */

void CEOs::build_CEOs(const Ref<const RowMajorMatrixXf> & matX)
{
    cout << "Building CEOs index..." << endl;
    cout << "n_points: " << CEOs::n_points << endl;
    cout << "n_features: " << CEOs::n_features << endl;
    cout << "n_repeats: " << CEOs::n_repeats << endl;
    cout << "n_proj: " << CEOs::n_proj << endl;
    cout << "fhtDim: " << CEOs::fhtDim << endl;


    auto start = chrono::high_resolution_clock::now();

    omp_set_num_threads(CEOs::n_threads);

    auto copy_start = chrono::high_resolution_clock::now();

    CEOs::matrix_X = matX; // No centering
    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - copy_start);
    cout << "Copying data time (s): " << (float)duration.count() / 1000 << endl;

    // Note that if fhtDim > n_proj, then we need to retrieve the first n_proj columns of the projections
    // This will save memory footprint if fhtDim is much larger than n_proj
    // We need N x (proj * repeat) since query phase will access each column corresponding each random vector
    CEOs::matrix_P = MatrixXf::Zero(CEOs::n_points, CEOs::n_proj * CEOs::n_repeats);

    bitHD3Generator(CEOs::fhtDim * CEOs::n_rotate * CEOs::n_repeats, CEOs::seed, CEOs::bitHD1);

    int log2_FHT = log2(CEOs::fhtDim);

#pragma omp parallel for
    for (int n = 0; n < CEOs::n_points; ++n)
    {
        VectorXf tempX = VectorXf::Zero(CEOs::fhtDim);
        tempX.segment(0, CEOs::n_features) = CEOs::matrix_X.row(n); //row-major

        // For each repeat
        for (int repeat = 0; repeat < CEOs::n_repeats; ++repeat) {

            VectorXf rotatedX = tempX;
            int baseIdx = CEOs::fhtDim * CEOs::n_rotate * repeat;

            for (int rotate = 0; rotate < CEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedX(d) *= (2 * static_cast<float>(CEOs::bitHD1[baseIdx + rotate * CEOs::fhtDim + d]) - 1);
                }

                fht_float(rotatedX.data(), log2_FHT);
            }

            // Store it into the matrix_P of size N x (n_proj * n_repeats)
//            cout << CEOs::n_proj * r + 0 << " " << CEOs::n_proj * r + CEOs::n_proj << endl;
//            cout << rotatedX.segment(0, CEOs::n_proj).transpose() << endl;

            // Note for segment(i, size) where i is starting index, size is segment size
            // Note that matrix_P is col-major, of size (n, n_repeat * n_proj)
            // For each repeat, we store the n_proj projection values
            CEOs::matrix_P.row(n).segment(CEOs::n_proj * repeat + 0, CEOs::n_proj) = rotatedX.segment(0, CEOs::n_proj); // only get up to #n_proj
//            CEOs::matrix_P.block(n, CEOs::n_proj * r + 0, 1, CEOs::n_proj) = rotatedX.segment(0, CEOs::n_proj).transpose();
//            cout << matrix_P.row(n) << endl;
        }
    }

    double dSize = 1.0 * (CEOs::matrix_P.rows() * CEOs::matrix_P.cols() + CEOs::matrix_X.rows() * CEOs::matrix_X.cols()) * sizeof(float) / (1 << 30);
    cout << "Size of CEOs index (including X) in GB: " << dSize << endl;

    dSize = 1.0 * CEOs::matrix_X.rows() * CEOs::matrix_X.cols() * sizeof(float) / (1 << 30);
    cout << "Size of data set in GB: " << dSize << endl;

    duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "CEOs construction time (s): " << (float)duration.count() / 1000 << endl;
}

/**
 * We treat all vectors n_repeats * D as independent random vectors, selecting top-s closest/furthest vectors to the query
 * then aggregate the projections of these vectors to estimate the inner product with all data points
 *
 * Algorithm:
 * - We parallel on the query Qi
 * - For each Qi, we execute n_repeats times of FHT, getting its projection values to all n_repeats * D random vectors, and selecting top-n_probed_vectors closest/furthest vectors
 * - We aggregate the projections of these selected vectors to estimate the inner product with all data points
 * - There is a re-ranking process with n_cand where we extract top-n_cand points with largest estimated inner product, and compute the exact inner product with these n_cand points
 *
 * @param matQ
 * @param n_neighbors
 * @param verbose
 * @return
 */
tuple<RowMajorMatrixXi, RowMajorMatrixXf> CEOs::search_CEOs(const Ref<const RowMajorMatrixXf> & matQ, int n_neighbors, bool verbose)
{
    int n_queries = matQ.rows();

    if (verbose)
    {
        cout << "n_probed_vectors: " << CEOs::n_probed_vectors << endl;
        cout << "n_probed_points: " << CEOs::n_probed_points << endl;
        cout << "n_cand: " << CEOs::n_cand << endl;
        cout << "n_threads: " << CEOs::n_threads << endl;

        cout << "n_queries: " << n_queries << endl;
    }

    // Collecting time for each thread
    // vector<double> projTime_thr(CEOs::n_threads, 0.0);
    // vector<double> estTime_thr(CEOs::n_threads, 0.0);
    // vector<double> candTime_thr(CEOs::n_threads, 0.0);
    // vector<double> distTime_thr(CEOs::n_threads, 0.0);
    // vector<double> threadWall_thr(CEOs::n_threads, 0.0);
    //
    // double wall_start = omp_get_wtime();

    auto startQueryTime = chrono::high_resolution_clock::now();

    float projTime = 0.0, estTime = 0.0, distTime = 0.0, candTime = 0.0;

    RowMajorMatrixXi matTopK = RowMajorMatrixXi::Zero(n_queries, n_neighbors );
    RowMajorMatrixXf matTopDist = RowMajorMatrixXf::Zero(n_queries, n_neighbors);

    int log2_FHT = log2(CEOs::fhtDim);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(CEOs::n_threads);

#pragma omp parallel for reduction(+:projTime, estTime, candTime, distTime)
    for (int q = 0; q < n_queries; ++q)
    {
        // const int tid = omp_get_thread_num();
        // double region_start = omp_get_wtime();
        // double t0 = omp_get_wtime();

        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.row(q);
        VectorXf vecProject = VectorXf (CEOs::n_proj * CEOs::n_repeats);

        // Step 1: Projection and getting top-s closest/furthest vectors among n_repeats * n_proj vectors
        for (int repeat = 0; repeat < CEOs::n_repeats; ++repeat)
        {
            int baseIdx = CEOs::fhtDim * CEOs::n_rotate * repeat;
            VectorXf rotatedQ = VectorXf::Zero(CEOs::fhtDim);

            rotatedQ.segment(0, CEOs::n_features) = vecQuery;

            for (int rotate = 0; rotate < CEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedQ(d) *= (2 * static_cast<float>(CEOs::bitHD1[baseIdx + rotate * CEOs::fhtDim + d]) - 1);
                }
                fht_float(rotatedQ.data(), log2_FHT);
            }

            // Note for segment(i, size) where i is starting index, size is segment size
            vecProject.segment(CEOs::n_proj * repeat + 0, CEOs::n_proj) = rotatedQ.segment(0, CEOs::n_proj); // only get up to #n_proj
        }

        // Note: Find the top-s (n_probed_vectors) closest/furthest random vectors among n_repeats * n_proj vectors
        // Quick-select might be faster but more complicated since we have to deal with both closest/furthest vectors
        priority_queue< IFPair, vector<IFPair>, greater<> > minQue;

        /**
         * matrix_P contains the projection values, of size n x (n_repeats * n_proj)
         * For query, we apply a simple trick to restore the furthest/closest vector
         * We increase index by 1 to get rid of the case of 0, and store a sign to differentiate the closest/furthest
         * Remember to convert this value back to the corresponding index of matrix_P
         * This fix is only for the rare case where r_0 at exp = 0 has been selected, which happen with very tiny probability
         */
        for (int d = 0; d < CEOs::n_repeats * CEOs::n_proj; ++d)
        {
            float fAbsProjValue = vecProject(d);

            // Hack: We increase by 1 since the index start from 0 and cannot have +/-
            // This trick would make implementation simpler compared to d + n_proj if negative since we have many repeats
            int iCol_plus_one = d + 1;

            if (fAbsProjValue < 0)
            {
                fAbsProjValue = -fAbsProjValue; // get abs
                iCol_plus_one = -iCol_plus_one; // use minus to indicate furthest vector
            }

            if ((int)minQue.size() < CEOs::n_probed_vectors)
                minQue.emplace(iCol_plus_one, fAbsProjValue);

            // queue is full
            else if (fAbsProjValue > minQue.top().m_fValue)
            {
                minQue.pop(); // pop max, and push min hash distance
                minQue.emplace(iCol_plus_one, fAbsProjValue);
            }

        }

        // projTime_thr[tid] += omp_get_wtime() - t0;
        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;


        // Step 2: Estimating inner product with all data points by aggregating the selected projections
        // t0 = omp_get_wtime();
        startTime = chrono::high_resolution_clock::now();

        VectorXf vecEst = VectorXf::Zero(CEOs::n_points);
        for (int i = 0; i < CEOs::n_probed_vectors; ++i)
        {
            IFPair ifPair = minQue.top();
            minQue.pop();

            // Negative, means furthest away
            if (ifPair.m_iIndex < 0)
                vecEst -= CEOs::matrix_P.col(-ifPair.m_iIndex - 1);  // We need to subtract 1 since we increased by 1 for the case of selected r_0
            else
                vecEst += CEOs::matrix_P.col(ifPair.m_iIndex - 1); // We need to subtract 1 since we increased by 1 for the case of selected r_0
        }

        // estTime_thr[tid] += omp_get_wtime() - t0;
        estTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        // Step 3: Extracting candidates with largest estimated inner product
        // t0 = omp_get_wtime();
        startTime = chrono::high_resolution_clock::now();

        for (int n = 0; n < CEOs::n_points; ++n)
        {
            if ((int)minQue.size() < CEOs::n_cand)
                minQue.emplace(n, vecEst(n));

                // queue is full
            else if (vecEst(n) > minQue.top().m_fValue)
            {
                minQue.pop(); // pop max, and push min hash distance
                minQue.emplace(n, vecEst(n));
            }
        }

        // candTime_thr[tid] += omp_get_wtime() - t0;
        candTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        // Step 4: Compute distance to n_cand candidates and extract top-k
        // t0 = omp_get_wtime();
        startTime = chrono::high_resolution_clock::now();
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;

        while (! minQue.empty())
        {
            IFPair ifPair = minQue.top();
            minQue.pop();

            int iPointIdx = ifPair.m_iIndex;

            float fInnerProduct = vecQuery.dot(CEOs::matrix_X.row(iPointIdx));

            // Add into priority queue
            if (int(minQueTopK.size()) < n_neighbors)
                minQueTopK.emplace(iPointIdx, fInnerProduct);

            else if (fInnerProduct > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.emplace(iPointIdx, fInnerProduct);
            }
        }

        for (int k = (int)minQueTopK.size() - 1; k >= 0; --k)
        {
            matTopK(q, k) = minQueTopK.top().m_iIndex;
            matTopDist(q, k) = minQueTopK.top().m_fValue;

            minQueTopK.pop();
        }

        // distTime_thr[tid] += omp_get_wtime() - t0;
        distTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        // threadWall_thr[tid] += omp_get_wtime() - region_start; // this thread's wall time inside region
    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startQueryTime);
    // double wall_elapsed = omp_get_wtime() - wall_start;

    if (verbose)
    {
        cout << "Test " << endl;
        cout << "Projecting and extracting top-vectors time: (ms) " << projTime << endl;
        cout << "Estimating time (ms): " << estTime << endl;
        cout << "Extracting candidates time (ms): " << candTime << endl;
        cout << "Computing distance time (ms): " << distTime << endl;
        cout << "Querying time  (ms): " << (float)durTime.count() << endl;

        // double projTime_sum =
        //     std::accumulate(projTime_thr.begin(), projTime_thr.end(), 0.0);
        // double estTime_sum =
        //         std::accumulate(estTime_thr.begin(), estTime_thr.end(), 0.0);
        // double candTime_sum =
        //         std::accumulate(candTime_thr.begin(), candTime_thr.end(), 0.0);
        // double distTime_sum =
        //         std::accumulate(distTime_thr.begin(), distTime_thr.end(), 0.0);
        // double threadWall_sum =
        //         std::accumulate(threadWall_thr.begin(), threadWall_thr.end(), 0.0);

        // cout << "Projecting and extracting top-vectors time: " << projTime_sum << " s" << endl;
        // cout << "Estimating time: " << estTime_sum << " s" << endl;
        // cout << "Extracting candidates time: " << candTime_sum << " s" << endl;
        // cout << "Computing distance time: " << distTime_sum << " s" << endl;
        // cout << "Total Region time: " << threadWall_sum << " s" << endl;
        //
        // cout << "Wall-elapsed (parallel) time: " << wall_elapsed << " s" << endl;

        // string sFileName = "CEOs_Est_" + int2str(n_neighbors) +
        //                    "_numProj_" + int2str(CEOs::n_proj) +
        //                    "_numRepeat_" + int2str(CEOs::n_repeats) +
        //                    "_topProj_" + int2str(CEOs::n_probed_vectors) +
        //                    "_cand_" + int2str(n_cand) + ".txt";
        //
        //
        // outputFile(matTopK, sFileName);
    }

    return make_tuple(matTopK, matTopDist);
}

/**
 * Build index of coCEOs-Est (1 layer) for estimating inner product
 * For each random vector, we only store the top-m closest/furthest points to this vector
 * So the index size is O(m * D * n_repeats), which is much smaller than CEOs where we need to store projection values of all points to all random vectors
 * We need 2 * D * n_repeats buckets, where the [0, D) is for positive projection values (closest) and [D, 2D) is for negative projections values (furthest)
 *
 * Data structure:
 * - vector<vector<IFPair>> vec2D_Pair_Buckets: each bucket contains a vector of (index, projection value) pairs (i.e. top-m pairs)
 * - bucketIdx ranges from [0, 2 * D * n_repeats)
 *
 * Algorithm:
 * - For each repeat, we parallel on the point Xi
 * - For each point Xi, we execute n_repeats times of FHT, and for each random vector, we insert the (index, projection value) pair into the corresponding bucket if
 *  the bucket has less than m pairs, or the projection value is larger than the minimum value in the bucket
 *
 * - For each repeat, we maintain 2 * D priority queues of size m (vectorMinQue_TopM) to store the top-m pairs for each bucket since there are 2D random vectors
 * - We use locks to avoid multiple threads writing to the same bucket at the same time
 * - After processing all points per each repeat, we dequeue and store the top-m on vec2D_Pair_Buckets
 *
 * @param matX
 */
void CEOs::build_coCEOs_Est1(const Ref<const RowMajorMatrixXf> &matX)
{
    cout << "Building coCEOs-Est1 index..." << endl;

    cout << "n_points: " << CEOs::n_points << endl;
    cout << "n_features: " << CEOs::n_features << endl;
    cout << "n_repeats: " << CEOs::n_repeats << endl;
    cout << "n_proj: " << CEOs::n_proj << endl;
    cout << "top_m: " << CEOs::top_m << endl;
    cout << "centering: " << CEOs::centering << endl;
    cout << "fhtDim: " << CEOs::fhtDim << endl;

    auto start = chrono::high_resolution_clock::now();

    omp_set_num_threads(CEOs::n_threads);

    CEOs::matrix_X = matX;

    // Compute the data center
    CEOs::vecCenter = CEOs::matrix_X.colwise().mean();


    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Copying data time (s): " << (float)duration.count() / 1000 << endl;

    int numBucketsPerRepeat = 2 * CEOs::n_proj;
    CEOs::vec2D_Pair_Buckets = vector<vector<IFPair>> (numBucketsPerRepeat * CEOs::n_repeats);

    // Need to store it for the query phase
    bitHD3Generator(CEOs::fhtDim * CEOs::n_rotate * CEOs::n_repeats, CEOs::seed, CEOs::bitHD1);

    int log2_FHT = log2(CEOs::fhtDim);
    float extractTopPointsTime = 0.0, projTime = 0.0;

    // Note: If NUM_LOCKS is large, we might not have enough stack memory if using array
    // if D = 128 = 2^7, then numBuckets = 2^16 = 65536. We aim at 256 KB memory for locks
    // 16K locks is good for million-point data set though it is not good for small data sets.
    constexpr size_t NUM_LOCKS = 16384;
    vector<omp_lock_t> locks(NUM_LOCKS); // NUM_LOCK = 16K locks = only 256 KB

    // Initialize locks since multi-thread can write to the same bucket at the same time
    // https://stackoverflow.com/questions/15175198/openmp-lock-array-initialization
#pragma omp parallel for
    for (size_t i = 0; i < NUM_LOCKS; i++) {
        omp_init_lock(&locks[i]);
    }

    for (int repeat = 0; repeat < CEOs::n_repeats; ++repeat)
    {
        int bucketBase = repeat * numBucketsPerRepeat;
        vector<priority_queue< IFPair, vector<IFPair>, greater<> >> vectorMinQue_TopM(numBucketsPerRepeat);

#pragma omp parallel for reduction(+:projTime)
        for (int n = 0; n < CEOs::n_points; ++n)
        {
            auto startTime = chrono::high_resolution_clock::now();
            VectorXf tempX = VectorXf::Zero(CEOs::fhtDim);

            if (CEOs::centering)
                tempX.segment(0, CEOs::n_features) = CEOs::matrix_X.row(n) - CEOs::vecCenter;
            else
                tempX.segment(0, CEOs::n_features) = CEOs::matrix_X.row(n);

            // Random projection and insert into a priority queue for each random vectors
            VectorXf rotatedX = tempX;
            int rotateBase = CEOs::fhtDim * CEOs::n_rotate * repeat;

            for (int rotate = 0; rotate < CEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedX(d) *= (2 * static_cast<float>(CEOs::bitHD1[rotateBase + rotate * CEOs::fhtDim + d]) - 1);
                }

                fht_float(rotatedX.data(), log2_FHT);
            }

            for (int r = 0; r < CEOs::n_proj; ++r)
            {
                int iSign = sgn(rotatedX(r));
                float fAbsHashValue = iSign * rotatedX(r);

                int Ri_2D = r; // index of random vector in [2D] after consider the sign
                if (iSign < 0)
                    // iBucketIndex |= 1UL << log2Project; // set bit at position log2(D)
                        Ri_2D += CEOs::n_proj; // Be aware the case that n_proj is not 2^(log2Proj)

                omp_set_lock(&locks[Ri_2D % NUM_LOCKS]);

                if ((int)vectorMinQue_TopM[Ri_2D].size() < CEOs::top_m)
                    vectorMinQue_TopM[Ri_2D].emplace(n, fAbsHashValue);

                else if (fAbsHashValue > vectorMinQue_TopM[Ri_2D].top().m_fValue)
                {
                    vectorMinQue_TopM[Ri_2D].pop();
                    vectorMinQue_TopM[Ri_2D].emplace(n, fAbsHashValue);
                }

                omp_unset_lock(&locks[Ri_2D % NUM_LOCKS]);
            } // End for each random vector

            projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        } // End for each point

        // For each bucket, we deque the top-m points into vec2D_Pair_Buckets
#pragma omp parallel for reduction(+: extractTopPointsTime)
        for (size_t b = 0; b < vectorMinQue_TopM.size(); ++b)
        {
            // b in range [0, numBucketsPerRepeat)
            auto startTime = chrono::high_resolution_clock::now();

            int m = (int)vectorMinQue_TopM[b].size();
            int new_bucketIdx = bucketBase + b; // Change to global bucket index

            CEOs::vec2D_Pair_Buckets[new_bucketIdx] = vector<IFPair>(m);

            while (!vectorMinQue_TopM[b].empty())
            {
                // Be aware of the index shift for different repeat
                CEOs::vec2D_Pair_Buckets[new_bucketIdx][m-1] = vectorMinQue_TopM[b].top();
                vectorMinQue_TopM[b].pop();
                m--;
            }

            extractTopPointsTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
        }
    } // End for each repeat

    double dSize = 1.0 * CEOs::matrix_X.rows() * CEOs::matrix_X.cols() * sizeof(float) / (1 << 30);
    cout << "Size of data set in GB: " << dSize << endl;

    for (size_t b = 0; b < CEOs::vec2D_Pair_Buckets.size(); ++b)
        dSize += 1.0 * CEOs::vec2D_Pair_Buckets[b].size() * sizeof(IFPair) / (1 << 30);

    cout << "Size of coCEOs-Est index (including X) in GB: " << dSize << endl;

#pragma omp parallel for
    for (size_t i = 0; i < NUM_LOCKS; ++i) {
        omp_destroy_lock(&locks[i]);
    }

    duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Projecting time (s): " << projTime / 1000 << endl;
    cout << "Extracting top-m points time (s): " << extractTopPointsTime / 1000 << endl;
    cout << "coCEos-Est1 constructing time (s): " << (float)duration.count() / 1000 << endl;
}

/**
 * Search coCEOs-Est1 index for estimating inner product
 *
 * Algorithm:
 * - We parallel on the query Qi
 * - For each Qi, we execute n_repeats times of FHT, getting its projection values to all n_repeats * D random vectors, and selecting top-n_probed_vectors closest/furthest vectors
 * - We aggregate the projections of the top-m pairs of these selected vectors to estimate the inner product using stl::robin_map
 * - There is a re-ranking process with n_cand where we extract top-n_cand points with largest estimated inner product, and compute the exact inner product with these n_cand points
 *
 * @param matQ
 * @param n_neighbors
 * @param verbose
 * @return
 */
tuple<RowMajorMatrixXi, RowMajorMatrixXf> CEOs::search_coCEOs_Est1(const Ref<const RowMajorMatrixXf> & matQ, int n_neighbors, bool verbose)
{
    if (CEOs::n_probed_points > CEOs::top_m)
    {
        cerr << "Error: Number of probed points must be smaller than number of indexed top-m points !" << endl;
        exit(1);
    }
    if (CEOs::n_probed_vectors > CEOs::n_proj * CEOs::n_repeats)
    {
        cerr << "Error: Number of probed vectors must be smaller than n_proj * n_repeats !" << endl;
        exit(1);
    }

    int n_queries = matQ.rows();
    if (verbose)
    {
        cout << "n_probed_vectors: " << CEOs::n_probed_vectors << endl;
        cout << "n_probed_points: " << CEOs::n_probed_points << endl;
        cout << "n_cand: " << CEOs::n_cand << endl;
        cout << "n_threads: " << CEOs::n_threads << endl;

        cout << "n_queries: " << n_queries << endl;
    }

    auto startQueryTime = chrono::high_resolution_clock::now();

    float projTime = 0.0, estTime = 0.0, distTime = 0.0, candTime = 0.0;

    RowMajorMatrixXi matTopK = RowMajorMatrixXi::Zero(n_queries, n_neighbors);
    RowMajorMatrixXf matTopDist = RowMajorMatrixXf::Zero(n_queries, n_neighbors);

    int log2_FHT = log2(CEOs::fhtDim);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(CEOs::n_threads);

#pragma omp parallel for reduction(+:projTime, estTime, candTime, distTime)
    for (int q = 0; q < n_queries; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.row(q);
        priority_queue< IFPair, vector<IFPair>, greater<> > minQue; // store the top-s closest/furthest vectors across all repetitions

        // For each repeat
        for (int repeat = 0; repeat < CEOs::n_repeats; ++repeat)
        {
            int bucketBase = repeat * CEOs::n_proj * 2;

            int baseIdx = CEOs::fhtDim * CEOs::n_rotate * repeat;
            VectorXf rotatedQ = VectorXf::Zero(CEOs::fhtDim);

            rotatedQ.segment(0, CEOs::n_features) = vecQuery;
            // Note: be careful on centering query since it completely changes to X-c and q-c
            // if (CEOs::centering)
            //     rotatedQ.segment(0, CEOs::n_features) = vecQuery - CEOs::vecCenter;

            for (int rotate = 0; rotate < CEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedQ(d) *= (2 * static_cast<float>(CEOs::bitHD1[baseIdx + rotate * CEOs::fhtDim + d]) - 1);
                }
                fht_float(rotatedQ.data(), log2_FHT);
            }

            // Note for segment(i, size) where i is starting index, size is segment size
            for (int d = 0; d < CEOs::n_proj; ++d)
            {
                int iBucketIdx = bucketBase + d; // in case positive projected value
                float fAbsProjValue = rotatedQ(d);

                if (fAbsProjValue < 0) {
                    fAbsProjValue = -fAbsProjValue; // get abs
                    iBucketIdx += CEOs::n_proj;
                }

                if ((int)minQue.size() < CEOs::n_probed_vectors)
                    minQue.emplace(iBucketIdx, fAbsProjValue);
                else if (fAbsProjValue > minQue.top().m_fValue) {
                    minQue.pop(); // pop max, and push min hash distance
                    minQue.emplace(iBucketIdx, fAbsProjValue); // Hack:
                }
            } // End for each random vector
        } // End for each repeat

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        startTime = chrono::high_resolution_clock::now();
        tsl::robin_map<int, float> mapEst;
        mapEst.reserve(CEOs::n_probed_vectors * CEOs::n_probed_points);

        while (!minQue.empty())
        {
            int iBucketIdx = minQue.top().m_iIndex;
            minQue.pop();

            vector<IFPair> bucket = CEOs::vec2D_Pair_Buckets[iBucketIdx];
            for (size_t i = 0; i < bucket.size(); ++i) // bucket.size() = top-m
            {
                int iPointIdx = bucket[i].m_iIndex;
                float fValue = bucket[i].m_fValue;

                if (mapEst.find(iPointIdx) == mapEst.end())
                    mapEst[iPointIdx] = fValue;
                else
                    mapEst[iPointIdx] += fValue;
            }
        } // End for each probed bucket

        estTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        // Note: We will use minQueHash again to store the candidate points as it is already empty
        startTime = chrono::high_resolution_clock::now();
        for (auto& it: mapEst)
        {
            float avgEst = it.second;

            if ((int)minQue.size() < CEOs::n_cand)
                minQue.emplace(it.first, avgEst); // use average value for estimation

            // queue is full
            else if (avgEst > minQue.top().m_fValue)
            {
                minQue.pop(); // pop max, and push min hash distance
                minQue.emplace(it.first, avgEst); // use average value for estimation
            }
        }

        assert(minQue.size() == CEOs::n_cand);

        candTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        startTime = chrono::high_resolution_clock::now();
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;
        for (int i = 0; i < CEOs::n_cand; ++i)
        {
            IFPair ifPair = minQue.top();
            minQue.pop();
            int iPointIdx = ifPair.m_iIndex;

            float fInnerProduct = vecQuery.dot(CEOs::matrix_X.row(iPointIdx));

            // Add into priority queue
            if (int(minQueTopK.size()) < n_neighbors)
                minQueTopK.emplace(iPointIdx, fInnerProduct);

            else if (fInnerProduct > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.emplace(iPointIdx, fInnerProduct);
            }
        }

        // There is the case that we get all 0 index if we do not have enough Top-K
        for (int k = (int)minQueTopK.size() - 1; k >= 0; --k)
        {
            matTopK(q, k) = minQueTopK.top().m_iIndex;
            matTopDist(q, k) = minQueTopK.top().m_fValue;

            minQueTopK.pop();
        }

        distTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startQueryTime);


    if (verbose)
    {
        cout << "Projecting and extracting top-vectors time (ms): " << projTime << endl;
        cout << "Estimating time (ms): " << estTime << endl;
        cout << "Extracting candidates time (ms): " << candTime << endl;
        cout << "Computing distance time (ms): " << distTime << endl;
        cout << "Querying time: " << (float)durTime.count() << endl;

        // string sFileName = "coCEOs_Est_" + int2str(n_neighbors) +
        //                    "_numProj_" + int2str(CEOs::n_proj) +
        //                    "_numRepeat_" + int2str(CEOs::n_repeats) +
        //                    "_topProj_" + int2str(CEOs::n_probed_vectors) +
        //                    "_topPoints_" + int2str(CEOs::n_probed_points) +
        //                    "_cand_" + int2str(n_cand) + ".txt";
        //
        //
        // outputFile(matTopK, sFileName);
    }

    return make_tuple(matTopK, matTopDist);
}

/**
 * Build index of CEOs-Hash (1 layer) for estimating inner product
 * For each random vector, we only store the top-m points to this vector
 * So the index size is O(m * D * n_repeats), which is much smaller than CEOs where we need to store projection values of all points to all random vectors
 * We need 2 * D * n_repeats buckets, where the [0, D) is for positive projection values (closest) and [D, 2D) is for negative projections values (furthest)
 *
 * Data structure:
 * - vector<vector<int>> vec2D_Buckets: each bucket contains a vector of pointIdx (i.e. top-m pairs)
 * - bucketIdx ranges from [0, 2 * D * n_repeats)
 *
 * Algorithm:
 * - For each repeat, we parallel on the point Xi
 * - For each point Xi, we execute n_repeats times of FHT, and for each random vector, we insert the (index, projection value) pair into the corresponding bucket if
 *  the bucket has less than m pairs, or the projection value is larger than the minimum value in the bucket
 *
 * - For each repeat, we maintain local 2 * D priority queues of size m (vectorMinQue_TopM) to store the top-m pairs for each bucket since there are 2D random vectors
 * - We use locks to avoid multiple threads writing to the same bucket at the same time
 * - After processing all points per each repeat, we update global index by dequeuing and storing the top-m pointIdx on vec2D_Buckets
 *
 * - The process is nearly the same as build_coCEOs_Est1 except that we only store pointIdx instead of (pointIdx, projection value) pair
 *
 * @param matX
 */
void CEOs::build_CEOs_Hash1(const Ref<const RowMajorMatrixXf> &matX)
{
    cout << "Building CEOs-Hash index..." << endl;

    cout << "n_points: " << CEOs::n_points << endl;
    cout << "n_features: " << CEOs::n_features << endl;
    cout << "n_repeats: " << CEOs::n_repeats << endl;
    cout << "n_proj: " << CEOs::n_proj << endl;
    cout << "top_m: " << CEOs::top_m << endl;
    cout << "centering: " << CEOs::centering << endl;
    cout << "fhtDim: " << CEOs::fhtDim << endl;

    auto start = chrono::high_resolution_clock::now();

    omp_set_num_threads(CEOs::n_threads);

    CEOs::matrix_X = matX;

    // Compute the data center
    CEOs::vecCenter = CEOs::matrix_X.colwise().mean();

    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Copying data time (s): " << (float)duration.count() / 1000 << endl;

    // We need 2 * top-points position for (index, value)
    int numBucketsPerRepeat = 2 * CEOs::n_proj;
    CEOs::vec2D_Buckets = vector<vector<int>> (numBucketsPerRepeat * CEOs::n_repeats);


    // Need to store it for the query phase
    bitHD3Generator(CEOs::fhtDim * CEOs::n_rotate * CEOs::n_repeats, CEOs::seed, CEOs::bitHD1);

    int log2_FHT = log2(CEOs::fhtDim);
    float extractTopPointsTime = 0.0, projTime = 0.0;

    // Note: If NUM_LOCKS is large, we might not have enough stack memory if using array
    // if D = 128 = 2^7, then numBuckets = 2^16 = 65536. We aim at 256 KB memory for locks
    // 16K locks is good for million-point data set though it is not good for small data sets.
    constexpr size_t NUM_LOCKS = 16384;
    vector<omp_lock_t> locks(NUM_LOCKS); // NUM_LOCK = 16K locks = only 256 KB

    // Initialize locks since multi-thread can write to the same bucket at the same time
    // https://stackoverflow.com/questions/15175198/openmp-lock-array-initialization
#pragma omp parallel for
    for (size_t i = 0; i < NUM_LOCKS; i++) {
        omp_init_lock(&locks[i]);
    }

    // For each repeat, we compute the local index of (2D) buckets (by parallel on points).
    // After that, we update the global index of (2D) * n_repeats buckets (by parallel on local buckets)
    for (int repeat = 0; repeat < CEOs::n_repeats; ++repeat) {
        int bucketBase = repeat * numBucketsPerRepeat;
        vector<priority_queue< IFPair, vector<IFPair>, greater<> >> vectorMinQue_TopM(numBucketsPerRepeat);

#pragma omp parallel for reduction(+:projTime)
        for (int n = 0; n < CEOs::n_points; ++n)
        {
            auto startTime = chrono::high_resolution_clock::now();
            VectorXf tempX = VectorXf::Zero(CEOs::fhtDim);

            if (CEOs::centering)
                tempX.segment(0, CEOs::n_features) = CEOs::matrix_X.row(n) - CEOs::vecCenter;
            else
                tempX.segment(0, CEOs::n_features) = CEOs::matrix_X.row(n);

            // Random projection and insert into a priority queue for each random vectors
            VectorXf rotatedX = tempX;
            int rotateBase = CEOs::fhtDim * CEOs::n_rotate * repeat;

            for (int rotate = 0; rotate < CEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedX(d) *= (2 * static_cast<float>(CEOs::bitHD1[rotateBase + rotate * CEOs::fhtDim + d]) - 1);
                }

                fht_float(rotatedX.data(), log2_FHT);
            }

            for (int r = 0; r < CEOs::n_proj; ++r)
            {

                int iSign = sgn(rotatedX(r));
                float fAbsHashValue = iSign * rotatedX(r);

                int Ri_2D = r; // index of random vector in [2D] after consider the sign
                if (iSign < 0)
                    // iBucketIndex |= 1UL << log2Project; // set bit at position log2(D)
                        Ri_2D += CEOs::n_proj; // Be aware the case that n_proj is not 2^(log2Proj)

                omp_set_lock(&locks[Ri_2D % NUM_LOCKS]);

                if ((int)vectorMinQue_TopM[Ri_2D].size() < CEOs::top_m)
                    vectorMinQue_TopM[Ri_2D].emplace(n, fAbsHashValue);

                else if (fAbsHashValue > vectorMinQue_TopM[Ri_2D].top().m_fValue)
                {
                    vectorMinQue_TopM[Ri_2D].pop();
                    vectorMinQue_TopM[Ri_2D].emplace(n, fAbsHashValue);
                }

                omp_unset_lock(&locks[Ri_2D % NUM_LOCKS]);
            } // End for each random vector

            projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        } // End for each point

        // Update global data structure
#pragma omp parallel for reduction(+: extractTopPointsTime)
        for (size_t b = 0; b < vectorMinQue_TopM.size(); ++b)
        {
            // b in range [0, numBucketsPerRepeat * n_repeats)
            auto startTime = chrono::high_resolution_clock::now();

            int m = (int)vectorMinQue_TopM[b].size();
            int new_bucketIdx = bucketBase + b;

            CEOs::vec2D_Buckets[new_bucketIdx] = vector<int>(m);

            while (!vectorMinQue_TopM[b].empty())
            {
                // Be aware of the index shift for different repeat
                CEOs::vec2D_Buckets[new_bucketIdx][m-1] = vectorMinQue_TopM[b].top().m_iIndex;
                vectorMinQue_TopM[b].pop();
                m--;
            }

            extractTopPointsTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
        }
    } // End for each repeat

    double dSize = 1.0 * CEOs::matrix_X.rows() * CEOs::matrix_X.cols() * sizeof(float) / (1 << 30);
    cout << "Size of data set in GB: " << dSize << endl;

    for (size_t b = 0; b < CEOs::vec2D_Buckets.size(); ++b)
        dSize += 1.0 * CEOs::vec2D_Buckets[b].size() * sizeof(int) / (1 << 30);

    cout << "Size of coCEOs-Est index (including data) in GB: " << dSize << endl;

#pragma omp parallel for
    for (size_t i = 0; i < NUM_LOCKS; ++i) {
        omp_destroy_lock(&locks[i]);
    }

    duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Projecting time (s): " << projTime / 1000 << endl;
    cout << "Extracting top-m points time (s): " << extractTopPointsTime / 1000 << endl;
    cout << "CEOs-Hash1 constructing time (s): " << (float)duration.count() / 1000 << endl;
}

/**
 * Search CEOs-Hash (1 layer) for estimating inner product
 *
 * Algorithm:
 * - We parallel on the query Qi
 * - For each Qi, we execute n_repeats times of FHT, getting its projection values to all n_repeats * D random vectors, and selecting top-n_probed_vectors closest/furthest vectors
 * - We always compute the inner product between Qi with all top-m points associated with these selected vectors, and return top-K points with largest inner product
 * - Compared to coCEOs-Est1, we do not have the estimation phase using stl::robin_map, and we do not have the re-ranking phase using n_cand points
 * - Number of distance = n_probed_vectors * top-m
 *
 * @param matQ
 * @param n_neighbors
 * @param verbose
 * @return
 */
tuple<RowMajorMatrixXi, RowMajorMatrixXf> CEOs::search_CEOs_Hash1(const Ref<const RowMajorMatrixXf> & matQ, int n_neighbors, bool verbose)
{
    if (CEOs::n_probed_points > CEOs::top_m)
    {
        cerr << "Error: Number of probed points must be smaller than number of indexed top-m points !" << endl;
        exit(1);
    }
    if (CEOs::n_probed_vectors > CEOs::n_proj * CEOs::n_repeats)
    {
        cerr << "Error: Number of probed vectors must be smaller than n_proj * n_repeats !" << endl;
        exit(1);
    }

    int n_queries = matQ.rows();
    if (verbose)
    {
        cout << "n_probedVectors: " << CEOs::n_probed_vectors << endl;
        cout << "n_probedPoints: " << CEOs::n_probed_points << endl;
        cout << "n_cand: " << CEOs::n_probed_vectors *  CEOs::n_probed_points << endl;
        cout << "n_threads: " << CEOs::n_threads << endl;

        cout << "n_queries: " << n_queries << endl;
    }

    auto startQueryTime = chrono::high_resolution_clock::now();

    float projTime = 0.0, distTime = 0.0;

    RowMajorMatrixXi matTopK = RowMajorMatrixXi::Zero(n_queries, n_neighbors);
    RowMajorMatrixXf matTopDist = RowMajorMatrixXf::Zero(n_queries, n_neighbors);

    int log2_FHT = log2(CEOs::fhtDim);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(CEOs::n_threads);

#pragma omp parallel for reduction(+:projTime, distTime)
    for (int q = 0; q < n_queries; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.row(q);
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueHash;

        // For each repeat
        for (int repeat = 0; repeat < CEOs::n_repeats; ++repeat)
        {
            int bucketBase = repeat * CEOs::n_proj * 2;

            int baseIdx = CEOs::fhtDim * CEOs::n_rotate * repeat;
            VectorXf rotatedQ = VectorXf::Zero(CEOs::fhtDim);

            rotatedQ.segment(0, CEOs::n_features) = vecQuery;
            // Note: be careful on centering query since it completely changes to X-c and q-c
            // if (CEOs::centering)
            //     rotatedQ.segment(0, CEOs::n_features) = vecQuery - CEOs::vecCenter;

            for (int rotate = 0; rotate < CEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedQ(d) *= (2 * static_cast<float>(CEOs::bitHD1[baseIdx + rotate * CEOs::fhtDim + d]) - 1);
                }
                fht_float(rotatedQ.data(), log2_FHT);
            }

            // Note for segment(i, size) where i is starting index, size is segment size
            for (int d = 0; d < CEOs::n_proj; ++d)
            {
                int iBucketIdx = d; // in case positive projected value
                float fAbsProjValue = rotatedQ(d);

                if (fAbsProjValue < 0) {
                    fAbsProjValue = -fAbsProjValue; // get abs
                    iBucketIdx += CEOs::n_proj;
                }

                iBucketIdx += bucketBase;

                if ((int)minQueHash.size() < CEOs::n_probed_vectors)
                    minQueHash.emplace(iBucketIdx, fAbsProjValue);
                else if (fAbsProjValue > minQueHash.top().m_fValue) {
                    minQueHash.pop(); // pop max, and push min hash distance
                    minQueHash.emplace(iBucketIdx, fAbsProjValue); // Hack:
                }
            }
        } // End for each repeat

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        startTime = chrono::high_resolution_clock::now();
        boost::dynamic_bitset<> bitsetHist(CEOs::n_points);
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;

        while (!minQueHash.empty())
        {
            int bucketIdx = minQueHash.top().m_iIndex;
            minQueHash.pop();

            // Sample one random point from the bucket Ri
            for (const int pointIdx: CEOs::vec2D_Buckets[bucketIdx]) {
                if (~bitsetHist[pointIdx]) // do not put the query point itself
                {
                    bitsetHist[pointIdx] = true;
                    float fInnerProduct = vecQuery.dot(CEOs::matrix_X.row(pointIdx));

                    if ((int)minQueTopK.size() < n_neighbors)
                        minQueTopK.emplace(pointIdx, fInnerProduct); // emplace is push without creating temp data
                    else if (fInnerProduct > minQueTopK.top().m_fValue)
                    {
                        minQueTopK.pop();
                        minQueTopK.emplace(pointIdx, fInnerProduct); // No need IFPair()
                    }
                }
            }
        } // End for each probed bucket

        // There is the case that we get all 0 index if we do not have enough Top-K
        for (int k = (int)minQueTopK.size() - 1; k >= 0; --k)
        {
            matTopK(q, k) = minQueTopK.top().m_iIndex;
            matTopDist(q, k) = minQueTopK.top().m_fValue;

            minQueTopK.pop();
        }

        distTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startQueryTime);


    if (verbose)
    {
        cout << "Projecting and extracting top-vectors time (ms): " << projTime << endl;
        cout << "Computing distance time (ms): " << distTime << endl;
        cout << "Querying time (ms): " << (float)durTime.count() << endl;

        // string sFileName = "coCEOs_Est_" + int2str(n_neighbors) +
        //                    "_numProj_" + int2str(CEOs::n_proj) +
        //                    "_numRepeat_" + int2str(CEOs::n_repeats) +
        //                    "_topProj_" + int2str(CEOs::n_probed_vectors) +
        //                    "_topPoints_" + int2str(CEOs::n_probed_points) +
        //                    "_cand_" + int2str(n_cand) + ".txt";
        //
        //
        // outputFile(matTopK, sFileName);
    }

    return make_tuple(matTopK, matTopDist);
}

/**
 * Build index of coCEOs-Estimate (2 layers) for estimating inner product
 * For each repeat, we use two set S, R random vectors, and for each set, we consider top-iProbe vectors closest to the point.
 * For each point Xi, we insert it into top-iProbe^2 buckets, corresponding to these top-iProbe vectors from S and R
 *
 * Data structure:
 * - vector<vector<IFPair>> vec2D_Pair_Buckets: each bucket contains a vector of (pointIdx, projection value) pair (i.e. upper bounded by top-m)
 * - bucketIdx ranges from [0, 4 * D^2 * n_repeats) since we use 2D random vectors for each layer.
 * - Note: Space complexity is O(m * 4 * D^2 * n_repeats) since we have 4D^2 random vectors
 *
 * Algorithm:
 * - We parallel on the point Xi
 * - For each point Xi, we execute n_repeats times of FHT twice for R and S, and for each set, extracting top-iProbe closest random vectors.
 * - Then we insert Xi into these top-iProbe^2 buckets, if the bucket has less than m pairs, or the projection value is larger than the minimum value in the bucket
 *
 * - For each repeat, we maintain 4D^2 priority queues of size m (vectorMinQue_TopM) to store the top-m pairs for each bucket since there are 4D^2 pairs of random vectors
 * - We consider [0, D) for positive projection values (closest) and [D, 2D) for negative projections values (furthest) for each set
 *
 * - Note: Falconn++ only insert a point into iProbe buckets, but we insert into iProbe^2 buckets.
 * - We can support this feature in future (by using an additional minQueue or scale iProbe to sqrt{iProbe})
 *
 * - We use locks to avoid multiple threads writing to the same bucket at the same time
 * - After processing all points per each repeat, we dequeue and store the top-m (pointIdx, projection value) pairs on vec2D_Pair_Buckets
 *
 * - The process is nearly the same as build_coCEOs_Est1 except that we have two times of FHT and projection for each point
 *
 * @param matX
 */
void CEOs::build_coCEOs_Est2(const Ref<const RowMajorMatrixXf> &matX)
{
    cout << "Building coCEOs-Estimate index (2 layers)..." << endl;

    cout << "n_points: " << CEOs::n_points << endl;
    cout << "n_features: " << CEOs::n_features << endl;
    cout << "n_repeats: " << CEOs::n_repeats << endl;
    cout << "n_proj: " << CEOs::n_proj << endl;
    cout << "iProbe: " << CEOs::iProbe << endl;
    cout << "top_m: " << CEOs::top_m << endl;
    cout << "centering: " << CEOs::centering << endl;
    cout << "fhtDim: " << CEOs::fhtDim << endl;

    auto start = chrono::high_resolution_clock::now();

    omp_set_num_threads(CEOs::n_threads);

    CEOs::matrix_X = matX;

    // Compute the data center
    CEOs::vecCenter = CEOs::matrix_X.colwise().mean();

    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Copying data time (s): " << (float)duration.count() / 1000 << endl;

    /** Global parameter **/
    int numBucketsPerRepeat = 4 * CEOs::n_proj * CEOs::n_proj;
    int num2D = 2 * CEOs::n_proj;
    int log2_FHT = log2(CEOs::fhtDim);

    CEOs::vec2D_Pair_Buckets = vector<vector<IFPair>> (numBucketsPerRepeat * CEOs::n_repeats);

    // Need to store it for the query phase
    bitHD3Generator2(CEOs::fhtDim * CEOs::n_rotate * CEOs::n_repeats, CEOs::seed, CEOs::bitHD1, CEOs::bitHD2);

    // Note: If NUM_LOCKS is large, we might not have enough stack memory if using array
    // if D = 128 = 2^7, then numBuckets = 2^16 = 65536. We aim at 256 KB memory for locks
    // 16K locks is good for million-point data set though it is not good for small data sets.
    constexpr size_t NUM_LOCKS = 16384;
    vector<omp_lock_t> locks(NUM_LOCKS); // NUM_LOCK = 16K locks = only 256 KB

    // Initialize locks
#pragma omp parallel for
    for (size_t i = 0; i < NUM_LOCKS; i++) {
        omp_init_lock(&locks[i]);
    }

    for (int repeat = 0; repeat < n_repeats; ++repeat)
    {
        int bucketBase = repeat * numBucketsPerRepeat;
        vector<priority_queue< IFPair, vector<IFPair>, greater<> >> vectorMinQue_TopM(numBucketsPerRepeat);

        /**
        Parallel for each the point Xi: (1) Compute and store dot product, and (2) Extract top-k close/far random vectors
        **/
#pragma omp parallel for
        for (int n = 0; n < CEOs::n_points; ++n)
        {
            VectorXf rotatedX1 = VectorXf::Zero(CEOs::fhtDim); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
            if (CEOs::centering)
                rotatedX1.segment(0, CEOs::n_features) = CEOs::matrix_X.row(n) - CEOs::vecCenter;
            else
                rotatedX1.segment(0, CEOs::n_features) = CEOs::matrix_X.row(n);

            VectorXf rotatedX2 = rotatedX1;

            int rotateBase = CEOs::fhtDim * CEOs::n_rotate * repeat;

            for (int rotate = 0; rotate < CEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedX1(d) *= (2 * static_cast<float>(CEOs::bitHD1[rotateBase + rotate * CEOs::fhtDim + d]) - 1);
                    rotatedX2(d) *= (2 * static_cast<float>(CEOs::bitHD2[rotateBase + rotate * CEOs::fhtDim + d]) - 1);
                }

                fht_float(rotatedX1.data(), log2_FHT);
                fht_float(rotatedX2.data(), log2_FHT);
            }

            // rotateX1 = {x1 * r1, ..., x1 * r_D}
            // rotateX2 = {x1 * s1, ..., x1 * s_D}

            // cout << "We finish random rotating" << endl;

            // This queue is used for finding top-k max hash values and hash index for iProbes on each layer
            // priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopI1, minQueTopI2;
            //
            // /**
            // We use a priority queue to keep top-max abs projection for each repeat
            // Always ensure fhtDim >= n_proj
            // **/
            // // We can speed up with n-th selection but have to deal with the case that we have negative values
            // for (int r = 0; r < CEOs::n_proj; ++r)
            // {
            //     // 1st rotation
            //     int iSign = sgn(rotatedX1(r));
            //     float fAbsHashValue = iSign * rotatedX1(r);
            //
            //     int Ri_2D = r; // index of random vector in [2D] after consider the sign
            //     if (iSign < 0)
            //         // iBucketIndex |= 1UL << log2Project; // set bit at position log2(D)
            //             Ri_2D += CEOs::n_proj; // Be aware the case that n_proj is not 2^(log2Proj)
            //
            //     // iProbe
            //     if ((int)minQueTopI1.size() < CEOs::iProbe)
            //         minQueTopI1.emplace(Ri_2D, fAbsHashValue); // emplace is push without creating temp data
            //     else if (fAbsHashValue > minQueTopI1.top().m_fValue)
            //     {
            //         minQueTopI1.pop();
            //         minQueTopI1.emplace(Ri_2D, fAbsHashValue); // No need IFPair()
            //     }
            //
            //     // 2nd rotation
            //     iSign = sgn(rotatedX2(r));
            //     fAbsHashValue = iSign * rotatedX2(r);
            //
            //     Ri_2D = r;
            //     if (iSign < 0)
            //         // iBucketIndex |= 1UL << log2Project; // set bit at position log2(D)
            //             Ri_2D += CEOs::n_proj; // set bit at position log2(D)
            //
            //     // iProbe (top-iProbe random vector closest to Xn)
            //     if ((int)minQueTopI2.size() < CEOs::iProbe)
            //         minQueTopI2.emplace(Ri_2D, fAbsHashValue);
            //     else if (fAbsHashValue > minQueTopI2.top().m_fValue)
            //     {
            //         minQueTopI2.pop();
            //         minQueTopI2.emplace(Ri_2D, fAbsHashValue);
            //     }
            // }
            //
            // // Convert to vector
            // vector<IFPair> vec_topI1(CEOs::iProbe), vec_topI2(CEOs::iProbe);
            //
            // // iProbe-Falconn++
            // for (int p = CEOs::iProbe - 1; p >= 0; --p)
            // {
            //     vec_topI1[p] = minQueTopI1.top();
            //     minQueTopI1.pop();
            //
            //     vec_topI2[p] = minQueTopI2.top();
            //     minQueTopI2.pop();
            // }
            //
            // // assert(vec_topI1.size() == FalconnLite::iProbe);
            // // assert(vec_topI2.size() == FalconnLite::iProbe);
            //
            // /**
            // Use minQue to find the top-qProbe over 2 layers via sum of 2 estimators
            // vec1 and vec2 are already sorted, and has length of sOptics::topK
            // Note: Heuristic: We consider top-k * top-k pairs for Top-K, and top-p * top-p pairs for Top-M
            // Note: We cannot check all combinations due to significant cost
            // **/
            //
            // for (const auto& ifPair1: vec_topI1) {
            //     int Ri_2D_1st = ifPair1.m_iIndex;
            //     float fAbsHashValue1 = ifPair1.m_fValue;
            //
            //     for (const auto& ifPair2: vec_topI2)
            //     {
            //         int R2_2D_2nd = ifPair2.m_iIndex;
            //         float fAbsSumHash = ifPair2.m_fValue + fAbsHashValue1; // sum of 2 estimators
            //
            //         //We have 2D * 2D buckets (i.e. random vectors)
            //         int iBucketIndex = Ri_2D_1st * num2D + R2_2D_2nd; // (totally we have 2D * 2D buckets)
            //
            //         // assert(iBucketIndex < vectorMinQue_TopM.size());
            //
            //         omp_set_lock(&locks[iBucketIndex % NUM_LOCKS]);
            //
            //         // Note: This implementation is used for controlling the size of dense bucket.
            //
            //         if ((int)vectorMinQue_TopM[iBucketIndex].size() < CEOs::top_m)
            //             vectorMinQue_TopM[iBucketIndex].emplace(n, fAbsSumHash);
            //
            //         else if (fAbsSumHash > vectorMinQue_TopM[iBucketIndex].top().m_fValue)
            //         {
            //             vectorMinQue_TopM[iBucketIndex].pop();
            //             vectorMinQue_TopM[iBucketIndex].emplace(n, fAbsSumHash);
            //         }
            //
            //         omp_unset_lock(&locks[iBucketIndex % NUM_LOCKS]);
            //     }
            // } // End for each pair of random vectors

            // Create a vector from Rotate1 and Rotate2 with size 2 * n_proj, and then find top-iProbe pairs using partial_sort
            VectorXf Y1(2 * CEOs::n_proj);
            Y1.head(CEOs::n_proj) = rotatedX1.segment(0, CEOs::n_proj);
            Y1.tail(CEOs::n_proj) = -rotatedX1.segment(0, CEOs::n_proj);
            vector<int> idx1(2 * CEOs::n_proj);
            iota(idx1.begin(), idx1.end(), 0);

            // Partial sort indices by corresponding Y value (descending)
            std::partial_sort(idx1.begin(), idx1.begin() + CEOs::iProbe, idx1.end(),[&](int i, int j) { return Y1[i] > Y1[j]; });

            VectorXf Y2(2 * CEOs::n_proj);
            Y2.head(CEOs::n_proj) = rotatedX2.segment(0, CEOs::n_proj);
            Y2.tail(CEOs::n_proj) = -rotatedX2.segment(0, CEOs::n_proj);
            vector<int> idx2(2 * CEOs::n_proj);
            iota(idx2.begin(), idx2.end(), 0);
            std::partial_sort(idx2.begin(), idx2.begin() + CEOs::iProbe, idx2.end(),[&](int i, int j) { return Y2[i] > Y2[j]; });

            for (int i = 0; i < CEOs::iProbe; ++i)
            {
                int Ri_2D_1st = idx1[i];
                float fAbsHashValue1 = Y1[Ri_2D_1st];

                for (int j = 0; j < CEOs::iProbe; ++j) {
                    int R2_2D_2nd = idx2[j];
                    float fAbsSumHash = Y2[R2_2D_2nd] + fAbsHashValue1; // sum of 2 estimators

                    // We have 2D * 2D buckets (i.e. random vectors)
                    int iBucketIndex = Ri_2D_1st * num2D + R2_2D_2nd; // (totally we have 2D * 2D buckets)

                    omp_set_lock(&locks[iBucketIndex % NUM_LOCKS]);

                    if ((int)vectorMinQue_TopM[iBucketIndex].size() < CEOs::top_m)
                        vectorMinQue_TopM[iBucketIndex].emplace(n, fAbsSumHash);

                    else if (fAbsSumHash > vectorMinQue_TopM[iBucketIndex].top().m_fValue)
                    {
                        vectorMinQue_TopM[iBucketIndex].pop();
                        vectorMinQue_TopM[iBucketIndex].emplace(n, fAbsSumHash);
                    }

                    omp_unset_lock(&locks[iBucketIndex % NUM_LOCKS]);
                }
            } // End for each pair of random vectors
        } // End for each point

    // Extract top-M for each bucketIdx - Falconn++
#pragma omp parallel for
        for (int b = 0; b < numBucketsPerRepeat; ++b)
        {
            if (!vectorMinQue_TopM[b].empty())
            {
                // bucket-idx shift for different repeat
                int new_bucketIdx = b + bucketBase;
                int m = (int)vectorMinQue_TopM[b].size();

                CEOs::vec2D_Pair_Buckets[new_bucketIdx] = vector<IFPair>(m);

                while (!vectorMinQue_TopM[b].empty())
                {
                    // Be aware of the index shift for different repeat
                    CEOs::vec2D_Pair_Buckets[new_bucketIdx][m-1] = vectorMinQue_TopM[b].top(); // add into the end of vector
                    vectorMinQue_TopM[b].pop();
                    m--;
                }
            }

        }
    } // End for each repeat

    float totalPoints = 0.0;
    for (size_t i = 0; i < CEOs::vec2D_Pair_Buckets.size(); ++i)
        totalPoints += CEOs::vec2D_Pair_Buckets[i].size();
    cout << "Total points in all buckets (in millions): " << totalPoints / 1000000 << endl;

    double total_bytes = 0.0;

    // Outer vector storage
    total_bytes += CEOs::vec2D_Pair_Buckets.capacity() * sizeof(vector<IFPair>);

    // Inner vector storage
    for (auto const& inner : CEOs::vec2D_Pair_Buckets) {
        total_bytes += inner.capacity() * sizeof(IFPair);
    }
    cout << "Size of coCEOs-Est2 (without data) index in GB: " << total_bytes / (1 << 30) << "\n";

    // Destroy locks for Falconn++
#pragma omp parallel for
    for (size_t i = 0; i < NUM_LOCKS; ++i) {
        omp_destroy_lock(&locks[i]);
    }

    // Note: Should not clear when running testcase
    // FalconnLite::matrix_X.resize(0, 0);
    // FalconnLite::matrix_R.resize(0, 0);

    duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "coCEOs-Est2 construction time (s): " << (float)duration.count() / 1000 << endl;
}

/**
 * Search coCEOs-Est (2 layers)
 *
 * Algorithm:
 * - We parallel on the query Qi
 * - For each Qi, we execute n_repeats times of FHT, getting its projection values to ALL n_repeats * D random vectors, and selecting top-n_probed_vectors closest/furthest vectors
 * - We aggregate the projections of the top-m pairs of these selected vectors to estimate the inner product using stl::robin_map
 * - There is a re-ranking process with n_cand where we extract top-n_cand points with largest estimated inner product, and compute the exact inner product with these n_cand points
 *
 * @param matQ
 * @param n_neighbors
 * @param verbose
 * @return
 */
tuple<RowMajorMatrixXi, RowMajorMatrixXf> CEOs::search_coCEOs_Est2(const Ref<const RowMajorMatrixXf> & matQ, int n_neighbors, bool verbose)
{
    if (CEOs::n_probed_points > CEOs::top_m)
    {
        cerr << "Error: Number of probed points must be smaller than number of indexed top-m points !" << endl;
        exit(1);
    }
    if (CEOs::n_probed_vectors > CEOs::n_proj * CEOs::n_repeats)
    {
        cerr << "Error: Number of probed vectors must be smaller than n_proj * n_repeats !" << endl;
        exit(1);
    }

    int n_queries = matQ.rows();
    if (verbose)
    {
        cout << "n_probed_vectors: " << CEOs::n_probed_vectors << endl;
        cout << "n_probed_points: " << CEOs::n_probed_points << endl;
        cout << "n_cand: " << CEOs::n_cand << endl;
        cout << "n_threads: " << CEOs::n_threads << endl;

        cout << "n_queries: " << n_queries << endl;
    }

    auto startQueryTime = chrono::high_resolution_clock::now();

    float projTime = 0.0, estTime = 0.0, distTime = 0.0, candTime = 0.0;

    RowMajorMatrixXi matTopK = RowMajorMatrixXi::Zero(n_queries, n_neighbors);
    RowMajorMatrixXf matTopDist = RowMajorMatrixXf::Zero(n_queries, n_neighbors);

    int numBucketsPerRepeat = 4 * CEOs::n_proj * CEOs::n_proj;
    int num2D = 2 * CEOs::n_proj;
    int log2_FHT = log2(CEOs::fhtDim);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(CEOs::n_threads);

    float avgDist = 0.0, avgProbes = 0.0;

#pragma omp parallel for reduction(+:projTime, estTime, candTime, distTime, avgDist, avgProbes)
    for (int q = 0; q < n_queries; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.row(q);
        priority_queue< IFPair, vector<IFPair>, greater<> > minQue; // Get top-s vectors over all repetitions

        // For each repeat
        for (int repeat = 0; repeat < CEOs::n_repeats; ++repeat)
        {
            int bucketBase = repeat * numBucketsPerRepeat;
            int rotateBase = CEOs::fhtDim * CEOs::n_rotate * repeat;

            VectorXf rotatedQ1 = VectorXf::Zero(CEOs::fhtDim);
            rotatedQ1.segment(0, CEOs::n_features) = vecQuery;
            VectorXf rotatedQ2 = rotatedQ1;

            // Note: be careful on centering query since it completely changes to X-c and q-c
            // if (CEOs::centering)
            //     rotatedQ.segment(0, CEOs::n_features) = vecQuery - CEOs::vecCenter;

            for (int rotate = 0; rotate < CEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedQ1(d) *= (2 * static_cast<float>(CEOs::bitHD1[rotateBase + rotate * CEOs::fhtDim + d]) - 1);
                    rotatedQ2(d) *= (2 * static_cast<float>(CEOs::bitHD2[rotateBase + rotate * CEOs::fhtDim + d]) - 1);
                }
                fht_float(rotatedQ1.data(), log2_FHT);
                fht_float(rotatedQ2.data(), log2_FHT);
            }

            // This queue is used for finding top-k max hash values and hash index for iProbes on each layer
            priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopQ1, minQueTopQ2;

            /**
            We use a priority queue to keep top-max abs projection for each repeat
            Always ensure fhtDim >= n_proj
            **/
            for (int r = 0; r < CEOs::n_proj; ++r)
            {
                // 1st rotation
                int iSign = sgn(rotatedQ1(r));
                float fAbsHashValue = iSign * rotatedQ1(r);

                int Ri_2D = r; // index of random vector in [2D] after consider the sign
                if (iSign < 0)
                    // iBucketIndex |= 1UL << log2Project; // set bit at position log2(D)
                        Ri_2D += CEOs::n_proj; // Be aware the case that n_proj is not 2^(log2Proj)

                // iProbe
                if ((int)minQueTopQ1.size() < CEOs::n_probed_vectors)
                    minQueTopQ1.emplace(Ri_2D, fAbsHashValue); // emplace is push without creating temp data
                else if (fAbsHashValue > minQueTopQ1.top().m_fValue)
                {
                    minQueTopQ1.pop();
                    minQueTopQ1.emplace(Ri_2D, fAbsHashValue); // No need IFPair()
                }

                // 2nd rotation
                iSign = sgn(rotatedQ2(r));
                fAbsHashValue = iSign * rotatedQ2(r);

                Ri_2D = r;
                if (iSign < 0)
                    // iBucketIndex |= 1UL << log2Project; // set bit at position log2(D)
                        Ri_2D += CEOs::n_proj; // set bit at position log2(D)

                // iProbe (top-iProbe random vector closest to Xn)
                if ((int)minQueTopQ2.size() < CEOs::n_probed_vectors)
                    minQueTopQ2.emplace(Ri_2D, fAbsHashValue);
                else if (fAbsHashValue > minQueTopQ2.top().m_fValue)
                {
                    minQueTopQ2.pop();
                    minQueTopQ2.emplace(Ri_2D, fAbsHashValue);
                }
            }

            // Convert to vector
            vector<IFPair> vec_topQ1(CEOs::n_probed_vectors), vec_topQ2(CEOs::n_probed_vectors);
            for (int p = CEOs::n_probed_vectors - 1; p >= 0; --p)
            {
                vec_topQ1[p] = minQueTopQ1.top();
                minQueTopQ1.pop();

                vec_topQ2[p] = minQueTopQ2.top();
                minQueTopQ2.pop();
            }


            /**
            Use minQue to find the top-qProbe over 2 layers via sum of 2 estimators
            vec1 and vec2 are already sorted, and has length of sOptics::topK
            Note: Heuristic: We consider top-k * top-k pairs for Top-K, and top-p * top-p pairs for Top-M
            Note: We cannot check all combinations due to significant cost
            **/

            for (const auto& ifPair1: vec_topQ1)
            {
                int Ri_2D_1st = ifPair1.m_iIndex;
                float fAbsHashValue1 = ifPair1.m_fValue;

                for (const auto& ifPair2: vec_topQ2)
                {
                    int R2_2D_2nd = ifPair2.m_iIndex;
                    float fAbsSumHash = ifPair2.m_fValue + fAbsHashValue1; // sum of 2 estimators

                    //We have 2D * 2D buckets (i.e. random vectors)
                    int iBucketIndex = Ri_2D_1st * num2D + R2_2D_2nd + bucketBase; // change to global bucket index since minQue considers all repetitions

                    // Note: This implementation is used for controlling the size of dense bucket.

                    if ((int)minQue.size() < CEOs::n_probed_vectors)
                        minQue.emplace(iBucketIndex, fAbsSumHash);

                    else if (fAbsSumHash > minQue.top().m_fValue)
                    {
                        minQue.pop();
                        minQue.emplace(iBucketIndex, fAbsSumHash);
                    }
                }
            } // End for each pair of random vectors

            // Create a vector from Rotate1 and Rotate2 with size 2 * n_proj, and then find top-iProbe pairs using partial_sort
            // VectorXf Y1(2 * CEOs::n_proj);
            // Y1.head(CEOs::n_proj) = rotatedQ1.segment(0, CEOs::n_proj);
            // Y1.tail(CEOs::n_proj) = -rotatedQ1.segment(0, CEOs::n_proj);
            // vector<int> idx1(2 * CEOs::n_proj);
            // iota(idx1.begin(), idx1.end(), 0);
            //
            // // Partial sort indices by corresponding Y value (descending)
            // std::partial_sort(idx1.begin(), idx1.begin() + CEOs::n_probed_vectors, idx1.end(),[&](int i, int j) { return Y1[i] > Y1[j]; });
            //
            // VectorXf Y2(2 * CEOs::n_proj);
            // Y2.head(CEOs::n_proj) = rotatedQ2.segment(0, CEOs::n_proj);
            // Y2.tail(CEOs::n_proj) = -rotatedQ2.segment(0, CEOs::n_proj);
            // vector<int> idx2(2 * CEOs::n_proj);
            // iota(idx2.begin(), idx2.end(), 0);
            // std::partial_sort(idx2.begin(), idx2.begin() + CEOs::n_probed_vectors, idx2.end(),[&](int i, int j) { return Y2[i] > Y2[j]; });
            //
            // for (int i = 0; i < CEOs::n_probed_vectors; ++i)
            // {
            //     int Ri_2D_1st = idx1[i];
            //     float fAbsHashValue1 = Y1[Ri_2D_1st];
            //
            //     for (int j = 0; j < CEOs::n_probed_vectors; ++j) {
            //         int R2_2D_2nd = idx2[j];
            //         float fAbsSumHash = Y2[R2_2D_2nd] + fAbsHashValue1; // sum of 2 estimators
            //
            //         // We have 2D * 2D buckets (i.e. random vectors)
            //         int iBucketIndex = Ri_2D_1st * num2D + R2_2D_2nd + bucketBase;
            //
            //         if ((int)minQue.size() < CEOs::n_probed_vectors)
            //             minQue.emplace(iBucketIndex, fAbsSumHash);
            //
            //         else if (fAbsSumHash > minQue.top().m_fValue)
            //         {
            //             minQue.pop();
            //             minQue.emplace(iBucketIndex, fAbsSumHash);
            //         }
            //     }
            // } // End for each pair of random vectors

        } // End for each repeat

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        startTime = chrono::high_resolution_clock::now();
        tsl::robin_map<int, float> mapEst;
        mapEst.reserve(CEOs::n_probed_vectors * CEOs::n_probed_points);

        while (!minQue.empty())
        {
            int iBucketIdx = minQue.top().m_iIndex;
            minQue.pop();

            vector<IFPair> bucket = CEOs::vec2D_Pair_Buckets[iBucketIdx];
            avgProbes += bucket.size();

            for (int i = 0; i < min(CEOs::n_probed_points, (int)bucket.size()); ++i) // bucket.size() = top-m
            {
                int iPointIdx = bucket[i].m_iIndex;
                float fValue = bucket[i].m_fValue;

                if (mapEst.find(iPointIdx) == mapEst.end())
                    mapEst[iPointIdx] = fValue;
                else
                    mapEst[iPointIdx] += fValue;
            }
        } // End for each probed bucket

        estTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        // Note: We use minQue again since it is empty now
        startTime = chrono::high_resolution_clock::now();
        for (auto& it: mapEst)
        {
            float avgEst = it.second;

            if ((int)minQue.size() < CEOs::n_cand)
                minQue.emplace(it.first, avgEst); // use average value for estimation

            // queue is full
            else if (avgEst > minQue.top().m_fValue)
            {
                minQue.pop(); // pop max, and push min hash distance
                minQue.emplace(it.first, avgEst); // use average value for estimation
            }
        }

        assert(minQue.size() == CEOs::n_cand);

        candTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        startTime = chrono::high_resolution_clock::now();
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;
        while (!minQue.empty())
        {
            IFPair ifPair = minQue.top();
            minQue.pop();
            int iPointIdx = ifPair.m_iIndex;

            float fInnerProduct = vecQuery.dot(CEOs::matrix_X.row(iPointIdx));
            avgDist++;

            // Add into priority queue
            if (int(minQueTopK.size()) < n_neighbors)
                minQueTopK.emplace(iPointIdx, fInnerProduct);

            else if (fInnerProduct > minQueTopK.top().m_fValue)
            {
                minQueTopK.pop();
                minQueTopK.emplace(iPointIdx, fInnerProduct);
            }
        }

        // There is the case that we get all 0 index if we do not have enough Top-K
        for (int k = (int)minQueTopK.size() - 1; k >= 0; --k)
        {
            matTopK(q, k) = minQueTopK.top().m_iIndex;
            matTopDist(q, k) = minQueTopK.top().m_fValue;

            minQueTopK.pop();
        }

        distTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startQueryTime);

    if (verbose)
    {
        cout << "Avg number of distance per point: " << avgDist / n_queries << endl;
        cout << "Avg number of probes per point: " << avgProbes / n_queries << endl;

        cout << "Projecting and extracting top-vectors time (ms): " << projTime << endl;
        cout << "Estimating time (ms): " << estTime << endl;
        cout << "Extracting candidates time (ms): " << candTime << endl;
        cout << "Computing distance time (ms): " << distTime << endl;
        cout << "Querying time (ms): " << (float)durTime.count() << endl;

        // string sFileName = "coCEOs_Est_" + int2str(n_neighbors) +
        //                    "_numProj_" + int2str(CEOs::n_proj) +
        //                    "_numRepeat_" + int2str(CEOs::n_repeats) +
        //                    "_topProj_" + int2str(CEOs::n_probed_vectors) +
        //                    "_topPoints_" + int2str(CEOs::n_probed_points) +
        //                    "_cand_" + int2str(n_cand) + ".txt";
        //
        //
        // outputFile(matTopK, sFileName);
    }

    return make_tuple(matTopK, matTopDist);
}

/**
 * Build index of CEOs-Hash (2 layers) for estimating inner product
 * For each repeat, we use two set S, R random vectors, and for each set, we consider top-iProbe vectors closest to the point.
 * For each point Xi, we insert it into top-iProbe^2 buckets, corresponding to these top-iProbe vectors from S and R
 *
 * Data structure:
 * - vector<vector<IFPair>> vec2D_Pair_Buckets: each bucket contains a vector of (pointIdx, projection value) pair (i.e. upper bounded by top-m)
 * - bucketIdx ranges from [0, 4 * D^2 * n_repeats) since we use 2D random vectors for each layer.
 * - Note: Space complexity is O(m * 4 * D^2 * n_repeats) since we have 4D^2 random vectors
 *
 * Algorithm:
 * - We parallel on the point Xi
 * - For each point Xi, we execute n_repeats times of FHT twice for R and S, and for each set, extracting top-iProbe closest random vectors.
 * - Then we insert Xi into these top-iProbe^2 buckets, if the bucket has less than m pairs, or the projection value is larger than the minimum value in the bucket
 *
 * - For each repeat, we maintain LOCAL 4D^2 priority queues of size m (vectorMinQue_TopM) to store the top-m pairs for each bucket since there are 4D^2 pairs of random vectors
 * - We consider [0, D) for positive projection values (closest) and [D, 2D) for negative projections values (furthest) for each set
 *
 * - Note: Falconn++ only insert a point into iProbe buckets, but we insert into iProbe^2 buckets.
 * - We can support this feature in future (by using an additional minQueue or scale iProbe to sqrt{iProbe})
 *
 * - We use locks to avoid multiple threads writing to the same bucket at the same time
 * - After processing all points per each repeat, we update GLOBAL index by dequeuing and storing the top-m (pointIdx, projection value) pairs on vec2D_Pair_Buckets
 *
 * - The process is nearly the same as build_coCEOs_Est1 except that we have two times of FHT and projection for each point
 *
 * @param matX
 */
void CEOs::build_CEOs_Hash2(const Ref<const RowMajorMatrixXf> &matX)
{
    cout << "Building CEOs-Hash index (2 layers)..." << endl;

    cout << "n_points: " << CEOs::n_points << endl;
    cout << "n_features: " << CEOs::n_features << endl;
    cout << "n_repeats: " << CEOs::n_repeats << endl;
    cout << "n_proj: " << CEOs::n_proj << endl;
    cout << "iProbe: " << CEOs::iProbe << endl;
    cout << "top_m: " << CEOs::top_m << endl;
    cout << "centering: " << CEOs::centering << endl;
    cout << "fhtDim: " << CEOs::fhtDim << endl;

    auto start = chrono::high_resolution_clock::now();

    omp_set_num_threads(CEOs::n_threads);

    CEOs::matrix_X = matX;

    // Compute the data center
    CEOs::vecCenter = CEOs::matrix_X.colwise().mean();

    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Copying data time (in seconds): " << (float)duration.count() / 1000 << endl;

    /** Global parameter **/
    int numBucketsPerRepeat = 4 * CEOs::n_proj * CEOs::n_proj;
    int num2D = 2 * CEOs::n_proj;
    int log2_FHT = log2(CEOs::fhtDim);

    CEOs::vec2D_Buckets = vector<vector<int>> (numBucketsPerRepeat * CEOs::n_repeats);

    // Need to store it for the query phase
    bitHD3Generator2(CEOs::fhtDim * CEOs::n_rotate * CEOs::n_repeats, CEOs::seed, CEOs::bitHD1, CEOs::bitHD2);

    // Note: If NUM_LOCKS is large, we might not have enough stack memory if using array
    // if D = 128 = 2^7, then numBuckets = 2^16 = 65536. We aim at 256 KB memory for locks
    // 16K locks is good for million-point data set though it is not good for small data sets.
    constexpr size_t NUM_LOCKS = 16384;
    vector<omp_lock_t> locks(NUM_LOCKS); // NUM_LOCK = 16K locks = only 256 KB

    // Initialize locks
#pragma omp parallel for
    for (size_t i = 0; i < NUM_LOCKS; i++) {
        omp_init_lock(&locks[i]);
    }

    // For each repeat, we compute the local index of (4D^2) buckets (by parallel on points).
    // After that, we update the global index of (4D^2) * n_repeats buckets (by parallel on local buckets)
    for (int repeat = 0; repeat < n_repeats; ++repeat)
    {
        int bucketBase = repeat * numBucketsPerRepeat;
        vector<priority_queue< IFPair, vector<IFPair>, greater<> >> vectorMinQue_TopM(numBucketsPerRepeat);

        /**
        Parallel for each the point Xi: (1) Compute and store dot product, and (2) Extract top-k close/far random vectors
        **/
#pragma omp parallel for
        for (int n = 0; n < CEOs::n_points; ++n)
        {
            VectorXf rotatedX1 = VectorXf::Zero(CEOs::fhtDim); // NUM_PROJECT > PARAM_KERNEL_EMBED_D
            if (CEOs::centering)
                rotatedX1.segment(0, CEOs::n_features) = CEOs::matrix_X.row(n) - CEOs::vecCenter;
            else
                rotatedX1.segment(0, CEOs::n_features) = CEOs::matrix_X.row(n);

            VectorXf rotatedX2 = rotatedX1;

            int rotateBase = CEOs::fhtDim * CEOs::n_rotate * repeat;

            for (int rotate = 0; rotate < CEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedX1(d) *= (2 * static_cast<float>(CEOs::bitHD1[rotateBase + rotate * CEOs::fhtDim + d]) - 1);
                    rotatedX2(d) *= (2 * static_cast<float>(CEOs::bitHD2[rotateBase + rotate * CEOs::fhtDim + d]) - 1);
                }

                fht_float(rotatedX1.data(), log2_FHT);
                fht_float(rotatedX2.data(), log2_FHT);
            }

            // rotateX1 = {x1 * r1, ..., x1 * r_D}
            // rotateX2 = {x1 * s1, ..., x1 * s_D}

            // cout << "We finish random rotating" << endl;

            // This queue is used for finding top-k max hash values and hash index for iProbes on each layer
            // priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopI1, minQueTopI2;
            //
            // /**
            // We use a priority queue to keep top-max abs projection for each repeat
            // Always ensure fhtDim >= n_proj
            // **/
            // // We can speed up with n-th selection but have to deal with the case that we have negative values
            // for (int r = 0; r < CEOs::n_proj; ++r)
            // {
            //     // 1st rotation
            //     int iSign = sgn(rotatedX1(r));
            //     float fAbsHashValue = iSign * rotatedX1(r);
            //
            //     int Ri_2D = r; // index of random vector in [2D] after consider the sign
            //     if (iSign < 0)
            //         // iBucketIndex |= 1UL << log2Project; // set bit at position log2(D)
            //             Ri_2D += CEOs::n_proj; // Be aware the case that n_proj is not 2^(log2Proj)
            //
            //     // iProbe
            //     if ((int)minQueTopI1.size() < CEOs::iProbe)
            //         minQueTopI1.emplace(Ri_2D, fAbsHashValue); // emplace is push without creating temp data
            //     else if (fAbsHashValue > minQueTopI1.top().m_fValue)
            //     {
            //         minQueTopI1.pop();
            //         minQueTopI1.emplace(Ri_2D, fAbsHashValue); // No need IFPair()
            //     }
            //
            //     // 2nd rotation
            //     iSign = sgn(rotatedX2(r));
            //     fAbsHashValue = iSign * rotatedX2(r);
            //
            //     Ri_2D = r;
            //     if (iSign < 0)
            //         // iBucketIndex |= 1UL << log2Project; // set bit at position log2(D)
            //             Ri_2D += CEOs::n_proj; // set bit at position log2(D)
            //
            //     // iProbe (top-iProbe random vector closest to Xn)
            //     if ((int)minQueTopI2.size() < CEOs::iProbe)
            //         minQueTopI2.emplace(Ri_2D, fAbsHashValue);
            //     else if (fAbsHashValue > minQueTopI2.top().m_fValue)
            //     {
            //         minQueTopI2.pop();
            //         minQueTopI2.emplace(Ri_2D, fAbsHashValue);
            //     }
            // }
            //
            // // Convert to vector
            // vector<IFPair> vec_topI1(CEOs::iProbe), vec_topI2(CEOs::iProbe);
            //
            // // iProbe-Falconn++
            // for (int p = CEOs::iProbe - 1; p >= 0; --p)
            // {
            //     vec_topI1[p] = minQueTopI1.top();
            //     minQueTopI1.pop();
            //
            //     vec_topI2[p] = minQueTopI2.top();
            //     minQueTopI2.pop();
            // }
            //
            // // assert(vec_topI1.size() == FalconnLite::iProbe);
            // // assert(vec_topI2.size() == FalconnLite::iProbe);
            //
            // /**
            // Use minQue to find the top-qProbe over 2 layers via sum of 2 estimators
            // vec1 and vec2 are already sorted, and has length of sOptics::topK
            // Note: Heuristic: We consider top-k * top-k pairs for Top-K, and top-p * top-p pairs for Top-M
            // Note: We cannot check all combinations due to significant cost
            // **/
            //
            // for (const auto& ifPair1: vec_topI1) {
            //     int Ri_2D_1st = ifPair1.m_iIndex;
            //     float fAbsHashValue1 = ifPair1.m_fValue;
            //
            //     for (const auto& ifPair2: vec_topI2)
            //     {
            //         int R2_2D_2nd = ifPair2.m_iIndex;
            //         float fAbsSumHash = ifPair2.m_fValue + fAbsHashValue1; // sum of 2 estimators
            //
            //         //We have 2D * 2D buckets (i.e. random vectors)
            //         int iBucketIndex = Ri_2D_1st * num2D + R2_2D_2nd; // (totally we have 2D * 2D buckets)
            //
            //         // assert(iBucketIndex < vectorMinQue_TopM.size());
            //
            //         omp_set_lock(&locks[iBucketIndex % NUM_LOCKS]);
            //
            //         // Note: This implementation is used for controlling the size of dense bucket.
            //
            //         if ((int)vectorMinQue_TopM[iBucketIndex].size() < CEOs::top_m)
            //             vectorMinQue_TopM[iBucketIndex].emplace(n, fAbsSumHash);
            //
            //         else if (fAbsSumHash > vectorMinQue_TopM[iBucketIndex].top().m_fValue)
            //         {
            //             vectorMinQue_TopM[iBucketIndex].pop();
            //             vectorMinQue_TopM[iBucketIndex].emplace(n, fAbsSumHash);
            //         }
            //
            //         omp_unset_lock(&locks[iBucketIndex % NUM_LOCKS]);
            //     }
            // } // End for each pair of random vectors

            // Create a vector from Rotate1 and Rotate2 with size 2 * n_proj, and then find top-iProbe pairs using partial_sort
            VectorXf Y1(2 * CEOs::n_proj);
            Y1.head(CEOs::n_proj) = rotatedX1.segment(0, CEOs::n_proj);
            Y1.tail(CEOs::n_proj) = -rotatedX1.segment(0, CEOs::n_proj);
            vector<int> idx1(2 * CEOs::n_proj);
            iota(idx1.begin(), idx1.end(), 0);

            // Partial sort indices by corresponding Y value (descending)
            std::partial_sort(idx1.begin(), idx1.begin() + CEOs::iProbe, idx1.end(),[&](int i, int j) { return Y1[i] > Y1[j]; });

            VectorXf Y2(2 * CEOs::n_proj);
            Y2.head(CEOs::n_proj) = rotatedX2.segment(0, CEOs::n_proj);
            Y2.tail(CEOs::n_proj) = -rotatedX2.segment(0, CEOs::n_proj);
            vector<int> idx2(2 * CEOs::n_proj);
            iota(idx2.begin(), idx2.end(), 0);
            std::partial_sort(idx2.begin(), idx2.begin() + CEOs::iProbe, idx2.end(),[&](int i, int j) { return Y2[i] > Y2[j]; });

            for (int i = 0; i < CEOs::iProbe; ++i)
            {
                int Ri_2D_1st = idx1[i];
                float fAbsHashValue1 = Y1[Ri_2D_1st];

                for (int j = 0; j < CEOs::iProbe; ++j) {
                    int R2_2D_2nd = idx2[j];
                    float fAbsSumHash = Y2[R2_2D_2nd] + fAbsHashValue1; // sum of 2 estimators

                    // We have 2D * 2D buckets (i.e. random vectors)
                    int iBucketIndex = Ri_2D_1st * num2D + R2_2D_2nd; // (totally we have 2D * 2D buckets)

                    omp_set_lock(&locks[iBucketIndex % NUM_LOCKS]);

                    if ((int)vectorMinQue_TopM[iBucketIndex].size() < CEOs::top_m)
                        vectorMinQue_TopM[iBucketIndex].emplace(n, fAbsSumHash);

                    else if (fAbsSumHash > vectorMinQue_TopM[iBucketIndex].top().m_fValue)
                    {
                        vectorMinQue_TopM[iBucketIndex].pop();
                        vectorMinQue_TopM[iBucketIndex].emplace(n, fAbsSumHash);
                    }

                    omp_unset_lock(&locks[iBucketIndex % NUM_LOCKS]);
                }
            } // End for each pair of random vectors
        } // End for each point

    // Extract top-M for each bucketIdx - Falconn++
#pragma omp parallel for
        for (int b = 0; b < numBucketsPerRepeat; ++b)
        {
            if (!vectorMinQue_TopM[b].empty())
            {
                // bucket-idx shift for different repeat
                int new_bucketIdx = b + bucketBase;
                int m = (int)vectorMinQue_TopM[b].size();

                CEOs::vec2D_Buckets[new_bucketIdx] = vector<int>(m);

                while (!vectorMinQue_TopM[b].empty())
                {
                    // Be aware of the index shift for different repeat
                    CEOs::vec2D_Buckets[new_bucketIdx][m-1] = vectorMinQue_TopM[b].top().m_iIndex; // add into the end of vector
                    vectorMinQue_TopM[b].pop();
                    m--;
                }
            }

        }
    } // End for each repeat

    float totalPoints = 0.0;
    for (size_t i = 0; i < CEOs::vec2D_Buckets.size(); ++i)
        totalPoints += CEOs::vec2D_Buckets[i].size();
    cout << "Total points in all buckets (in millions): " << totalPoints / 1000000 << endl;

    // Outer vector storage
    double total_bytes = CEOs::vec2D_Buckets.capacity() * sizeof(vector<int>) * 1.0;

    // Inner vector storage
    for (auto const& inner : CEOs::vec2D_Buckets) {
        total_bytes += inner.capacity() * sizeof(int) * 1.0;
    }
    cout << "Size of CEOs-Hash2 (without data) index in GB: " << total_bytes / (1 << 30) << "\n";

    // Destroy locks for Falconn++
#pragma omp parallel for
    for (size_t i = 0; i < NUM_LOCKS; ++i) {
        omp_destroy_lock(&locks[i]);
    }

    // Note: Should not clear when running testcase
    // FalconnLite::matrix_X.resize(0, 0);
    // FalconnLite::matrix_R.resize(0, 0);

    duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "CEos-Hash2 constructing time (s): " << (float)duration.count() / 1000 << endl;
}

/**
 * Search CEOs-Hash (2 layers)
 *
 * Algorithm:
 * - We parallel on the query Qi
 * - For each Qi, we execute n_repeats times of FHT, getting its projection values to ALL n_repeats * D random vectors, and selecting top-n_probed_vectors closest/furthest vectors
 * - We aggregate the projections of the top-m pairs of these selected vectors to estimate the inner product using stl::robin_map
 * - There is a re-ranking process with n_cand where we extract top-n_cand points with largest estimated inner product, and compute the exact inner product with these n_cand points
 *
 * @param matQ
 * @param n_neighbors
 * @param verbose
 * @return
 */
tuple<RowMajorMatrixXi, RowMajorMatrixXf> CEOs::search_CEOs_Hash2(const Ref<const RowMajorMatrixXf> & matQ, int n_neighbors, bool verbose)
{
    if (CEOs::n_probed_points > CEOs::top_m)
    {
        cerr << "Error: Number of probed points must be smaller than number of indexed top-m points !" << endl;
        throw std::runtime_error("Invalid n_probed_points");
    }
    if (CEOs::n_probed_vectors > 4 * CEOs::n_proj * CEOs::n_proj * CEOs::n_repeats)
    {
        cerr << "Error: Number of probed vectors must be smaller than 4 * (n_proj^2) * n_repeats !" << endl;
        throw std::runtime_error("Invalid n_probed_vectors");
    }

    int n_queries = matQ.rows();
    if (verbose)
    {
        cout << "n_probed_vectors: " << CEOs::n_probed_vectors << endl;
        cout << "n_probed_points: " << CEOs::n_probed_points << endl;
        cout << "n_cand: " << CEOs::n_cand << endl;
        cout << "n_threads: " << CEOs::n_threads << endl;

        cout << "n_queries: " << n_queries << endl;
    }

    auto startQueryTime = chrono::high_resolution_clock::now();

    float projTime = 0.0, distTime = 0.0;

    RowMajorMatrixXi matTopK = RowMajorMatrixXi::Zero(n_queries, n_neighbors);
    RowMajorMatrixXf matTopDist = RowMajorMatrixXf::Zero(n_queries, n_neighbors);

    int numBucketsPerRepeat = 4 * CEOs::n_proj * CEOs::n_proj;
    int num2D = 2 * CEOs::n_proj;
    int log2_FHT = log2(CEOs::fhtDim);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams

    omp_set_num_threads(CEOs::n_threads);

    // Note: Heuristic methods that consider only n_probed_vector/n_repeats buckets for each repeat
    // Then we only consider sqrt{n_probed_vector/n_repeats} closest vector for each layer
    int top_s = ceil(sqrt(1.0 * CEOs::n_probed_vectors / n_repeats));
    float avgDist = 0.0, avgProbes = 0.0;

#pragma omp parallel for reduction(+:projTime, distTime, avgDist, avgProbes)
    for (int q = 0; q < n_queries; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.row(q);
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueHash; // Get top-s vectors over all repetitions

        // For each repeat
        for (int repeat = 0; repeat < CEOs::n_repeats; ++repeat)
        {
            int bucketBase = repeat * numBucketsPerRepeat;
            int rotateBase = CEOs::fhtDim * CEOs::n_rotate * repeat;

            VectorXf rotatedQ1 = VectorXf::Zero(CEOs::fhtDim);
            rotatedQ1.segment(0, CEOs::n_features) = vecQuery;
            VectorXf rotatedQ2 = rotatedQ1;

            // Note: be careful on centering query since it completely changes to X-c and q-c
            // if (CEOs::centering)
            //     rotatedQ.segment(0, CEOs::n_features) = vecQuery - CEOs::vecCenter;

            for (int rotate = 0; rotate < CEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < CEOs::fhtDim; ++d) {
                    rotatedQ1(d) *= (2 * static_cast<float>(CEOs::bitHD1[rotateBase + rotate * CEOs::fhtDim + d]) - 1);
                    rotatedQ2(d) *= (2 * static_cast<float>(CEOs::bitHD2[rotateBase + rotate * CEOs::fhtDim + d]) - 1);
                }
                fht_float(rotatedQ1.data(), log2_FHT);
                fht_float(rotatedQ2.data(), log2_FHT);
            }

            // This queue is used for finding top-k max hash values and hash index for iProbes on each layer
            priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopQ1, minQueTopQ2;

            /**
            We use a priority queue to keep top-max abs projection for each repeat
            Always ensure fhtDim >= n_proj
            **/
            for (int r = 0; r < CEOs::n_proj; ++r)
            {
                // 1st rotation
                int iSign = sgn(rotatedQ1(r));
                float fAbsHashValue = iSign * rotatedQ1(r);

                int Ri_2D = r; // index of random vector in [2D] after consider the sign
                if (iSign < 0)
                    // iBucketIndex |= 1UL << log2Project; // set bit at position log2(D)
                        Ri_2D += CEOs::n_proj; // Be aware the case that n_proj is not 2^(log2Proj)

                // qProbe
                if ((int)minQueTopQ1.size() < top_s)
                    minQueTopQ1.emplace(Ri_2D, fAbsHashValue); // emplace is push without creating temp data
                else if (fAbsHashValue > minQueTopQ1.top().m_fValue)
                {
                    minQueTopQ1.pop();
                    minQueTopQ1.emplace(Ri_2D, fAbsHashValue); // No need IFPair()
                }

                // 2nd rotation
                iSign = sgn(rotatedQ2(r));
                fAbsHashValue = iSign * rotatedQ2(r);

                Ri_2D = r;
                if (iSign < 0)
                    // iBucketIndex |= 1UL << log2Project; // set bit at position log2(D)
                        Ri_2D += CEOs::n_proj; // set bit at position log2(D)

                // iProbe (top-iProbe random vector closest to Xn)
                if ((int)minQueTopQ2.size() < top_s)
                    minQueTopQ2.emplace(Ri_2D, fAbsHashValue);
                else if (fAbsHashValue > minQueTopQ2.top().m_fValue)
                {
                    minQueTopQ2.pop();
                    minQueTopQ2.emplace(Ri_2D, fAbsHashValue);
                }
            }

            // Convert to vector
            vector<IFPair> vec_topQ1(top_s), vec_topQ2(top_s);
            for (int p = top_s - 1; p >= 0; --p)
            {
                vec_topQ1[p] = minQueTopQ1.top();
                minQueTopQ1.pop();

                vec_topQ2[p] = minQueTopQ2.top();
                minQueTopQ2.pop();
            }


            /**
            Use minQue to find the top-qProbe over 2 layers via sum of 2 estimators
            vec1 and vec2 are already sorted, and has length of sOptics::topK
            Note: Heuristic: We consider top-k * top-k pairs for Top-K, and top-p * top-p pairs for Top-M
            Note: We cannot check all combinations due to significant cost
            **/

            for (const auto& ifPair1: vec_topQ1)
            {
                int Ri_2D_1st = ifPair1.m_iIndex;
                float fAbsHashValue1 = ifPair1.m_fValue;

                for (const auto& ifPair2: vec_topQ2)
                {
                    int R2_2D_2nd = ifPair2.m_iIndex;
                    float fAbsSumHash = ifPair2.m_fValue + fAbsHashValue1; // sum of 2 estimators

                    //We have 2D * 2D buckets (i.e. random vectors)
                    int iBucketIndex = Ri_2D_1st * num2D + R2_2D_2nd + bucketBase; // change to global bucket index since minQue considers all repetitions

                    // Note: This implementation is used for controlling the size of dense bucket.

                    if ((int)minQueHash.size() < CEOs::n_probed_vectors)
                        minQueHash.emplace(iBucketIndex, fAbsSumHash);

                    else if (fAbsSumHash > minQueHash.top().m_fValue)
                    {
                        minQueHash.pop();
                        minQueHash.emplace(iBucketIndex, fAbsSumHash);
                    }
                }
            } // End for each pair of random vectors

            // // Create a vector from Rotate1 and Rotate2 with size 2 * n_proj, and then find top-iProbe pairs using partial_sort
            // VectorXf Y1(2 * CEOs::n_proj);
            // Y1.head(CEOs::n_proj) = rotatedQ1.segment(0, CEOs::n_proj);
            // Y1.tail(CEOs::n_proj) = -rotatedQ1.segment(0, CEOs::n_proj);
            // vector<int> idx1(2 * CEOs::n_proj);
            // iota(idx1.begin(), idx1.end(), 0);
            //
            // // Partial sort indices by corresponding Y value (descending)
            // std::partial_sort(idx1.begin(), idx1.begin() + top_s, idx1.end(),[&](int i, int j) { return Y1[i] > Y1[j]; });
            //
            // VectorXf Y2(2 * CEOs::n_proj);
            // Y2.head(CEOs::n_proj) = rotatedQ2.segment(0, CEOs::n_proj);
            // Y2.tail(CEOs::n_proj) = -rotatedQ2.segment(0, CEOs::n_proj);
            // vector<int> idx2(2 * CEOs::n_proj);
            // iota(idx2.begin(), idx2.end(), 0);
            // std::partial_sort(idx2.begin(), idx2.begin() + top_s, idx2.end(),[&](int i, int j) { return Y2[i] > Y2[j]; });
            //
            // for (int i = 0; i < top_s; ++i)
            // {
            //     int Ri_2D_1st = idx1[i];
            //     float fAbsHashValue1 = Y1[Ri_2D_1st];
            //
            //     for (int j = 0; j < top_s; ++j) {
            //         int R2_2D_2nd = idx2[j];
            //         float fAbsSumHash = Y2[R2_2D_2nd] + fAbsHashValue1; // sum of 2 estimators
            //
            //         // We have 2D * 2D buckets (i.e. random vectors)
            //         int iBucketIndex = Ri_2D_1st * num2D + R2_2D_2nd + bucketBase;
            //
            //         if ((int)minQueHash.size() < CEOs::n_probed_vectors)
            //             minQueHash.emplace(iBucketIndex, fAbsSumHash);
            //
            //         else if (fAbsSumHash > minQueHash.top().m_fValue)
            //         {
            //             minQueHash.pop();
            //             minQueHash.emplace(iBucketIndex, fAbsSumHash);
            //         }
            //     }
            // } // End for each pair of random vectors

        } // End for each repeat

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        startTime = chrono::high_resolution_clock::now();
        boost::dynamic_bitset<> bitsetHist(CEOs::n_points);
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;

        while (!minQueHash.empty())
        {
            int bucketIdx = minQueHash.top().m_iIndex;
            minQueHash.pop();

            avgProbes += (float)CEOs::vec2D_Buckets[bucketIdx].size();

            // Sample one random point from the bucket Ri
            for (const int pointIdx: CEOs::vec2D_Buckets[bucketIdx]) {
                if (~bitsetHist[pointIdx]) // do not put the query point itself
                {
                    bitsetHist[pointIdx] = true;
                    float fInnerProduct = vecQuery.dot(CEOs::matrix_X.row(pointIdx));

                    if ((int)minQueTopK.size() < n_neighbors)
                        minQueTopK.emplace(pointIdx, fInnerProduct); // emplace is push without creating temp data
                    else if (fInnerProduct > minQueTopK.top().m_fValue)
                    {
                        minQueTopK.pop();
                        minQueTopK.emplace(pointIdx, fInnerProduct); // No need IFPair()
                    }
                }
            }
        } // End for each probed bucket

        avgDist += (float)bitsetHist.count();

        // There is the case that we get all 0 index if we do not have enough Top-K
        for (int k = (int)minQueTopK.size() - 1; k >= 0; --k)
        {
            matTopK(q, k) = minQueTopK.top().m_iIndex;
            matTopDist(q, k) = minQueTopK.top().m_fValue;

            minQueTopK.pop();
        }

        distTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
    }

    auto durTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startQueryTime);

    if (verbose)
    {
        cout << "Avg number of distance per point: " << avgDist / n_queries << endl;
        cout << "Avg number of probes per point: " << avgProbes / n_queries << endl;

        cout << "Projecting and extracting top-vectors time (ms): " << projTime << endl;
        cout << "Computing distance time (ms): " << distTime << endl;
        cout << "Querying time (ms): " << (float)durTime.count() << endl;

        // string sFileName = "coCEOs_Est_" + int2str(n_neighbors) +
        //                    "_numProj_" + int2str(CEOs::n_proj) +
        //                    "_numRepeat_" + int2str(CEOs::n_repeats) +
        //                    "_topProj_" + int2str(CEOs::n_probed_vectors) +
        //                    "_topPoints_" + int2str(CEOs::n_probed_points) +
        //                    "_cand_" + int2str(n_cand) + ".txt";
        //
        //
        // outputFile(matTopK, sFileName);
    }

    return make_tuple(matTopK, matTopDist);
}
