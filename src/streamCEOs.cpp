//
// Created by npha145 on 22/09/24.
//

#include "streamCEOs.h"
#include "Header.h"
#include "Utilities.h"


/**
 * Build index of CEOs-Hash (1 layer) for estimating inner product
 * For each random vector, we only store the top-m points to this vector
 * So the index size is O(m * D * n_repeats), which is much smaller than CEOs where we need to store projection values of all points to all random vectors
 * We need 2 * D * n_repeats buckets, where the [0, D) is for positive projection values (closest) and [D, 2D) is for negative projections values (furthest)
 *
 * Data structure:
 * - vector<vector<IFPair>> vec2D_Pair_Buckets: each bucket contains a vector of (pointIdx, estimateInner Product) (i.e. top-m pairs)
 * - We need the estimator since we will update top-m pairs to support streaming updates
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
 *
 * @param matX
 */
void streamCEOs::build1(const Ref<const RowMajorMatrixXf> &matX)
{
    cout << "Building streamCEOs-Hash index..." << endl;
    cout << "n_features: " << streamCEOs::n_features << endl;
    cout << "n_repeats: " << streamCEOs::n_repeats << endl;
    cout << "n_proj: " << streamCEOs::n_proj << endl;
    cout << "top_m: " << streamCEOs::top_m << endl;
    cout << "fhtDim: " << streamCEOs::fhtDim << endl;

    streamCEOs::n_points = matX.rows();
    cout << "n_points: " << streamCEOs::n_points << endl;

    auto start = chrono::high_resolution_clock::now();

    omp_set_num_threads(streamCEOs::n_threads);

    // Not sure how to do this in multi-thread
    for (int n = 0; n < streamCEOs::n_points; ++n)
    {
        streamCEOs::deque_X.push_back(matX.row(n)); //emplace_back() causes error if calling with only matX.row(n)
    }

    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Copying data time (s): " << (float)duration.count() / 1000 << endl;

    // streamCEOs::matrix_P has (4 * top-m) x (proj * repeats) since query phase will access each column corresponding each random vector
    // We need 2 * top-points position for (index, value)
    int numBucketsPerRepeat = 2 * streamCEOs::n_proj;
    streamCEOs::vec2D_Pair_Buckets = vector<vector<IFPair>> (numBucketsPerRepeat * streamCEOs::n_repeats);

    // Need to store it for the query phase
    bitHD3Generator(streamCEOs::fhtDim * streamCEOs::n_rotate * streamCEOs::n_repeats, streamCEOs::seed, streamCEOs::bitHD1);

    int log2_FHT = log2(streamCEOs::fhtDim);
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
    for (int repeat = 0; repeat < streamCEOs::n_repeats; ++repeat) {
        int bucketBase = repeat * numBucketsPerRepeat;
        vector<priority_queue< IFPair, vector<IFPair>, greater<> >> vectorMinQue_TopM(numBucketsPerRepeat);

#pragma omp parallel for reduction(+:projTime)
        for (int n = 0; n < streamCEOs::n_points; ++n)
        {
            auto startTime = chrono::high_resolution_clock::now();

            VectorXf rotatedX = VectorXf::Zero(streamCEOs::fhtDim);
            rotatedX.segment(0, streamCEOs::n_features) = matX.row(n);

            int rotateBase = streamCEOs::fhtDim * streamCEOs::n_rotate * repeat;

            for (int rotate = 0; rotate < streamCEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < streamCEOs::fhtDim; ++d) {
                    rotatedX(d) *= (2 * static_cast<float>(streamCEOs::bitHD1[rotateBase + rotate * streamCEOs::fhtDim + d]) - 1);
                }

                fht_float(rotatedX.data(), log2_FHT);
            }

            for (int r = 0; r < streamCEOs::n_proj; ++r)
            {

                int iSign = sgn(rotatedX(r));
                float fAbsHashValue = iSign * rotatedX(r);

                int Ri_2D = r; // index of random vector in [2D] after consider the sign
                if (iSign < 0)
                    // iBucketIndex |= 1UL << log2Project; // set bit at position log2(D)
                        Ri_2D += streamCEOs::n_proj; // Be aware the case that n_proj is not 2^(log2Proj)

                omp_set_lock(&locks[Ri_2D % NUM_LOCKS]);

                if ((int)vectorMinQue_TopM[Ri_2D].size() < streamCEOs::top_m)
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

            streamCEOs::vec2D_Pair_Buckets[new_bucketIdx] = vector<IFPair>(m);

            while (!vectorMinQue_TopM[b].empty())
            {
                // Be aware of the index shift for different repeat
                streamCEOs::vec2D_Pair_Buckets[new_bucketIdx][m-1] = vectorMinQue_TopM[b].top();
                vectorMinQue_TopM[b].pop();
                m--;
            }

            extractTopPointsTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
        }
    } // End for each repeat

    double dSize = 1.0 * streamCEOs::deque_X.size() * streamCEOs::n_features * sizeof(float) / (1 << 30);
    cout << "Size of data set in GB: " << dSize << endl;

    for (size_t b = 0; b < streamCEOs::vec2D_Pair_Buckets.size(); ++b)
        dSize += 1.0 * streamCEOs::vec2D_Pair_Buckets[b].size() * sizeof(IFPair) / (1 << 30);

    cout << "Size of streamCEOs-Hash1 index (including data) in GB: " << dSize << endl;

#pragma omp parallel for
    for (size_t i = 0; i < NUM_LOCKS; ++i) {
        omp_destroy_lock(&locks[i]);
    }

    duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Projecting time (s): " << projTime / 1000 << endl;
    cout << "Extracting top-m points time (s): " << extractTopPointsTime / 1000 << endl;
    cout << "streamCEOs-Hash1 constructing time (s): " << (float)duration.count() / 1000 << endl;
}

/**
 * Build index of CEOs-Hash (1 layer) for estimating inner product
 * For each random vector, we only store the top-m points to this vector
 * So the index size is O(m * D * n_repeats), which is much smaller than CEOs where we need to store projection values of all points to all random vectors
 * We need 2 * D * n_repeats buckets, where the [0, D) is for positive projection values (closest) and [D, 2D) is for negative projections values (furthest)
 *
 * Data structure:
 * - vector<vector<IFPair>> vec2D_Pair_Buckets: each bucket contains a vector of (pointIdx, estimateInner Product) (i.e. top-m pairs)
 * - We need the estimator since we will update top-m pairs to support streaming updates
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
 *
 * @param matX
 */
void streamCEOs::build2(const Ref<const RowMajorMatrixXf> &matX)
{
    cout << "Building streamCEOs-Hash index..." << endl;
    cout << "n_features: " << streamCEOs::n_features << endl;
    cout << "n_repeats: " << streamCEOs::n_repeats << endl;
    cout << "n_proj: " << streamCEOs::n_proj << endl;
    cout << "top_m: " << streamCEOs::top_m << endl;
    cout << "fhtDim: " << streamCEOs::fhtDim << endl;

    streamCEOs::n_points = matX.rows();
    cout << "n_points: " << streamCEOs::n_points << endl;

    auto start = chrono::high_resolution_clock::now();

    omp_set_num_threads(streamCEOs::n_threads);

    // Not sure how to do this in multi-thread
    for (int n = 0; n < streamCEOs::n_points; ++n)
    {
        streamCEOs::deque_X.push_back(matX.row(n)); //emplace_back() causes error if calling with only matX.row(n)
    }

    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Copying data time (s): " << (float)duration.count() / 1000 << endl;

    /** Global parameter **/
    int numBucketsPerRepeat = 4 * streamCEOs::n_proj * streamCEOs::n_proj;
    int num2D = 2 * streamCEOs::n_proj;

    streamCEOs::vec2D_Pair_Buckets = vector<vector<IFPair>> (numBucketsPerRepeat * streamCEOs::n_repeats);

    // Need to store it for the query phase
    bitHD3Generator2(streamCEOs::fhtDim * streamCEOs::n_rotate * streamCEOs::n_repeats, streamCEOs::seed, streamCEOs::bitHD1, streamCEOs::bitHD2);


    int log2_FHT = log2(streamCEOs::fhtDim);
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
    for (int repeat = 0; repeat < streamCEOs::n_repeats; ++repeat) {
        int bucketBase = repeat * numBucketsPerRepeat;
        vector<priority_queue< IFPair, vector<IFPair>, greater<> >> vectorMinQue_TopM(numBucketsPerRepeat);

#pragma omp parallel for reduction(+:projTime)
        for (int n = 0; n < streamCEOs::n_points; ++n)
        {
            auto startTime = chrono::high_resolution_clock::now();
            VectorXf rotatedX1 = VectorXf::Zero(streamCEOs::fhtDim);

            rotatedX1.segment(0, streamCEOs::n_features) = matX.row(n);
            VectorXf rotatedX2 = rotatedX1;

            // NOTE: Done heree
            int rotateBase = streamCEOs::fhtDim * streamCEOs::n_rotate * repeat;

            for (int rotate = 0; rotate < streamCEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < streamCEOs::fhtDim; ++d) {
                    rotatedX1(d) *= (2 * static_cast<float>(streamCEOs::bitHD1[rotateBase + rotate * streamCEOs::fhtDim + d]) - 1);
                    rotatedX2(d) *= (2 * static_cast<float>(streamCEOs::bitHD2[rotateBase + rotate * streamCEOs::fhtDim + d]) - 1);
                }

                fht_float(rotatedX1.data(), log2_FHT);
                fht_float(rotatedX2.data(), log2_FHT);
            }

            // Create a vector from Rotate1 and Rotate2 with size 2 * n_proj, and then find top-iProbe pairs using partial_sort
            VectorXf Y1(2 * streamCEOs::n_proj);
            Y1.head(streamCEOs::n_proj) = rotatedX1.segment(0, streamCEOs::n_proj);
            Y1.tail(streamCEOs::n_proj) = -rotatedX1.segment(0, streamCEOs::n_proj);
            vector<int> idx1(2 * streamCEOs::n_proj);
            iota(idx1.begin(), idx1.end(), 0);

            // Partial sort indices by corresponding Y value (descending)
            std::partial_sort(idx1.begin(), idx1.begin() + streamCEOs::iProbe, idx1.end(),[&](int i, int j) { return Y1[i] > Y1[j]; });

            VectorXf Y2(2 * streamCEOs::n_proj);
            Y2.head(streamCEOs::n_proj) = rotatedX2.segment(0, streamCEOs::n_proj);
            Y2.tail(streamCEOs::n_proj) = -rotatedX2.segment(0, streamCEOs::n_proj);
            vector<int> idx2(2 * streamCEOs::n_proj);
            iota(idx2.begin(), idx2.end(), 0);
            std::partial_sort(idx2.begin(), idx2.begin() + streamCEOs::iProbe, idx2.end(),[&](int i, int j) { return Y2[i] > Y2[j]; });

            for (int i = 0; i < streamCEOs::iProbe; ++i)
            {
                int Ri_2D_1st = idx1[i];
                float fAbsHashValue1 = Y1[Ri_2D_1st];

                for (int j = 0; j < streamCEOs::iProbe; ++j) {
                    int R2_2D_2nd = idx2[j];
                    float fAbsSumHash = Y2[R2_2D_2nd] + fAbsHashValue1; // sum of 2 estimators

                    // We have 2D * 2D buckets (i.e. random vectors)
                    int iBucketIndex = Ri_2D_1st * num2D + R2_2D_2nd; // (totally we have 2D * 2D buckets)

                    omp_set_lock(&locks[iBucketIndex % NUM_LOCKS]);

                    if ((int)vectorMinQue_TopM[iBucketIndex].size() < streamCEOs::top_m)
                        vectorMinQue_TopM[iBucketIndex].emplace(n, fAbsSumHash);

                    else if (fAbsSumHash > vectorMinQue_TopM[iBucketIndex].top().m_fValue)
                    {
                        vectorMinQue_TopM[iBucketIndex].pop();
                        vectorMinQue_TopM[iBucketIndex].emplace(n, fAbsSumHash);
                    }

                    omp_unset_lock(&locks[iBucketIndex % NUM_LOCKS]);
                }
            } // End for each pair of random vectors

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

            streamCEOs::vec2D_Pair_Buckets[new_bucketIdx] = vector<IFPair>(m);

            while (!vectorMinQue_TopM[b].empty())
            {
                // Be aware of the index shift for different repeat
                streamCEOs::vec2D_Pair_Buckets[new_bucketIdx][m-1] = vectorMinQue_TopM[b].top();
                vectorMinQue_TopM[b].pop();
                m--;
            }

            extractTopPointsTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
        }
    } // End for each repeat

    double dSize = 1.0 * streamCEOs::deque_X.size() * streamCEOs::n_features * sizeof(float) / (1 << 30);
    cout << "Size of data set in GB: " << dSize << endl;

    for (size_t b = 0; b < streamCEOs::vec2D_Pair_Buckets.size(); ++b)
        dSize += 1.0 * streamCEOs::vec2D_Pair_Buckets[b].size() * sizeof(IFPair) / (1 << 30);

    cout << "Size of streamCEOs-Hash2 index (including data) in GB: " << dSize << endl;

#pragma omp parallel for
    for (size_t i = 0; i < NUM_LOCKS; ++i) {
        omp_destroy_lock(&locks[i]);
    }

    duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Projecting time (s): " << projTime / 1000 << endl;
    cout << "Extracting top-m points time (s): " << extractTopPointsTime / 1000 << endl;
    cout << "streamCEOs-Hash2 constructing time (s): " << (float)duration.count() / 1000 << endl;
}

/**
 * Build index of CEOs-Hash (1 layer) for estimating inner product
 * For each random vector, we only store the top-m points to this vector
 * So the index size is O(m * D * n_repeats), which is much smaller than CEOs where we need to store projection values of all points to all random vectors
 * We need 2 * D * n_repeats buckets, where the [0, D) is for positive projection values (closest) and [D, 2D) is for negative projections values (furthest)
 *
 * Data structure:
 * - vector<vector<IFPair>> vec2D_Pair_Buckets: each bucket contains a vector of (pointIdx, estimateInner Product) (i.e. top-m pairs)
 * - We need the estimator since we will update top-m pairs to support streaming updates
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
 *
 * @param matX
 */
void streamCEOs::update1(const Ref<const RowMajorMatrixXf> & mat_newX, int n_delPoints)
{
    cout << "Updating CEOs-Hash1 index..." << endl;

    int n_newPoints = mat_newX.rows();
    int n_beforeDeleteSize = streamCEOs::deque_X.size();

    if (n_delPoints > n_beforeDeleteSize)
    {
        cerr << "Error: Number of removed points must be smaller than the current number of points !" << endl;
        exit(1);
    }

    if (streamCEOs::top_m > n_beforeDeleteSize + n_newPoints - n_delPoints)
    {
        cerr << "Error: There is not enough indexed top-points for coCEOs after update !" << endl;
        exit(1);
    }

    auto start = chrono::high_resolution_clock::now();

    for (int n = 0; n < n_delPoints; ++n)
        deque_X.pop_front(); // remove old points from the front
    for (int n = 0; n < n_newPoints; ++n)
        deque_X.push_back(mat_newX.row(n)); // Note: emplace_back(mat_newX.row(n)) causes bug

    float durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count() / 1000.0;
    cout << "Updating data time (in ms): " << durTime << " ms" << endl;

    int numBucketsPerRepeat = 2 * streamCEOs::n_proj;
    streamCEOs::vec2D_Pair_Buckets = vector<vector<IFPair>> (numBucketsPerRepeat * streamCEOs::n_repeats);

    int log2_FHT = log2(streamCEOs::fhtDim);
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
    for (int repeat = 0; repeat < streamCEOs::n_repeats; ++repeat) {
        int bucketBase = repeat * numBucketsPerRepeat;
        vector<priority_queue< IFPair, vector<IFPair>, greater<> >> vectorMinQue_TopM(numBucketsPerRepeat);

#pragma omp parallel for reduction(+:projTime)
        for (int n = 0; n < n_newPoints; ++n)
        {
            auto startTime = chrono::high_resolution_clock::now();

            VectorXf rotatedX = VectorXf::Zero(streamCEOs::fhtDim);
            rotatedX.segment(0, streamCEOs::n_features) = mat_newX.row(n);

            int rotateBase = streamCEOs::fhtDim * streamCEOs::n_rotate * repeat;

            for (int rotate = 0; rotate < streamCEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < streamCEOs::fhtDim; ++d) {
                    rotatedX(d) *= (2 * static_cast<float>(streamCEOs::bitHD1[rotateBase + rotate * streamCEOs::fhtDim + d]) - 1);
                }

                fht_float(rotatedX.data(), log2_FHT);
            }

            for (int r = 0; r < streamCEOs::n_proj; ++r)
            {

                int iSign = sgn(rotatedX(r));
                float fAbsHashValue = iSign * rotatedX(r);

                int Ri_2D = r; // index of random vector in [2D] after consider the sign
                if (iSign < 0)
                    // iBucketIndex |= 1UL << log2Project; // set bit at position log2(D)
                        Ri_2D += streamCEOs::n_proj; // Be aware the case that n_proj is not 2^(log2Proj)

                omp_set_lock(&locks[Ri_2D % NUM_LOCKS]);

                if ((int)vectorMinQue_TopM[Ri_2D].size() < streamCEOs::top_m)
                    vectorMinQue_TopM[Ri_2D].emplace(n, fAbsHashValue);

                else if (fAbsHashValue > vectorMinQue_TopM[Ri_2D].top().m_fValue)
                {
                    vectorMinQue_TopM[Ri_2D].pop();
                    vectorMinQue_TopM[Ri_2D].emplace(n, fAbsHashValue);
                }

                omp_unset_lock(&locks[Ri_2D % NUM_LOCKS]);
            } // End for each random vector

            projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        } // End for each new point

        // Update global data structure - delete old points and merge with new points
        // Update new idx for the new point
#pragma omp parallel for reduction(+: extractTopPointsTime)
        for (size_t b = 0; b < vectorMinQue_TopM.size(); ++b)
        {
            // b in range [0, numBucketsPerRepeat * n_repeats)
            auto startTime = chrono::high_resolution_clock::now();
            int new_bucketIdx = bucketBase + b;

            vector<IFPair> vecA; // points already in the index - remember to update pointIdx
            if (!streamCEOs::vec2D_Pair_Buckets[new_bucketIdx].empty())
            {
                for (auto ifpair: streamCEOs::vec2D_Pair_Buckets[new_bucketIdx]) {
                    if (ifpair.m_iIndex >= n_delPoints) // only keep points idx > num_delPoints
                    {
                        // update pointIdx by subtracting n_delPoint, i.e. old-pointIdx will be in range [0, beforeDeleteSize - n_delPoints]
                        ifpair.m_iIndex -= n_delPoints;
                        vecA.push_back(ifpair);
                    }
                }
            }

            int sizeA = (int)vecA.size();
            int sizeB = (int)vectorMinQue_TopM[b].size();

            vector<IFPair> vecB = vector<IFPair>(sizeB); // points will be inserted - remember to update pointIdx
            int i = sizeB - 1;
            while (!vectorMinQue_TopM[b].empty())
            {
                IFPair ifpair = vectorMinQue_TopM[b].top();
                ifpair.m_iIndex += n_beforeDeleteSize - n_delPoints; // new-pointIdx from [n_curSize, end)
                vecB[i] = ifpair;
                i--;
                vectorMinQue_TopM[b].pop();
            }

            // Now ready to merge two vectors
            // 2) Two-pointer truncated merge (DESC)
            vector<IFPair> out;
            out.reserve(streamCEOs::top_m);
            i = 0;
            int j = 0;

            while ((int)out.size() < streamCEOs::top_m && (i < sizeA || j < sizeB)) {
                bool takeA = (i < sizeA) && (j >= sizeB || vecA[i].m_fValue >= vecB[j].m_fValue);
                out.push_back(takeA ? vecA[i++] : vecB[j++]);
            }

            streamCEOs::vec2D_Pair_Buckets[new_bucketIdx] = out; // update global data structure

            extractTopPointsTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
        }
    } // End for each repeat

    streamCEOs::n_points = deque_X.size();
    cout << "n_points (after update): " << streamCEOs::n_points << endl;

    double dSize = 1.0 * streamCEOs::deque_X.size() * streamCEOs::n_features * sizeof(float) / (1 << 30);
    cout << "Size of data set in GB: " << dSize << endl;

    for (size_t b = 0; b < streamCEOs::vec2D_Pair_Buckets.size(); ++b)
        dSize += 1.0 * streamCEOs::vec2D_Pair_Buckets[b].size() * sizeof(IFPair) / (1 << 30);

    cout << "Size of streamCEOs-Hash1 index (including data) in GB: " << dSize << endl;

#pragma omp parallel for
    for (size_t i = 0; i < NUM_LOCKS; ++i) {
        omp_destroy_lock(&locks[i]);
    }

    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Projecting time (s): " << projTime / 1000 << endl;
    cout << "Extracting top-m points time (s): " << extractTopPointsTime / 1000 << endl;
    cout << "streamCEOs-Hash1 updating time (s): " << (float)duration.count() / 1000 << endl;
}

void streamCEOs::update2(const Ref<const RowMajorMatrixXf> & mat_newX, int n_delPoints)
{
    cout << "Updating CEOs-Hash2 index..." << endl;

    int n_newPoints = mat_newX.rows();
    int n_beforeDeleteSize = streamCEOs::deque_X.size();

    if (n_delPoints > n_beforeDeleteSize)
    {
        cerr << "Error: Number of removed points must be smaller than the current number of points !" << endl;
        exit(1);
    }

    if (streamCEOs::top_m > n_beforeDeleteSize + n_newPoints - n_delPoints)
    {
        cerr << "Error: There is not enough indexed top-points for coCEOs after update !" << endl;
        exit(1);
    }

    auto start = chrono::high_resolution_clock::now();

    for (int n = 0; n < n_delPoints; ++n)
        deque_X.pop_front(); // remove old points from the front
    for (int n = 0; n < n_newPoints; ++n)
        deque_X.push_back(mat_newX.row(n)); // Note: emplace_back(mat_newX.row(n)) causes bug

    float durTime = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count() / 1000.0;
    cout << "Updating data time (in ms): " << durTime << " ms" << endl;

    /** Global parameter **/
    int numBucketsPerRepeat = 4 * streamCEOs::n_proj * streamCEOs::n_proj;
    int num2D = 2 * streamCEOs::n_proj;

    streamCEOs::vec2D_Pair_Buckets = vector<vector<IFPair>> (numBucketsPerRepeat * streamCEOs::n_repeats);

    int log2_FHT = log2(streamCEOs::fhtDim);
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
    for (int repeat = 0; repeat < streamCEOs::n_repeats; ++repeat) {
        int bucketBase = repeat * numBucketsPerRepeat;
        vector<priority_queue< IFPair, vector<IFPair>, greater<> >> vectorMinQue_TopM(numBucketsPerRepeat);

#pragma omp parallel for reduction(+:projTime)
        for (int n = 0; n < n_newPoints; ++n)
        {
            auto startTime = chrono::high_resolution_clock::now();

            VectorXf rotatedX1 = VectorXf::Zero(streamCEOs::fhtDim);
            rotatedX1.segment(0, streamCEOs::n_features) = mat_newX.row(n);
            VectorXf rotatedX2 = rotatedX1;

            int rotateBase = streamCEOs::fhtDim * streamCEOs::n_rotate * repeat;

            for (int rotate = 0; rotate < streamCEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < streamCEOs::fhtDim; ++d) {
                    rotatedX1(d) *= (2 * static_cast<float>(streamCEOs::bitHD1[rotateBase + rotate * streamCEOs::fhtDim + d]) - 1);
                    rotatedX2(d) *= (2 * static_cast<float>(streamCEOs::bitHD2[rotateBase + rotate * streamCEOs::fhtDim + d]) - 1);
                }

                fht_float(rotatedX1.data(), log2_FHT);
                fht_float(rotatedX2.data(), log2_FHT);
            }

            // Create a vector from Rotate1 and Rotate2 with size 2 * n_proj, and then find top-iProbe pairs using partial_sort
            VectorXf Y1(2 * streamCEOs::n_proj);
            Y1.head(streamCEOs::n_proj) = rotatedX1.segment(0, streamCEOs::n_proj);
            Y1.tail(streamCEOs::n_proj) = -rotatedX1.segment(0, streamCEOs::n_proj);
            vector<int> idx1(2 * streamCEOs::n_proj);
            iota(idx1.begin(), idx1.end(), 0);

            // Partial sort indices by corresponding Y value (descending)
            std::partial_sort(idx1.begin(), idx1.begin() + streamCEOs::iProbe, idx1.end(),[&](int i, int j) { return Y1[i] > Y1[j]; });

            VectorXf Y2(2 * streamCEOs::n_proj);
            Y2.head(streamCEOs::n_proj) = rotatedX2.segment(0, streamCEOs::n_proj);
            Y2.tail(streamCEOs::n_proj) = -rotatedX2.segment(0, streamCEOs::n_proj);
            vector<int> idx2(2 * streamCEOs::n_proj);
            iota(idx2.begin(), idx2.end(), 0);
            std::partial_sort(idx2.begin(), idx2.begin() + streamCEOs::iProbe, idx2.end(),[&](int i, int j) { return Y2[i] > Y2[j]; });

            for (int i = 0; i < streamCEOs::iProbe; ++i)
            {
                int Ri_2D_1st = idx1[i];
                float fAbsHashValue1 = Y1[Ri_2D_1st];

                for (int j = 0; j < streamCEOs::iProbe; ++j) {
                    int R2_2D_2nd = idx2[j];
                    float fAbsSumHash = Y2[R2_2D_2nd] + fAbsHashValue1; // sum of 2 estimators

                    // We have 2D * 2D buckets (i.e. random vectors)
                    int iBucketIndex = Ri_2D_1st * num2D + R2_2D_2nd; // (totally we have 2D * 2D buckets)

                    omp_set_lock(&locks[iBucketIndex % NUM_LOCKS]);

                    if ((int)vectorMinQue_TopM[iBucketIndex].size() < streamCEOs::top_m)
                        vectorMinQue_TopM[iBucketIndex].emplace(n, fAbsSumHash);

                    else if (fAbsSumHash > vectorMinQue_TopM[iBucketIndex].top().m_fValue)
                    {
                        vectorMinQue_TopM[iBucketIndex].pop();
                        vectorMinQue_TopM[iBucketIndex].emplace(n, fAbsSumHash);
                    }

                    omp_unset_lock(&locks[iBucketIndex % NUM_LOCKS]);
                }
            } // End for each pair of random vectors

            projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        } // End for each new point

        // Update global data structure - delete old points and merge with new points
        // Update new idx for the new point
#pragma omp parallel for reduction(+: extractTopPointsTime)
        for (size_t b = 0; b < vectorMinQue_TopM.size(); ++b)
        {
            // b in range [0, numBucketsPerRepeat * n_repeats)
            auto startTime = chrono::high_resolution_clock::now();
            int new_bucketIdx = bucketBase + b;

            vector<IFPair> vecA; // points already in the index - remember to update pointIdx
            if (!streamCEOs::vec2D_Pair_Buckets[new_bucketIdx].empty())
            {
                for (auto ifpair: streamCEOs::vec2D_Pair_Buckets[new_bucketIdx]) {
                    if (ifpair.m_iIndex >= n_delPoints) // only keep points idx > num_delPoints
                    {
                        // update pointIdx by subtracting n_delPoint, i.e. old-pointIdx will be in range [0, beforeDeleteSize - n_delPoints]
                        ifpair.m_iIndex -= n_delPoints;
                        vecA.push_back(ifpair);
                    }
                }
            }

            int sizeA = (int)vecA.size();
            int sizeB = (int)vectorMinQue_TopM[b].size();

            vector<IFPair> vecB = vector<IFPair>(sizeB); // points will be inserted - remember to update pointIdx
            int i = sizeB - 1;
            while (!vectorMinQue_TopM[b].empty())
            {
                IFPair ifpair = vectorMinQue_TopM[b].top();
                ifpair.m_iIndex += n_beforeDeleteSize - n_delPoints; // new-pointIdx from [n_curSize, end)
                vecB[i] = ifpair;
                i--;
                vectorMinQue_TopM[b].pop();
            }

            // Now ready to merge two vectors
            // 2) Two-pointer truncated merge (DESC)
            vector<IFPair> out;
            out.reserve(streamCEOs::top_m);
            i = 0;
            int j = 0;

            while ((int)out.size() < streamCEOs::top_m && (i < sizeA || j < sizeB)) {
                bool takeA = (i < sizeA) && (j >= sizeB || vecA[i].m_fValue >= vecB[j].m_fValue);
                out.push_back(takeA ? vecA[i++] : vecB[j++]);
            }

            streamCEOs::vec2D_Pair_Buckets[new_bucketIdx] = out; // update global data structure

            extractTopPointsTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;
        }
    } // End for each repeat

    streamCEOs::n_points = deque_X.size();
    cout << "n_points (after update): " << streamCEOs::n_points << endl;

    double dSize = 1.0 * streamCEOs::deque_X.size() * streamCEOs::n_features * sizeof(float) / (1 << 30);
    cout << "Size of data set in GB: " << dSize << endl;

    for (size_t b = 0; b < streamCEOs::vec2D_Pair_Buckets.size(); ++b)
        dSize += 1.0 * streamCEOs::vec2D_Pair_Buckets[b].size() * sizeof(IFPair) / (1 << 30);

    cout << "Size of streamCEOs-Hash2 index (including data) in GB: " << dSize << endl;

#pragma omp parallel for
    for (size_t i = 0; i < NUM_LOCKS; ++i) {
        omp_destroy_lock(&locks[i]);
    }

    auto duration = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start);
    cout << "Projecting time (s): " << projTime / 1000 << endl;
    cout << "Extracting top-m points time (s): " << extractTopPointsTime / 1000 << endl;
    cout << "streamCEOs-Hash2 updating time (s): " << (float)duration.count() / 1000 << endl;
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
tuple<RowMajorMatrixXi, RowMajorMatrixXf> streamCEOs::search1(const Ref<const RowMajorMatrixXf> & matQ, int n_neighbors, bool verbose)
{
    if (streamCEOs::n_probed_vectors > streamCEOs::n_proj * streamCEOs::n_repeats)
    {
        cerr << "Error: Number of probed vectors must be smaller than n_proj * n_repeats !" << endl;
        exit(1);
    }

    int n_queries = matQ.rows();
    if (verbose)
    {
        cout << "n_probed_vectors: " << streamCEOs::n_probed_vectors << endl;
        cout << "n_cand: " << streamCEOs::n_probed_vectors *  streamCEOs::top_m << endl;
        cout << "n_threads: " << streamCEOs::n_threads << endl;

        cout << "n_queries: " << n_queries << endl;
    }

    auto startQueryTime = chrono::high_resolution_clock::now();

    float projTime = 0.0, distTime = 0.0;

    RowMajorMatrixXi matTopK = RowMajorMatrixXi::Zero(n_queries, n_neighbors);
    RowMajorMatrixXf matTopDist = RowMajorMatrixXf::Zero(n_queries, n_neighbors);

    int log2_FHT = log2(streamCEOs::fhtDim);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(streamCEOs::n_threads);

#pragma omp parallel for reduction(+:projTime, distTime)
    for (int q = 0; q < n_queries; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.row(q);
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueHash;

        // For each repeat
        for (int repeat = 0; repeat < streamCEOs::n_repeats; ++repeat)
        {
            int bucketBase = repeat * streamCEOs::n_proj * 2;

            int rotateBase = streamCEOs::fhtDim * streamCEOs::n_rotate * repeat;
            VectorXf rotatedQ = VectorXf::Zero(streamCEOs::fhtDim);

            rotatedQ.segment(0, streamCEOs::n_features) = vecQuery;
            // Note: be careful on centering query since it completely changes to X-c and q-c
            // if (streamCEOs::centering)
            //     rotatedQ.segment(0, streamCEOs::n_features) = vecQuery - streamCEOs::vecCenter;

            for (int rotate = 0; rotate < streamCEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < streamCEOs::fhtDim; ++d) {
                    rotatedQ(d) *= (2 * static_cast<float>(streamCEOs::bitHD1[rotateBase + rotate * streamCEOs::fhtDim + d]) - 1);
                }
                fht_float(rotatedQ.data(), log2_FHT);
            }

            // Note for segment(i, size) where i is starting index, size is segment size
            for (int d = 0; d < streamCEOs::n_proj; ++d)
            {
                int iBucketIdx = d; // in case positive projected value
                float fAbsProjValue = rotatedQ(d);

                if (fAbsProjValue < 0) {
                    fAbsProjValue = -fAbsProjValue; // get abs
                    iBucketIdx += streamCEOs::n_proj;
                }

                iBucketIdx += bucketBase;

                if ((int)minQueHash.size() < streamCEOs::n_probed_vectors)
                    minQueHash.emplace(iBucketIdx, fAbsProjValue);
                else if (fAbsProjValue > minQueHash.top().m_fValue) {
                    minQueHash.pop(); // pop max, and push min hash distance
                    minQueHash.emplace(iBucketIdx, fAbsProjValue); // Hack:
                }
            }
        } // End for each repeat

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        startTime = chrono::high_resolution_clock::now();
        boost::dynamic_bitset<> bitsetHist(streamCEOs::n_points);
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;

        while (!minQueHash.empty())
        {
            int bucketIdx = minQueHash.top().m_iIndex;
            minQueHash.pop();

            // Sample one random point from the bucket Ri
            for (const auto ifpair: streamCEOs::vec2D_Pair_Buckets[bucketIdx])
            {
                int pointIdx = ifpair.m_iIndex;

                if (~bitsetHist[pointIdx]) // do not put the query point itself
                {
                    bitsetHist[pointIdx] = true;
                    float fInnerProduct = vecQuery.dot(streamCEOs::deque_X[pointIdx]);

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
        cout << "StreamCEOs-Hash1 querying time (ms): " << (float)durTime.count() << endl;
    }

    return make_tuple(matTopK, matTopDist);
}

tuple<RowMajorMatrixXi, RowMajorMatrixXf> streamCEOs::search2(const Ref<const RowMajorMatrixXf> & matQ, int n_neighbors, bool verbose)
{
    if (streamCEOs::n_probed_vectors > streamCEOs::n_proj * streamCEOs::n_repeats)
    {
        cerr << "Error: Number of probed vectors must be smaller than n_proj * n_repeats !" << endl;
        exit(1);
    }

    int n_queries = matQ.rows();
    if (verbose)
    {
        cout << "n_probed_vectors: " << streamCEOs::n_probed_vectors << endl;
        cout << "n_cand: " << streamCEOs::n_probed_vectors *  streamCEOs::top_m << endl;
        cout << "n_threads: " << streamCEOs::n_threads << endl;

        cout << "n_queries: " << n_queries << endl;
    }

    auto startQueryTime = chrono::high_resolution_clock::now();

    float projTime = 0.0, distTime = 0.0;

    RowMajorMatrixXi matTopK = RowMajorMatrixXi::Zero(n_queries, n_neighbors);
    RowMajorMatrixXf matTopDist = RowMajorMatrixXf::Zero(n_queries, n_neighbors);

    int numBucketsPerRepeat = 4 * streamCEOs::n_proj * streamCEOs::n_proj;
    int num2D = 2 * streamCEOs::n_proj;
    int log2_FHT = log2(streamCEOs::fhtDim);

    // omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(streamCEOs::n_threads);

    // Note: Heuristic methods that consider only n_probed_vector/n_repeats buckets for each repeat
    // Then we only consider sqrt{n_probed_vector/n_repeats} closest vector for each layer
    int top_s = ceil(sqrt(1.0 * streamCEOs::n_probed_vectors / n_repeats));
    float avgDist = 0.0, avgProbes = 0.0;

#pragma omp parallel for reduction(+:projTime, distTime, avgDist, avgProbes)
    for (int q = 0; q < n_queries; ++q)
    {
        auto startTime = chrono::high_resolution_clock::now();

        // Get hash value of all hash table first
        VectorXf vecQuery = matQ.row(q);
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueHash;

        // For each repeat
        for (int repeat = 0; repeat < streamCEOs::n_repeats; ++repeat)
        {
            int bucketBase = repeat * numBucketsPerRepeat;
            int rotateBase = streamCEOs::fhtDim * streamCEOs::n_rotate * repeat;

            VectorXf rotatedQ1 = VectorXf::Zero(streamCEOs::fhtDim);
            rotatedQ1.segment(0, streamCEOs::n_features) = vecQuery;
            VectorXf rotatedQ2 = rotatedQ1;

            // Note: be careful on centering query since it completely changes to X-c and q-c
            // if (CEOs::centering)
            //     rotatedQ.segment(0, CEOs::n_features) = vecQuery - CEOs::vecCenter;

            for (int rotate = 0; rotate < streamCEOs::n_rotate; ++rotate)
            {
                for (int d = 0; d < streamCEOs::fhtDim; ++d) {
                    rotatedQ1(d) *= (2 * static_cast<float>(streamCEOs::bitHD1[rotateBase + rotate * streamCEOs::fhtDim + d]) - 1);
                    rotatedQ2(d) *= (2 * static_cast<float>(streamCEOs::bitHD2[rotateBase + rotate * streamCEOs::fhtDim + d]) - 1);
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
            for (int r = 0; r < streamCEOs::n_proj; ++r)
            {
                // 1st rotation
                int iSign = sgn(rotatedQ1(r));
                float fAbsHashValue = iSign * rotatedQ1(r);

                int Ri_2D = r; // index of random vector in [2D] after consider the sign
                if (iSign < 0)
                    // iBucketIndex |= 1UL << log2Project; // set bit at position log2(D)
                    Ri_2D += streamCEOs::n_proj; // Be aware the case that n_proj is not 2^(log2Proj)

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
                    Ri_2D += streamCEOs::n_proj; // set bit at position log2(D)

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

                    if ((int)minQueHash.size() < streamCEOs::n_probed_vectors)
                        minQueHash.emplace(iBucketIndex, fAbsSumHash);

                    else if (fAbsSumHash > minQueHash.top().m_fValue)
                    {
                        minQueHash.pop();
                        minQueHash.emplace(iBucketIndex, fAbsSumHash);
                    }
                }
            } // End for each pair of random vectors
        } // End for each repeat

        projTime += (float)chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

        startTime = chrono::high_resolution_clock::now();
        boost::dynamic_bitset<> bitsetHist(streamCEOs::n_points);
        priority_queue< IFPair, vector<IFPair>, greater<> > minQueTopK;

        while (!minQueHash.empty())
        {
            int bucketIdx = minQueHash.top().m_iIndex;
            minQueHash.pop();

            avgProbes += (float)streamCEOs::vec2D_Pair_Buckets[bucketIdx].size();

            // Sample one random point from the bucket Ri
            for (const auto ifpair: streamCEOs::vec2D_Pair_Buckets[bucketIdx])
            {
                int pointIdx = ifpair.m_iIndex;

                if (~bitsetHist[pointIdx]) // do not put the query point itself
                {
                    bitsetHist[pointIdx] = true;
                    float fInnerProduct = vecQuery.dot(streamCEOs::deque_X[pointIdx]);

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
        cout << "StreamCEOs-Hash2 querying time (ms): " << (float)durTime.count() << endl;
    }

    return make_tuple(matTopK, matTopDist);
}

