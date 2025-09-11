//
// Created by npha145 on 22/09/24.
//

#ifndef STREAMCEOS_H
#define STREAMCEOS_H


#include "Header.h"
#include <tuple>

class streamCEOs{

protected:

    int n_points;
    int n_features;

    int n_proj = 256;
    int n_rotate = 3;
    int top_m = 100;
    int iProbe = 1;

    int n_repeats = 1;
    int seed = -1;

    deque<VectorXf> deque_X; // It is fast for remove and add at the end of queue
    vector<vector<IFPair>> vec2D_Pair_Buckets;

    int fhtDim;
    boost::dynamic_bitset<> bitHD1, bitHD2;

public:

    int n_threads = -1;

    // Query param
    int n_probed_vectors = 10;

    // we need n_features to design fhtDim
    streamCEOs(int d){
//        n_points = n; // we do not need n_points as we support add_remove
        n_features = d;
    }

    void set_streamCEOsParam(int numProj, int repeats, int m, int i, int t, int s) {

        n_proj = numProj;
        n_repeats = repeats;
        top_m = m;
        n_probed_vectors = 20;
        iProbe = i;

        set_threads(t);
        seed = s;

        // setting fht dimension. Note n_proj must be 2^a, and > n_features
        // Ensure fhtDim > n_proj
        if (n_proj < n_features)
            fhtDim = 1 << int(ceil(log2(n_features)));
        else
            fhtDim = 1 << int(ceil(log2(n_proj)));

    }

    void clear() {

        deque_X.clear();
        vec2D_Pair_Buckets.clear();

        bitHD1.clear();
        bitHD2.clear();
    }

    void set_threads(int t)
    {
        if (t <= 0)
            n_threads = omp_get_max_threads();
        else
            n_threads = t;
    }

    void build1(const Ref<const RowMajorMatrixXf> &);
    void update1(const Ref<const RowMajorMatrixXf> &, int = 0);
    tuple<RowMajorMatrixXi, RowMajorMatrixXf> search1(const Ref<const RowMajorMatrixXf> &, int, bool=false);

    void build2(const Ref<const RowMajorMatrixXf> &);
    void update2(const Ref<const RowMajorMatrixXf> &, int = 0);
    tuple<RowMajorMatrixXi, RowMajorMatrixXf> search2(const Ref<const RowMajorMatrixXf> &, int, bool=false);

    //================================================================
    void build1_HighMem(const Ref<const RowMajorMatrixXf> &);
    void update1_HighMem(const Ref<const RowMajorMatrixXf> &, int = 0);
    tuple<RowMajorMatrixXi, RowMajorMatrixXf> search_Est1_HighMem(const Ref<const RowMajorMatrixXf> &, int, bool=false);
    tuple<RowMajorMatrixXi, RowMajorMatrixXf> search_Hash1_HighMem(const Ref<const RowMajorMatrixXf> &, int, bool=false);
    //================================================================

    ~streamCEOs() { clear(); }

// TODO: Support other distances
//  Add support L2 via transformation: x --> {2x, -|x|^2}, q --> {q, 1}
//  If called via Python on million points, call this transformation externally
//  If called via loading file on billion points, then it must an internal transformation

// TODO: Support billion points
//  Add sketch to estimate distance
//  coCEOs might be useful to estimate distance if increasing top-r. Then we need small n_cand.
//  Since n_cand is small, then disk-based (SSD) index should work very well on coCEOs

};


#endif //STREAMCEOS_H
