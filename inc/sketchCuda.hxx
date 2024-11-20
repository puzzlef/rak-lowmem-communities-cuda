#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "_main.hxx"




#pragma region METHODS
#pragma region ACCUMULATE
/**
 * Accumulate edge weight to community in a Misra-Gries sketch [device function].
 * @tparam SHARED are the slots shared among threads?
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param has has community in list / free slot index (scratch)
 * @param c community to accumulate edge weight to
 * @param w edge weight to accumulate
 * @param g cooperative thread group
 * @param s slot index
 */
template <bool SHARED=false, class K, class V, class TG>
inline void __device__ sketchAccumulateCudU(K *mcs, V *mws, int *has, K c, V w, const TG& g, int s) {
  // Initialize variables.
  if (s==0) *has = -1;
  g.sync();
  // Add edge weight to community.
  if (mcs[s]==c) {
    if (!SHARED) mws[s] += w;
    else atomicAdd(&mws[s], w);
    *has = 0;
  }
  g.sync();
  // Done if community is already in the list.
  if (*has==0) return;
  do {
    // Find empty slot.
    if (mws[s]==V()) atomicMax(has, s);
    g.sync();
    if (*has<0) break;
    // Add community to list.
    if (*has==s) {
      if (!SHARED) {
        mcs[s] = c;
        mws[s] = w;
      }
      else {
        if (atomicCAS(&mws[s], V(), w)==V()) mcs[s] = c;
        else *has = -1;
      }
    }
    if (SHARED) g.sync();  // `has` may be been updated
  } while (SHARED && *has<0);
  // Subtract edge weight from non-matching communities.
  if (*has<0) {
    if (!SHARED) mws[s] -= w;
    else atomicAdd(&mws[s], -w);
  }
}


/**
 * Accumulate edge weight to community in a warp-sized Misra-Gries sketch [device function].
 * @tparam SHARED are the slots shared among threads?
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param c community to accumulate edge weight to
 * @param w edge weight to accumulate
 * @param g cooperative thread group
 * @param s slot index
 * @note Uses warp-specific optimization, but the sketch size must be <=32.
 */
template <bool SHARED=false, class K, class V, class TG>
inline void __device__ sketchAccumulateWarpCudU(K *mcs, V *mws, K c, V w, const TG& g, int s) {
  // Add edge weight to community.
  if (mcs[s]==c) {
    if (!SHARED) mws[s] += w;
    else atomicAdd(&mws[s], w);
  }
  uint32_t has = g.ballot(mcs[s]==c);
  // Done if community is already in the list.
  if (has) return;
  uint32_t fre = 0;
  while (1) {
    // Find empty slot.
    fre = g.ballot(mws[s]==V());
    uint32_t frs = __ffs(fre) - 1;
    if (fre==0) break;
    // Add community to list.
    if (frs==s) {
      if (!SHARED) {
        mcs[s] = c;
        mws[s] = w;
      }
      else {
        if (atomicCAS(&mws[s], V(), w)==V()) mcs[s] = c;
        else fre = 0;
      }
    }
    if (SHARED) fre = g.all(fre!=0);  // `fre` may be been updated
    if (!SHARED || fre!=0) break;
  }
  // Subtract edge weight from non-matching communities.
  if (fre==0) {
    if (!SHARED) mws[s] -= w;
    else atomicAdd(&mws[s], -w);
  }
}
#pragma endregion




#pragma region MERGE
/**
 * Merge two Misra-Gries sketches [device function].
 * @tparam SHARED are the slots shared among threads?
 * @tparam SLOTS number of slots in the sketch
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param has has community in list / free slot index (scratch)
 * @param pmcs majority linked communities to merge
 * @param pmws total edge weight to each majority community to merge
 * @param g cooperative thread group
 * @param s slot index
 */
template <bool SHARED=false, int SLOTS=8, class K, class V, class TG>
inline void __device__ sketchMergeCudU(K *mcs, V *mws, int *has, const K* pmcs, const V* pmws, const TG& g, int s) {
  for (int i=0; i<SLOTS; ++i) {
    K c = pmcs[i];
    V w = pmws[i];
    if (!w) continue;
    sketchAccumulateCudU<SHARED>(mcs, mws, has, c, w, g, s);
  }
}


/**
 * Merge two warp-sized Misra-Gries sketches [device function].
 * @tparam SHARED are the slots shared among threads?
 * @tparam SLOTS number of slots in the sketch
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param pmcs majority linked communities to merge
 * @param pmws total edge weight to each majority community to merge
 * @param g cooperative thread group
 * @param s slot index
 * @note Uses warp-specific optimization, but the sketch size must be <=32.
 */
template <bool SHARED=false, int SLOTS=8, class K, class V, class TG>
inline void __device__ sketchMergeWarpCudU(K *mcs, V *mws, const K* pmcs, const V* pmws, const TG& g, int s) {
  for (int i=0; i<SLOTS; ++i) {
    K c = pmcs[i];
    V w = pmws[i];
    if (!w) continue;
    sketchAccumulateWarpCudU<SHARED>(mcs, mws, c, w, g, s);
  }
}
#pragma endregion




#pragma region CLEAR
/**
 * Clear a Misra-Gries sketch [device function].
 * @tparam SHARED are the slots shared among threads?
 * @tparam SLOTS number of slots in the sketch
 * @param mws total edge weight to each majority community (updated)
 * @param i thread index
 */
template <bool SHARED=false, int SLOTS=8, class V>
inline void __device__ sketchClearCudU(V *mws, int i) {
  if (!SHARED || i<SLOTS) mws[i] = V();
}
#pragma endregion




#pragma region MAX
/**
 * Find entry in Misra-Gries sketch with maximum value, using cooperative group [device function].
 * @param mcs majority linked communities (updated, entry 0 is max)
 * @param mws total edge weight to each majority community (updated, entry 0 is max)
 * @param SLOTS number of slots in the sketch
 * @param g cooperative thread group
 * @param i thread index
 */
template <class K, class V, class TG>
inline void __device__ sketchMaxGroupReduceCudU(K *mcs, V *mws, int SLOTS, const TG& g, int i) {
  for (; SLOTS>1;) {
    int DS = SLOTS/2;
    if (i<DS && mws[i+DS] > mws[i]) {
      mcs[i] = mcs[i+DS];
      mws[i] = mws[i+DS];
    }
    g.sync();
    SLOTS = DS;
  }
}


/**
 * Find entry in Misra-Gries sketch with maximum value, using thread block [device function].
 * @param mcs majority linked communities (updated, entry 0 is max)
 * @param mws total edge weight to each majority community (updated, entry 0 is max)
 * @param SLOTS number of slots in the sketch
 * @param i thread index
 */
template <class K, class V>
inline void __device__ sketchMaxBlockReduceCudU(K *mcs, V *mws, int SLOTS, int i) {
  for (; SLOTS>32;) {
    int DS = SLOTS/2;
    if (i<DS && mws[i+DS] > mws[i]) {
      mcs[i] = mcs[i+DS];
      mws[i] = mws[i+DS];
    }
    __syncthreads();
    SLOTS = DS;
  }
  for (; SLOTS>1;) {
    int DS = SLOTS/2;
    if (i<DS && mws[i+DS] > mws[i]) {
      mcs[i] = mcs[i+DS];
      mws[i] = mws[i+DS];
    }
    __syncwarp();
    SLOTS = DS;
  }
}
#pragma endregion
#pragma endregion
