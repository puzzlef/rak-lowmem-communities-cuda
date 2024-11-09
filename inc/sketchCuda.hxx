#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "_main.hxx"




#pragma region METHODS
#pragma region ACCUMULATE
/**
 * Accumulate edge weight to community in a Misra-Gries sketch.
 * @tparam SHARED are the slots shared among threads?
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param has is community already in the list? (scratch)
 * @param frs a free slot in the list (scratch)
 * @param c community to accumulate edge weight to
 * @param w edge weight to accumulate
 * @param g cooperative thread group
 * @param s slot index
 */
template <bool SHARED=false, class K, class V, class TG>
inline void __device__ sketchAccumulateCudU(K *mcs, V *mws, int *has, int *frs, K c, V w, const TG& g, int s) {
  // Initialize variables.
  if (s==0) {
    *has = 0;
    *frs = -1;
  }
  g.sync();
  // Add edge weight to community.
  if (mcs[s]==c) {
    if (!SHARED) mws[s] += w;
    else atomicAdd(mws+s, w);
    *has = 1;
  }
  g.sync();
  // Done if community is already in the list.
  if (*has) return;
  // Find empty slot.
  if (mws[s]==0) atomicMax(frs, s);
  g.sync();
  // Add community to list.
  if (*frs==s) {
    if (!SHARED) {
      mcs[s] = c;
      mws[s] = w;
    }
    else {
      if (atomicCAS(mws+s, V(), w)==V()) mcs[s] = c;
      else *frs = -1;
    }
  }
  if (SHARED) g.sync();  // `frs` may be been updated
  // Subtract edge weight from non-matching communities.
  if (*frs<0) {
    if (!SHARED) mws[s] -= w;
    else atomicAdd(mws+s, -w);
  }
}


/**
 * Accumulate edge weight to community in a warp-sized Misra-Gries sketch.
 * @tparam SHARED are the slots shared among threads?
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param c community to accumulate edge weight to
 * @param w edge weight to accumulate
 * @param g cooperative thread group
 * @param s slot index
 * @note Uses warp-specific optimization, but the sketch size must be 32.
 */
template <class K, class V, class TG>
inline void __device__ sketchAccumulateWarpCudU(K *mcs, V *mws, K c, V w, const TG& g, int s) {
  const uint32_t ALL = 0xFFFFFFFF;
  // Add edge weight to community.
  if (mcs[s]==c) {
    if (!SHARED) mws[s] += w;
    else atomicAdd(mws+s,  w);
  }
  uint32_t has = __ballot_sync(ALL, mcs[s]==c);
  // Done if community is already in the list.
  if (has) return;
  // Find empty slot.
  uint32_t fre = __ballot_sync(ALL, mws[s]==0);
  uint32_t frs = __ffs(fre) - 1;
  // Add community to list.
  if (frs==s) {
    if (!SHARED) {
      mcs[s] = c;
      mws[s] = w;
    }
    else {
      if (atomicCAS(mws+s, V(), w)==V()) mcs[s] = c;
      else fre = 0;
    }
  }
  if (SHARED) g.sync();  // `fre` may be been updated
  // Subtract edge weight from non-matching communities.
  if (fre==0) {
    if (!SHARED) mws[s] -= w;
    else atomicAdd(mws+s, -w);
  }
}
#pragma endregion




#pragma region CLEAR
/**
 * Clear a Misra-Gries sketch.
 * @tparam SHARED are the slots shared among threads?
 * @tparam SLOTS number of slots in the sketch
 * @param mws total edge weight to each majority community (updated)
 * @param t thread index
 */
template <bool SHARED=false, int SLOTS=8, class V>
inline void __device__ sketchClearCudU(V *mws, int t) {
  if (!SHARED || t<SLOTS) mws[t] = V();
}
#pragma endregion




#pragma region MAX
/**
 * Find entry in Misra-Gries sketch with maximum value [device function].
 * @param mcs majority linked communities (updated, entry 0 is max)
 * @param mws total edge weight to each majority community (updated, entry 0 is max)
 * @param SLOTS number of slots in the sketch
 * @param i thread index
 */
template <class K, class V>
inline void __device__ sketchMaxCudU(K *mcs, V *mws, int SLOTS, int i) {
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
