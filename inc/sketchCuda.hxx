#pragma once
#include <cstdint>
#include "_main.hxx"




#pragma region METHODS
/**
 * Accumulate edge weight to community in a full Misra-Gries sketch.
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param has is community already in the list? (temporary buffer, updated)
 * @param fre an empty slot in the list (temporary buffer, updated)
 * @param c community to accumulate edge weight to
 * @param w edge weight to accumulate
 * @param i thread index
 */
template <class K, class V>
inline void __device__ fullSketchAccumulateCudU(K *mcs, V *mws, int *has, int *fre, K c, V w, int i) {
  // Initialize variables.
  if (i==0) *has = 0;
  if (i==0) *fre = -1;
  // Add edge weight to community.
  if (mcs[i]==c) {
    mws[i] += w;
    *has = 1;
  }
  __syncthreads();
  // Done if community is already in the list.
  if (*has) return;
  // Find empty slot.
  if (mws[i]==0) atomicMax(fre, i);
  __syncthreads();
  // Add community to list.
  if (*fre==i) {
    mcs[i] = c;
    mws[i] = w;
  }
  // Subtract edge weight from non-matching communities.
  if (*fre<0) mws[i] -= w;
}


/**
 * Accumulate edge weight to community in a warp-sized full Misra-Gries sketch.
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param c community to accumulate edge weight to
 * @param w edge weight to accumulate
 * @param i thread index
 * @note The size of the sketch has to be 32.
 */
template <class K, class V>
inline void __device__ fullSketchAccumulateWarpCudU(K *mcs, V *mws, K c, V w, int i) {
  const uint32_t ALL = 0xFFFFFFFF;
  // Add edge weight to community.
  if (mcs[i]==c) mws[i] += w;
  uint32_t has = __ballot_sync(ALL, mcs[i]==c);
  // Done if community is already in the list.
  if (has) return;
  // Find empty slot.
  uint32_t fre = __ffs(__ballot_sync(ALL, mws[i]==0)) - 1;
  // Add community to list.
  if (fre==i) {
    mcs[i] = c;
    mws[i] = w;
  }
  // Subtract edge weight from non-matching communities.
  if (fre==0) mws[i] -= w;
}


/**
 * Accumulate edge weight to community in a small Misra-Gries sketch.
 * @tparam SLOTS number of slots in the sketch
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param has is community already in the list? (temporary buffer, updated)
 * @param fre an empty slot in the list (temporary buffer, updated)
 * @param c community to accumulate edge weight to
 * @param w edge weight to accumulate
 * @param i thread index
 */
template <int SLOTS=8, class K, class V>
inline void __device__ smallSketchAccumulateCudU(K *mcs, V *mws, int *has, int *fre, K c, V w, int i) {
  const int s = i % SLOTS;  // Slot index
  const int p = i / SLOTS;  // Page index
  // Initialize variables.
  if (s==0) has[p] = 0;
  if (s==0) fre[p] = -1;
  // Add edge weight to community.
  if (mcs[s]==c) {
    atomicAdd(&mws[s], w);
    has[p] = 1;
  }
  __syncthreads();
  // Done if community is already in the list.
  if (has[p]) return;
  // Find empty slot.
  if (mws[s]==0) atomicMax(&fre[p], s);
  __syncthreads();
  // Add community to list.
  if (fre[p]==s) {
    if (atomicCAS(&mws[s], V(), w)==V()) mcs[s] = c;
    else fre[p] = -1;
  }
  __syncthreads();
  // Subtract edge weight from non-matching communities.
  if (fre[p]<0) atomicAdd(&mws[s], -w);
}


/**
 * Accumulate edge weight to community in a warp-sized small Misra-Gries sketch.
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param c community to accumulate edge weight to
 * @param w edge weight to accumulate
 * @param i thread index
 * @note The size of the sketch has to be 32.
 */
template <class K, class V>
inline void __device__ smallSketchAccumulateWarpCudU(K *mcs, V *mws, K c, V w, int i) {
  const uint32_t ALL = 0xFFFFFFFF;
  const int s = i % 32;  // Slot index
  // Add edge weight to community.
  if (mcs[s]==c) atomicAdd(mws[s], w);
  uint32_t has = __ballot_sync(ALL, mcs[s]==c);
  // Done if community is already in the list.
  if (has) return;
  // Find empty slot.
  uint32_t fre = __ffs(__ballot_sync(ALL, mws[s]==0)) - 1;
  // Add community to list.
  if (fre==s) {
    if (atomicCAS(&mws[s], V(), w)==V()) mcs[s] = c;
    else fre = -1;
  }
  uint32_t ful = __all_sync(ALL, fre<0);
  // Subtract edge weight from non-matching communities.
  if (ful) atomicAdd(mws[s], -w);
}


/**
 * Accumulate edge weight to community in a full Misra-Gries sketch.
 * @tparam SLOTS number of slots in the sketch
 * @tparam BLIM size of each thread block
 * @tparam WARP whether to use warp-sized sketch
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param has is community already in the list? (temporary buffer, updated)
 * @param fre an empty slot in the list (temporary buffer, updated)
 * @param c community to accumulate edge weight to
 * @param w edge weight to accumulate
 * @param i thread index
 */
template <int SLOTS=8, int BLIM=128, bool WARP=true, class K, class V>
inline void __device__ sketchAccumulateCudU(K *mcs, V *mws, int *has, int *fre, K c, V w, int i) {
  if (SLOTS < BLIM) {
    if (SLOTS==32 && WARP) smallSketchAccumulateWarpCudU(mcs, mws, c, w, i);
    else smallSketchAccumulateCudU<SLOTS>(mcs, mws, has, fre, c, w, i);
  }
  else {
    if (SLOTS==32 && WARP) fullSketchAccumulateWarpCudU(mcs, mws, c, w, i);
    else fullSketchAccumulateCudU(mcs, mws, has, fre, c, w, i);
  }
}
#pragma endregion
