#pragma once
#include <cstdint>
#include <utility>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "rak.hxx"
#include "rakCuda.hxx"

using std::vector;
using std::pair;
using std::make_pair;
using std::max;
using std::distance;
using std::partition;
using cooperative_groups::tiled_partition;
using cooperative_groups::this_thread_block;




#pragma region METHODS
#pragma region SCAN COMMUNITIES
/**
 * Scan communities connected to a vertex, for Boyer-Moore voting algorithm [device function].
 * @tparam SELF include self-loops?
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param vcom community each vertex belongs to
 * @param u given vertex
 * @param d original community of u
 * @param i start index
 * @param DI index stride
 * @returns majority community, and total edge weight to it
 */
template <bool SELF=false, class O, class K, class V>
inline auto __device__ rakLowmemBmScanCommunitiesCud(const O *xoff, const K *xedg, const V *xwei, const K *vcom, K u, K d, O i, O DI) {
  O EO = xoff[u];
  O EN = xoff[u+1] - xoff[u];
  K mc = d;
  V mw = V();
  for (; i<EN; i+=DI) {
    K v = xedg[EO+i];
    V w = xwei[EO+i];
    K c = vcom[v];
    if (!SELF && u==v) continue;
    if (c==mc)     mw += w;
    else if (mw>w) mw -= w;
    else { mc = c; mw  = w; }
  }
  return makePairCu(mc, mw);
}


/**
 * Scan communities connected to a vertex, for Misra-Gries sketch [device function].
 * @tparam SELF include self-loops?
 * @tparam SHARED are the slots shared among threads?
 * @tparam USEWARP use warp-specific optimization?
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param has has community in list / free slot index (scratch)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param vcom community each vertex belongs to
 * @param u given vertex
 * @param g cooperative thread group
 * @param s slot index
 * @param i start index
 * @param DI index stride
 */
template <bool SELF=false, bool SHARED=false, bool USEWARP=true, class O, class K, class V, class TG>
inline void __device__ rakLowmemMgScanCommunitiesCudU(K *mcs, V *mws, int *has, const O *xoff, const K *xedg, const V *xwei, const K *vcom, K u, const TG& g, int s, O i, O DI) {
  O EO = xoff[u];
  O EN = xoff[u+1] - xoff[u];
  for (; i<EN; i+=DI) {
    K v = xedg[EO+i];
    V w = xwei[EO+i];
    K c = vcom[v];
    if (!SELF && u==v) continue;
    if (USEWARP) sketchAccumulateWarpCudU<SHARED>(mcs, mws, c, w, g, s);
    else sketchAccumulateCudU<SHARED>(mcs, mws, has, c, w, g, s);
  }
}


/**
 * Rescan communities connected to a vertex, for Misra-Gries sketch [device function].
 * @tparam SELF include self-loops?
 * @tparam SHARED are the slots shared among threads?
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param vcom community each vertex belongs to
 * @param u given vertex
 * @param s slot index
 * @param i start index
 * @param DI index stride
 */
template <bool SELF=false, bool SHARED=false, class O, class K, class V>
inline void __device__ rakLowmemMgRescanCommunitiesCudU(K *mcs, V *mws, const O *xoff, const K *xedg, const V *xwei, const K *vcom, K u, int s, O i, O DI) {
  O EO = xoff[u];
  O EN = xoff[u+1] - xoff[u];
  for (; i<EN; i+=DI) {
    K v = xedg[EO+i];
    V w = xwei[EO+i];
    K c = vcom[v];
    if (!SELF && u==v) continue;
    if (mcs[s]!=c) continue;
    if (!SHARED) mws[s] += w;
    else atomicAdd(mws+s, w);
  }
}
#pragma endregion




#pragma region MOVE ITERATION
/**
 * Move each vertex to its best community, for Boyer-Moore voting algorithm, using thread-per-vertex approach [kernel].
 * @tparam BLIM size of each thread block
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int BLIM=32, class O, class K, class V, class F>
void __global__ rakLowmemBmMoveIterationThreadCukU(uint64_cu *ncom, K *vcom, F *vaff, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  DEFINE_CUDA(t, b, B, G);
  __shared__ uint64_cu ncomb[BLIM];
  ncomb[t] = 0;
  for (K u=NB+b*B+t; u<NE; u+=G*B) {
    if (!vaff[u]) continue;
    K d = vcom[u];
    // Scan communities connected to u.
    auto mcw = rakLowmemBmScanCommunitiesCud<false>(xoff, xedg, xwei, vcom, u, d, O(0), O(1));
    // Find best community for u.
    K c = mcw.first;
    vaff[u] = F(0);  // Mark u as unaffected
    if (c==d) continue;
    if (PICKLESS && c>d) continue;  // Pick smaller community-id (to avoid community swaps)
    vcom[u] = c;
    ++ncomb[t];
    rakMarkNeighborsCudU(vaff, xoff, xedg, u, 0, 1);
  }
  // Update number of changed vertices.
  __syncthreads();
  sumValuesBlockReduceCudU(ncomb, BLIM, t);
  if (t==0) atomicAdd(ncom, ncomb[0]);
}


/**
 * Move each vertex to its best community, for Boyer-Moore voting algorithm, using thread-per-vertex approach.
 * @tparam BLIM size of each thread block
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int BLIM=32, class O, class K, class V, class F>
inline void rakLowmemBmMoveIterationThreadCuU(uint64_cu *ncom, K *vcom, F *vaff, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  const int B = BLIM;
  const int G = gridSizeCu(NE-NB, B, GRID_LIMIT_MAP_CUDA);
  rakLowmemBmMoveIterationThreadCukU<BLIM><<<G, B>>>(ncom, vcom, vaff, xoff, xedg, xwei, NB, NE, PICKLESS);
}


/**
 * Move each vertex to its best community, for Boyer-Moore voting algorithm, using block-per-vertex approach [kernel].
 * @tparam BLIM size of each thread block
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int BLIM=256, class O, class K, class V, class F>
void __global__ rakLowmemBmMoveIterationBlockCukU(uint64_cu *ncom, K *vcom, F *vaff, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  DEFINE_CUDA(t, b, B, G);
  __shared__ K mcs[BLIM];
  __shared__ V mws[BLIM];
  __shared__ uint64_cu ncomb;
  __shared__ F vaffb;
  const auto g = this_thread_block();
  if (t==0) ncomb = 0;
  for (K u=NB+b; u<NE; u+=G) {
    if (t==0) vaffb = vaff[u];
    __syncthreads();
    if (!vaffb) continue;
    K d = vcom[u];
    // Scan communities connected to u.
    auto mcw = rakLowmemBmScanCommunitiesCud<false>(xoff, xedg, xwei, vcom, u, d, O(t), O(BLIM));
    mcs[t] = mcw.first;
    mws[t] = mcw.second;
    // Find best community for u.
    __syncthreads();
    sketchMaxGroupReduceCudU(mcs, mws, BLIM, g, t);
    K c = mcs[0];
    if (t==0) vaff[u] = F(0);  // Mark u as unaffected
    if (c==d) continue;
    if (PICKLESS && c>d) continue;  // Pick smaller community-id (to avoid community swaps)
    if (t==0) vcom[u] = c;
    if (t==0) ++ncomb;
    rakMarkNeighborsCudU(vaff, xoff, xedg, u, t, BLIM);
  }
  // Update number of changed vertices.
  if (t==0) atomicAdd(ncom, ncomb);
}


/**
 * Move each vertex to its best community, for Boyer-Moore voting algorithm, using block-per-vertex approach.
 * @tparam BLIM size of each thread block
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int BLIM=256, class O, class K, class V, class F>
inline void rakLowmemBmMoveIterationBlockCuU(uint64_cu *ncom, K *vcom, F *vaff, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  const int B = BLIM;
  const int G = gridSizeCu<true>(NE-NB, B, GRID_LIMIT_MAP_CUDA);
  rakLowmemBmMoveIterationBlockCukU<BLIM><<<G, B>>>(ncom, vcom, vaff, xoff, xedg, xwei, NB, NE, PICKLESS);
}


/**
 * Move each vertex to its best community, for Misra-Gries sketch, using group-per-vertex approach [kernel].
 * @tparam SLOTS number of slots in hashtable
 * @tparam BLIM size of each thread block
 * @tparam RESCAN rescan communities after populating sketch?
 * @tparam TRYWARP try warp-specific optimization?
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int SLOTS=8, int BLIM=32, bool RESCAN=false, bool TRYWARP=true, class O, class K, class V, class F>
void __global__ rakLowmemMgMoveIterationGroupCukU(uint64_cu *ncom, K *vcom, F *vaff, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  DEFINE_CUDA(t, b, B, G);
  constexpr int  PLIM    = BLIM/SLOTS;
  constexpr bool USEWARP = TRYWARP && SLOTS<=32;
  __shared__ K   mcs[BLIM];
  __shared__ V   mws[BLIM];
  __shared__ int has[USEWARP? 1 : PLIM];
  __shared__ uint64_cu ncomb[PLIM];
  __shared__ F vaffb[PLIM];
  const auto g = tiled_partition<SLOTS>(this_thread_block());
  const int  s = t % SLOTS;
  const int  p = t / SLOTS;
  K* const pmcs = &mcs[p*SLOTS];
  V* const pmws = &mws[p*SLOTS];
  if (s==0) ncomb[p] = 0;
  for (K u=NB+b*PLIM+p; u<NE; u+=G*PLIM) {
    if (s==0) vaffb[p] = vaff[u];
    g.sync();
    if (!vaffb[p]) continue;
    K d = vcom[u];
    // Scan communities connected to u.
    sketchClearCudU<false, SLOTS>(pmws, s);
    g.sync();
    rakLowmemMgScanCommunitiesCudU<false, false, USEWARP>(pmcs, pmws, has+p, xoff, xedg, xwei, vcom, u, g, s, O(0), O(1));
    g.sync();
    // Rescan communities if necessary.
    if (RESCAN) {
      sketchClearCudU<false, SLOTS>(pmws, s);
      g.sync();
      rakLowmemMgRescanCommunitiesCudU<false, false>(pmcs, pmws, xoff, xedg, xwei, vcom, u, s, O(0), O(1));
      g.sync();
    }
    // Find best community for u.
    sketchMaxGroupReduceCudU(pmcs, pmws, SLOTS, g, s);
    g.sync();
    if (s==0) vaff[u] = F(0);  // Mark u as unaffected
    if  (!pmws[0]) continue;    // No community found
    K c = pmcs[0];              // Best community
    if (c==d) continue;
    if (PICKLESS && c>d) continue;  // Pick smaller community-id (to avoid community swaps)
    // Change community of u.
    if (s==0) vcom[u] = c;
    if (s==0) ++ncomb[p];
    rakMarkNeighborsCudU(vaff, xoff, xedg, u, s, PLIM);
  }
  // Update number of changed vertices.
  if (s==0) atomicAdd(ncom, ncomb[p]);
}


/**
 * Move each vertex to its best community, for Misra-Gries sketch, using block-per-vertex approach.
 * @tparam SLOTS number of slots in hashtable
 * @tparam BLIM size of each thread block
 * @tparam RESCAN rescan communities after populating sketch?
 * @tparam TRYWARP try warp-specific optimization?
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int SLOTS=8, int BLIM=32, bool RESCAN=false, bool TRYWARP=true, class O, class K, class V, class F>
inline void rakLowmemMgMoveIterationGroupCuU(uint64_cu *ncom, K *vcom, F *vaff, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  constexpr int PLIM = BLIM/SLOTS;
  const int B = BLIM;
  const int G = gridSizeCu(NE-NB, PLIM, GRID_LIMIT_MAP_CUDA);
  rakLowmemMgMoveIterationGroupCukU<SLOTS, BLIM, RESCAN, TRYWARP><<<G, B>>>(ncom, vcom, vaff, xoff, xedg, xwei, NB, NE, PICKLESS);
}


/**
 * Move each vertex to its best community, for Misra-Gries sketch, using block-per-vertex approach [kernel].
 * @tparam SLOTS number of slots in hashtable
 * @tparam BLIM size of each thread block
 * @tparam RESCAN rescan communities after populating sketch?
 * @tparam TRYWARP try warp-specific optimization?
 * @tparam TRYMERGE try merging separate sketches?
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int SLOTS=8, int BLIM=256, bool RESCAN=false, bool TRYWARP=true, bool TRYMERGE=false, class O, class K, class V, class F>
void __global__ rakLowmemMgMoveIterationBlockCukU(uint64_cu *ncom, K *vcom, F *vaff, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  DEFINE_CUDA(t, b, B, G);
  constexpr int  PLIM = BLIM/SLOTS;
  constexpr bool USEWARP     =  TRYWARP  && SLOTS<=32;
  constexpr bool USEMERGE    =  TRYMERGE && PLIM > 1;
  constexpr bool SHARED      = !USEMERGE && SLOTS < BLIM;
  constexpr bool SHAREDMERGE =  USEMERGE && PLIM > 2;
  __shared__ K mcs[USEMERGE? BLIM : SLOTS];
  __shared__ V mws[USEMERGE? BLIM : SLOTS];
  __shared__ int has[USEWARP? 1 : PLIM];
  __shared__ uint64_cu ncomb;
  __shared__ F vaffb;
  const auto g = tiled_partition<SLOTS>(this_thread_block());
  const int  s = t % SLOTS;
  const int  p = t / SLOTS;
  K* const pmcs = USEMERGE? &mcs[p*SLOTS] : mcs;
  V* const pmws = USEMERGE? &mws[p*SLOTS] : mws;
  if (t==0) ncomb = 0;
  for (K u=NB+b; u<NE; u+=G) {
    if (t==0) vaffb = vaff[u];
    __syncthreads();
    if (!vaffb) continue;
    K d = vcom[u];
    // Scan communities connected to u.
    sketchClearCudU<SHARED, SLOTS>(mws, t);
    __syncthreads();
    rakLowmemMgScanCommunitiesCudU<false, SHARED, USEWARP>(pmcs, pmws, has+p, xoff, xedg, xwei, vcom, u, g, s, O(p), O(PLIM));
    __syncthreads();
    // Merge sketches if necessary.
    if (USEMERGE && p>0) {
      if (!USEWARP)   sketchMergeCudU<SHAREDMERGE>(mcs, mws, has+p, pmcs, pmws, g, s);
      else sketchMergeWarpCudU<SHAREDMERGE, SLOTS>(mcs, mws, pmcs, pmws, g, s);
    }
    if (USEMERGE) __syncthreads();
    // Rescan communities if necessary.
    if (RESCAN) {
      sketchClearCudU<SHARED, SLOTS>(mws, t);
      __syncthreads();
      rakLowmemMgRescanCommunitiesCudU<false, SHARED>(mcs, mws, xoff, xedg, xwei, vcom, u, s, O(t), O(BLIM));
      __syncthreads();
    }
    // Find best community for u.
    sketchMaxBlockReduceCudU(mcs, mws, SLOTS, t);
    __syncthreads();
    if (t==0) vaff[u] = F(0);  // Mark u as unaffected
    if  (!mws[0]) continue;    // No community found
    K c = mcs[0];              // Best community
    if (c==d) continue;
    if (PICKLESS && c>d) continue;  // Pick smaller community-id (to avoid community swaps)
    // Change community of u.
    if (t==0) vcom[u] = c;
    if (t==0) ++ncomb;
    rakMarkNeighborsCudU(vaff, xoff, xedg, u, t, BLIM);
  }
  // Update number of changed vertices.
  if (t==0) atomicAdd(ncom, ncomb);
}


/**
 * Move each vertex to its best community, for Misra-Gries sketch, using block-per-vertex approach.
 * @tparam SLOTS number of slots in hashtable
 * @tparam BLIM size of each thread block
 * @tparam RESCAN rescan communities after populating sketch?
 * @tparam TRYWARP try warp-specific optimization?
 * @tparam TRYMERGE try merging separate sketches?
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int SLOTS=8, int BLIM=256, bool RESCAN=false, bool TRYWARP=true, bool TRYMERGE=false, class O, class K, class V, class F>
inline void rakLowmemMgMoveIterationBlockCuU(uint64_cu *ncom, K *vcom, F *vaff, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  const int B = BLIM;
  const int G = gridSizeCu<true>(NE-NB, B, GRID_LIMIT_MAP_CUDA);
  rakLowmemMgMoveIterationBlockCukU<SLOTS, BLIM, RESCAN, TRYWARP, TRYMERGE><<<G, B>>>(ncom, vcom, vaff, xoff, xedg, xwei, NB, NE, PICKLESS);
}
#pragma endregion




#pragma region COMPUTATION LOOP
/**
 * Perform RAK iterations.
 * @tparam SLOTS number of slots in hashtable
 * @tparam RESCAN rescan communities after populating sketch?
 * @tparam TRYWARP try warp-specific optimization?
 * @tparam TRYMERGE try merging separate sketches?
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param N number of vertices
 * @param NL number of low-degree vertices
 * @param E tolerance for convergence [0.05]
 * @param L maximum number of iterations [20]
 * @returns number of iterations performed
 */
template <int SLOTS=8, bool RESCAN=false, bool TRYWARP=true, bool TRYMERGE=true, class O, class K, class V, class F>
inline int rakLowmemLoopCuU(uint64_cu *ncom, K *vcom, F *vaff, const O *xoff, const K *xedg, const V *xwei, K N, K NL, double E, int L) {
  int l = 0;
  uint64_cu n = 0;
  const int PICKSTEP = 8;
  const K NH = N - NL;
  while (l<L) {
    bool PICKLESS = l % PICKSTEP == 0;
    fillValueCuW(ncom, 1, uint64_cu());
    if (SLOTS==1) {
      if (NL) rakLowmemBmMoveIterationThreadCuU<32>(ncom, vcom, vaff, xoff, xedg, xwei, K(), NL, PICKLESS);
      if (NH) rakLowmemBmMoveIterationBlockCuU<256>(ncom, vcom, vaff, xoff, xedg, xwei, NL,  N,  PICKLESS);
    }
    else {
      if (NL) rakLowmemMgMoveIterationGroupCuU<SLOTS, 32,  RESCAN, TRYWARP>          (ncom, vcom, vaff, xoff, xedg, xwei, K(), NL, PICKLESS);
      if (NH) rakLowmemMgMoveIterationBlockCuU<SLOTS, 256, RESCAN, TRYWARP, TRYMERGE>(ncom, vcom, vaff, xoff, xedg, xwei, NL,  N,  PICKLESS);
    }
    TRY_CUDA( cudaMemcpy(&n, ncom, sizeof(uint64_cu), cudaMemcpyDeviceToHost) ); ++l;
    if (!PICKLESS && double(n)/N <= E) break;
  }
  return l;
}
#pragma endregion




#pragma region PARTITION
/**
 * Partition vertices into low and high-degree sets.
 * @param ks vertex keys (updated)
 * @param x original graph
 * @tparam DLOW low-degree threshold [128]
 * @tparam DVLOW very low-degree threshold [4]
 * @returns number of low-degree vertices
 */
template <class G, class K>
inline size_t rakLowmemPartitionVerticesCudaU(vector<K>& ks, const G& x, size_t DLOW=128, size_t DVLOW=4) {
  const size_t SWITCH_LIMIT = 64;  // Avoid switching if number of vertices < SWITCH_LIMIT
  size_t N = ks.size();
  auto  kb = ks.begin(), ke = ks.end();
  auto  fl = [&](K v) { return x.degree(v) < DVLOW; };
  auto  fm = [&](K v) { return x.degree(v) < DLOW; };
  auto  il = partition(kb, ke, fl);
  auto  im = partition(il, ke, fm);
  size_t NK = distance(kb, il);
  size_t NL = distance(il, im);
  if (NK < SWITCH_LIMIT) { NL += NK; NK = 0; }
  if (NL < SWITCH_LIMIT) NL = 0;
  return NK + NL;
}
#pragma endregion




#pragma region ENVIRONMENT SETUP
/**
 * Setup and perform the RAK algorithm.
 * @tparam SLOTS number of slots in hashtable
 * @tparam RESCAN rescan communities after populating sketch?
 * @tparam TRYWARP try warp-specific optimization?
 * @tparam TRYMERGE try merging separate sketches?
 * @param x original graph
 * @param o rak options
 * @param fi initialzing community membership (vcomD)
 * @param fm marking affected vertices (vaffD)
 * @returns rak result
 */
template <int SLOTS=8, bool RESCAN=false, bool TRYWARP=true, bool TRYMERGE=true, class G, class FI, class FM>
inline auto rakLowmemInvokeCuda(const G& x, const RakOptions& o, FI fi, FM fm) {
  using K = typename G::key_type;
  using V = typename G::edge_value_type;
  using O = uint32_t;
  using F = char;
  // Get graph properties.
  size_t S = x.span();
  size_t N = x.order();
  size_t M = x.size();
  // Get RAK options.
  int    L = o.maxIterations, l = 0;
  double E = o.tolerance;
  // Allocate buffers.
  vector<O> xoff(N+1);  // CSR offsets array
  vector<K> xedg(M);    // CSR edge keys array
  vector<V> xwei(M);    // CSR edge values array
  vector<K> vcom(S), vcomc(N);  // Community membership of each vertex, compressed
  O *xoffD = nullptr;  // CSR offsets array [device]
  K *xedgD = nullptr;  // CSR edge keys array [device]
  V *xweiD = nullptr;  // CSR edge values array [device]
  K *vcomD = nullptr;  // Community membership [device]
  F *vaffD = nullptr;  // Affected vertex flag [device]
  uint64_cu *ncomD = nullptr;  // Number of changed vertices [device]
  // Partition vertices into low-degree and high-degree sets.
  vector<K> ks = vertexKeys(x);
  size_t NL = rakLowmemPartitionVerticesCudaU(ks, x, 128, 4);
  // Obtain data for CSR.
  csrCreateOffsetsW (xoff, x, ks);
  csrCreateEdgeKeysW(xedg, x, ks);
  csrCreateEdgeValuesW(xwei, x, ks);
  // Allocate device memory.
  TRY_CUDA( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY_CUDA( cudaMalloc(&xoffD, (N+1) * sizeof(O)) );
  TRY_CUDA( cudaMalloc(&xedgD,  M    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&xweiD,  M    * sizeof(V)) );
  // Measure initial memory usage.
  float m0 = measureMemoryUsageCu();
  // Allocate device memory.
  TRY_CUDA( cudaMalloc(&vcomD,  N    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&vaffD,  N    * sizeof(F)) );
  TRY_CUDA( cudaMalloc(&ncomD,  1    * sizeof(uint64_cu)) );
  // Copy data to device.
  TRY_CUDA( cudaMemcpy(xoffD, xoff.data(), (N+1) * sizeof(O), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(xedgD, xedg.data(),  M    * sizeof(K), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(xweiD, xwei.data(),  M    * sizeof(V), cudaMemcpyHostToDevice) );
  // Measure memory usage after allocation.
  float m1 = measureMemoryUsageCu();
  // Perform RAK algorithm on device.
  float tm = 0, ti = 0;
  float t  = measureDuration([&]() {
    // Setup initial community membership.
    ti += measureDuration([&]() { fi(vcomD, ks); });
    // Mark initial affected vertices.
    tm += measureDuration([&]() { fm(vaffD, ks); });
    // Perform RAK iterations.
    l = rakLowmemLoopCuU<SLOTS, RESCAN, TRYWARP, TRYMERGE>(ncomD, vcomD, vaffD, xoffD, xedgD, xweiD, K(N), K(NL), E, L);
  }, o.repeat);
  // Obtain final community membership.
  TRY_CUDA( cudaMemcpy(vcomc.data(), vcomD, N * sizeof(K), cudaMemcpyDeviceToHost) );
  scatterValuesOmpW(vcom, vcomc, ks);
  // Free device memory.
  TRY_CUDA( cudaFree(xoffD) );
  TRY_CUDA( cudaFree(xedgD) );
  TRY_CUDA( cudaFree(xweiD) );
  TRY_CUDA( cudaFree(vcomD) );
  TRY_CUDA( cudaFree(vaffD) );
  TRY_CUDA( cudaFree(ncomD) );
  return RakResult<K>(vcom, l, t, tm/o.repeat, ti/o.repeat, m1-m0);
}
#pragma endregion




#pragma region STATIC
/**
 * Obtain the community membership of each vertex with Static RAK.
 * @tparam SLOTS number of slots in hashtable
 * @tparam RESCAN rescan communities after populating sketch?
 * @tparam TRYWARP try warp-specific optimization?
 * @tparam TRYMERGE try merging separate sketches?
 * @param x original graph
 * @param o rak options
 * @returns rak result
 */
template <int SLOTS=8, bool RESCAN=false, bool TRYWARP=true, bool TRYMERGE=true, class G>
inline auto rakLowmemStaticCuda(const G& x, const RakOptions& o={}) {
  using  K = typename G::key_type;
  using  F = char;
  size_t N = x.order();
  auto  fi = [&](K *vcomD, const auto& ks) { rakInitializeCuW(vcomD, K(), K(N)); };
  auto  fm = [&](F *vaffD, const auto& ks) { fillValueCuW(vaffD, N, F(1)); };
  return rakLowmemInvokeCuda<SLOTS, RESCAN, TRYWARP, TRYMERGE>(x, o, fi, fm);
}
#pragma endregion
#pragma endregion
