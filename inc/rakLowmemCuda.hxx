#pragma once
#include <cstdint>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "rak.hxx"
#include "rakCuda.hxx"

using std::vector;
using std::make_pair;




#pragma region METHODS
#pragma region SCAN COMMUNITIES
/**
 * Scan communities connected to a vertex [device function].
 * @tparam SELF include self-loops?
 * @tparam SLOTS number of slots in the sketch
 * @tparam BLIM size of each thread block
 * @tparam WARP use warp-specific optimization?
 * @param mcs majority linked communities (updated)
 * @param mws total edge weight to each majority community (updated)
 * @param has is community already in the list? (scratch)
 * @param frs a free slot in the list (scratch)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param vcom community each vertex belongs to
 * @param u given vertex
 * @param i start index
 * @param DI index stride
 */
template <bool SELF=false, int SLOTS=8, int BLIM=128, bool WARP=true, class O, class K, class V, class TG>
inline void __device__ rakLowmemScanCommunitiesCudU(K *mcs, V *mws, int *has, int *frs, const O *xoff, const K *xedg, const V *xwei, const K *vcom, K u, const TG& g, int s, O i, O DI) {
  constexpr bool SHARED = SLOTS < BLIM;
  O EO = xoff[u];
  O EN = xoff[u+1] - xoff[u];
  for (; i<EN; i+=DI) {
    K v = xedg[EO+i];
    V w = xwei[EO+i];
    K c = vcom[v];
    if (!SELF && u==v) continue;
    if (SLOTS==32 && WARP) sketchAccumulateWarpCudU(mcs, mws, c, w, g, s);
    else sketchAccumulateCudU<SHARED>(mcs, mws, has, frs, c, w, g, s);
  }
}
#pragma endregion




#pragma region MOVE ITERATION
/**
 * Move each vertex to its best community, using group-per-vertex approach [kernel].
 * @tparam SLOTS number of slots in hashtable
 * @tparam BLIM size of each thread block
 * @tparam WARP use warp-specific optimization?
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
template <int SLOTS=8, int BLIM=128, bool WARP=true, class O, class K, class V, class F>
void __global__ rakLowmemMoveIterationGroupCukU(uint64_cu *ncom, K *vcom, F *vaff, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  DEFINE_CUDA(t, b, B, G);
  constexpr int PLIM = BLIM/SLOTS;
  __shared__ K mcs[BLIM];
  __shared__ V mws[BLIM];
  __shared__ int has[WARP? 1 : PLIM];
  __shared__ int frs[WARP? 1 : PLIM];
  __shared__ uint64_cu ncomb[PLIM];
  const auto g = tiled_partition<SLOTS>(this_thread_block());
  const int  s = t % SLOTS;
  const int  p = t / SLOTS;
  if (s==0) ncomb[p] = 0;
  for (K u=NB+b*PLIM+p; u<NE; u+=G*PLIM) {
    if (!vaff[u]) continue;
    K d = vcom[u];
    // Scan communities connected to u.
    sketchClearCudU<false, SLOTS>(mcs+p, mws+p, s);
    g.sync();
    rakLowmemScanCommunitiesCudU<false, SLOTS, BLIM, WARP>(mcs+p, mws+p, has+p, frs+p, xoff, xedg, xwei, vcom, u, g, s, 0, 1);
    g.sync();
    // Find best community for u.
    sketchMaxCudU(mcs+p, mws+p, SLOTS, s);
    g.sync();
    if (s==0) vaff[u] = F(0);  // Mark u as unaffected
    if  (!mws[p]) continue;    // No community found
    K c = mcs[p];              // Best community
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
 * Move each vertex to its best community, using group-per-vertex approach [kernel].
 * @tparam SLOTS number of slots in hashtable
 * @tparam BLIM size of each thread block
 * @tparam WARP use warp-specific optimization?
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
template <int SLOTS=8, int BLIM=128, bool WARP=true, class O, class K, class V, class F>
void __global__ rakLowmemMoveIterationGroupCuU(uint64_cu *ncom, K *vcom, F *vaff, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  constexpr int PLIM = BLIM/SLOTS;
  const int B = BLIM;
  const int G = gridSizeCu<true>(NE-NB, B*PLIM, GRID_LIMIT_MAP_CUDA);
  rakLowmemMoveIterationGroupCukU<SLOTS, BLIM, WARP><<<G, B>>>(ncom, vcom, vaff, xoff, xedg, xwei, NB, NE, PICKLESS);
}


/**
 * Move each vertex to its best community, using block-per-vertex approach [kernel].
 * @tparam SLOTS number of slots in hashtable
 * @tparam BLIM size of each thread block
 * @tparam WARP use warp-specific optimization?
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
template <int SLOTS=8, int BLIM=128, bool WARP=true, class O, class K, class V, class F>
void __global__ rakLowmemMoveIterationBlockCukU(uint64_cu *ncom, K *vcom, F *vaff, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  DEFINE_CUDA(t, b, B, G);
  constexpr int  PLIM   = BLIM/SLOTS;
  constexpr bool SHARED = SLOTS<BLIM;
  __shared__ K mcs[SLOTS];
  __shared__ V mws[SLOTS];
  __shared__ int has[WARP? 1 : PLIM];
  __shared__ int frs[WARP? 1 : PLIM];
  __shared__ uint64_cu ncomb;
  const auto g = tiled_partition<SLOTS>(this_thread_block());
  const int  s = t % SLOTS;
  const int  p = t / SLOTS;
  if (t==0) ncomb = 0;
  for (K u=NB+b; u<NE; u+=G) {
    if (!vaff[u]) continue;
    K d = vcom[u];
    // Scan communities connected to u.
    sketchClearCudU<SHARED, SLOTS>(mws, t);
    __syncthreads();
    rakLowmemScanCommunitiesCudU<false, SLOTS, BLIM, WARP>(mcs, mws, has+p, frs+p, xoff, xedg, xwei, vcom, u, g, s, p, PLIM);
    __syncthreads();
    // Find best community for u.
    sketchMaxCudU(mcs, mws, SLOTS, t);
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
 * Move each vertex to its best community, using block-per-vertex approach.
 * @tparam SLOTS number of slots in hashtable
 * @tparam BLIM size of each thread block
 * @tparam WARP use warp-specific optimization?
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
template <int SLOTS=8, int BLIM=128, bool WARP=true, class O, class K, class V, class F>
inline void rakLowmemMoveIterationBlockCuU(uint64_cu *ncom, K *vcom, F *vaff, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  const int B = BLIM;
  const int G = gridSizeCu<true>(NE-NB, B, GRID_LIMIT_MAP_CUDA);
  rakLowmemMoveIterationBlockCukU<SLOTS, BLIM, WARP><<<G, B>>>(ncom, vcom, vaff, xoff, xedg, xwei, NB, NE, PICKLESS);
}
#pragma endregion




#pragma region COMPUTATION LOOP
/**
 * Perform RAK iterations.
 * @tparam SLOTS number of slots in hashtable
 * @tparam WARP use warp-specific optimization?
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param N number of vertices
 * @param NL number of low-degree vertices
 * @param NM number of mid-degree vertices
 * @param E tolerance for convergence [0.05]
 * @param L maximum number of iterations [20]
 * @returns number of iterations performed
 */
template <int SLOTS=8, bool WARP=true, class O, class K, class V, class W, class F>
inline int rakLowmemLoopCuU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K N, K NL, K NM, double E, int L) {
  int l = 0;
  uint64_cu n = 0;
  const int PICKSTEP = 4;
  while (l<L) {
    bool PICKLESS = l % PICKSTEP == 0;
    fillValueCuW(ncom, 1, uint64_cu());
    rakLowmemMoveIterationGroupCuU<SLOTS,  32, WARP>(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, K(), NL,   PICKLESS);
    rakLowmemMoveIterationBlockCuU<SLOTS,  32, WARP>(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, NL, NL+NM, PICKLESS);
    rakLowmemMoveIterationBlockCuU<SLOTS, 128, WARP>(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, NL+NM, N,  PICKLESS); ++l;
    TRY_CUDA( cudaMemcpy(&n, ncom, sizeof(uint64_cu), cudaMemcpyDeviceToHost) );
    if (!PICKLESS && double(n)/N <= E) break;
  }
  return l;
}
#pragma endregion




#pragma region PARTITION
/**
 * Partition vertices into low, mid, and high-degree sets.
 * @param ks vertex keys (updated)
 * @param x original graph
 * @returns number of low, mid-degree vertices
 */
template <class G, class K>
inline auto rakLowmemPartitionVerticesCudaU(vector<K>& ks, const G& x) {
  const K SWITCH_DEGREEL = 4;   // Switch to group-per-vertex approach if degree < SWITCH_DEGREEL
  const K SWITCH_DEGREEM = 32;  // Switch to block-per-vertex approach if degree >= SWITCH_DEGREEM
  const K SWITCH_LIMIT   = 64;  // Avoid switching if number of vertices < SWITCH_LIMIT
  size_t N = ks.size();
  auto  kb = ks.begin(), ke = ks.end();
  auto  fl = [&](K v) { return x.degree(v) < SWITCH_DEGREEL; };
  auto  fm = [&](K v) { return x.degree(v) < SWITCH_DEGREEM; };
  auto  il = partition(kb, ke, ft);
  auto  im = partition(il, ke, fm);
  size_t NL = distance(kb, il);
  size_t NM = distance(il, im);
  if (NL < SWITCH_LIMIT) { NM += NL; NL = 0; }
  if (NM < SWITCH_LIMIT) NM = 0;
  return make_pair(NL, NM);
}
#pragma endregion




#pragma region ENVIRONMENT SETUP
/**
 * Setup and perform the RAK algorithm.
 * @tparam SLOTS number of slots in hashtable
 * @tparam WARP use warp-specific optimization?
 * @param x original graph
 * @param o rak options
 * @param fi initialzing community membership (vcomD)
 * @param fm marking affected vertices (vaffD)
 * @returns rak result
 */
template <int SLOTS=8, bool WARP=false, class G, class FI, class FM>
inline auto rakInvokeCuda(const G& x, const RakOptions& o, FI fi, FM fm) {
  using K = typename G::key_type;
  using V = typename G::edge_value_type;
  using O = uint32_t;
  using F = char;
  // Get graph properties.
  size_t S = x.span();
  size_t N = x.order();
  size_t M = x.size();
  int    R = reduceSizeCu(N);
  // Get RAK options.
  int    L = o.maxIterations, l = 0;
  double E = o.tolerance;
  // Allocate buffers.
  vector<O> xoff(N+1);  // CSR offsets array
  vector<K> xedg(M);    // CSR edge keys array
  vector<V> xwei(M);    // CSR edge values array
  vector<K> vcom(S), vcomc(N);
  vector<F> vaff(S), vaffc(N);
  O *xoffD = nullptr;  // CSR offsets array [device]
  K *xedgD = nullptr;  // CSR edge keys array [device]
  V *xweiD = nullptr;  // CSR edge values array [device]
  F *vaffD = nullptr;  // Affected vertex flag [device]
  K *vcomD = nullptr;  // Community membership [device]
  uint64_cu *ncomD = nullptr;  // Number of changed vertices [device]
  // Partition vertices into low-degree and high-degree sets.
  vector<K> ks = vertexKeys(x);
  auto [NL, NM] = rakLowmemPartitionVerticesCudaU(ks, x);
  // Obtain data for CSR.
  csrCreateOffsetsW (xoff, x, ks);
  csrCreateEdgeKeysW(xedg, x, ks);
  csrCreateEdgeValuesW(xwei, x, ks);
  // Allocate device memory.
  TRY_CUDA( cudaSetDeviceFlags(cudaDeviceMapHost) );
  TRY_CUDA( cudaMalloc(&xoffD, (N+1) * sizeof(O)) );
  TRY_CUDA( cudaMalloc(&xedgD,  M    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&xweiD,  M    * sizeof(V)) );
  TRY_CUDA( cudaMalloc(&vcomD,  N    * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&vaffD,  N    * sizeof(F)) );
  TRY_CUDA( cudaMalloc(&ncomD,  1    * sizeof(uint64_cu)) );
  // Copy data to device.
  TRY_CUDA( cudaMemcpy(xoffD, xoff.data(), (N+1) * sizeof(O), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(xedgD, xedg.data(),  M    * sizeof(K), cudaMemcpyHostToDevice) );
  TRY_CUDA( cudaMemcpy(xweiD, xwei.data(),  M    * sizeof(V), cudaMemcpyHostToDevice) );
  // Perform RAK algorithm on device.
  float tm = 0, ti = 0;
  float t  = measureDuration([&]() {
    // Setup initial community membership.
    ti += measureDuration([&]() { fi(vcomD, ks); });
    // Mark initial affected vertices.
    tm += measureDuration([&]() { fm(vaffD, ks); });
    // Perform RAK iterations.
    l = rakLowmemLoopCuU<SLOTS, WARP>(ncomD, vcomD, vaffD, xoffD, xedgD, xweiD, K(N), K(NL), K(NM), E, L);
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
  return RakResult<K>(vcom, l, t, tm/o.repeat, ti/o.repeat);
}
#pragma endregion




#pragma region STATIC
/**
 * Obtain the community membership of each vertex with Static RAK.
 * @tparam SLOTS number of slots in hashtable
 * @tparam WARP use warp-specific optimization?
 * @param x original graph
 * @param o rak options
 * @returns rak result
 */
template <int SLOTS=8, bool WARP=false, class G>
inline auto rakLowmemStaticCuda(const G& x, const RakOptions& o={}) {
  using  K = typename G::key_type;
  using  F = char;
  size_t N = x.order();
  auto  fi = [&](K *vcomD, const auto& ks) { rakInitializeCuW(vcomD, K(), K(N)); };
  auto  fm = [&](F *vaffD, const auto& ks) { fillValueCuW(vaffD, N, F(1)); };
  return rakLowmemInvokeCuda<SLOTS, WARP>(x, o, fi, fm);
}
#pragma endregion
#pragma endregion
