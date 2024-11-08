#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "rak.hxx"
#include "hashtableCuda.hxx"

using std::vector;
using std::count_if;
using std::partition;




#pragma region METHODS
#pragma region INITIALIZE
/**
 * Initialize communities such that each vertex is its own community [kernel].
 * @param vcom community each vertex belongs to (output)
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K>
void __global__ rakInitializeCukW(K *vcom, K NB, K NE) {
  DEFINE_CUDA(t, b, B, G);
  for (K u=NB+B*b+t; u<NE; u+=G*B)
    vcom[u] = u;
}


/**
 * Initialize communities such that each vertex is its own community.
 * @param vcom community each vertex belongs to (output)
 * @param NB begin vertes (inclusive)
 * @param NE end vertex (exclusive)
 */
template <class K>
inline void rakInitializeCuW(K *vcom, K NB, K NE) {
  const int B = blockSizeCu(NE-NB,   BLOCK_LIMIT_MAP_CUDA);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  rakInitializeCukW<<<G, B>>>(vcom, NB, NE);
}
#pragma endregion




#pragma region CHOOSE COMMUNITY
/**
 * Scan communities connected to a vertex [device function].
 * @tparam SELF include self-loops?
 * @tparam BLOCK called from a thread block?
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @param hk hashtable keys (updated)
 * @param hv hashtable values (updated)
 * @param H capacity of hashtable (prime)
 * @param T secondary prime (>H)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 * @param i start index
 * @param DI index stride
 */
template <bool SELF=false, bool BLOCK=false, int HTYPE=3, class O, class K, class V, class W>
inline void __device__ rakScanCommunitiesCudU(K *hk, W *hv, size_t H, size_t T, const O *xoff, const K *xedg, const V *xwei, K u, const K *vcom, size_t i, size_t DI) {
  size_t EO = xoff[u];
  size_t EN = xoff[u+1] - xoff[u];
  for (; i<EN; i+=DI) {
    K v = xedg[EO+i];
    W w = xwei[EO+i];
    K c = vcom[v];
    if (!SELF && u==v) continue;
    hashtableAccumulateCudU<BLOCK, HTYPE>(hk, hv, H, T, c+1, w);
  }
}
#pragma endregion




#pragma region MOVE ITERATION
/**
 * Mark out-neighbors of a vertex as affected [device function].
 * @param vaff vertex affected flags (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param u given vertex
 * @param i start index
 * @param DI index stride
 */
template <class O, class K, class F>
inline void __device__ rakMarkNeighborsCudU(F *vaff, const O *xoff, const K *xedg, K u, size_t i, size_t DI) {
  size_t EO = xoff[u];
  size_t EN = xoff[u+1] - xoff[u];
  for (; i<EN; i+=DI) {
    K v = xedg[EO+i];
    vaff[v] = F(1);  // Use two (synchronous) buffers?
  }
}


/**
 * Move each vertex to its best community, using thread-per-vertex approach [kernel].
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam BLIM maximum number of threads per block
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int HTYPE=3, int BLIM=32, class O, class K, class V, class W, class F>
void __global__ rakMoveIterationThreadCukU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  DEFINE_CUDA(t, b, B, G);
  __shared__ uint64_cu ncomb[BLIM];
  // const int DMAX = BLIM;
  // K shrk[2*DMAX];
  // W shrw[2*DMAX];
  ncomb[t] = 0;
  for (K u=NB+B*b+t; u<NE; u+=G*B) {
    if (!vaff[u]) continue;
    // Scan communities connected to u.
    K d = vcom[u];
    size_t EO = xoff[u];
    size_t EN = xoff[u+1] - xoff[u];
    size_t H  = nextPow2Cud(EN) - 1;
    size_t T  = nextPow2Cud(H)  - 1;
    K *hk = bufk + 2*EO;  // shrk;
    W *hv = bufw + 2*EO;  // shrw;
    hashtableClearCudW(hk, hv, H, 0, 1);
    rakScanCommunitiesCudU<false, false, HTYPE>(hk, hv, H, T, xoff, xedg, xwei, u, vcom, 0, 1);
    // Find best community for u.
    hashtableMaxCudU(hk, hv, H, 0, 1);
    vaff[u] = F(0);         // Mark u as unaffected (Use two buffers?)
    if  (!hk[0]) continue;  // No community found
    K c = hk[0] - 1;        // Best community
    if (c==d) continue;
    if (PICKLESS && c>d) continue;  // Pick smaller community-id (to avoid community swaps)
    // Change community of u.
    vcom[u] = c; ++ncomb[t];
    rakMarkNeighborsCudU(vaff, xoff, xedg, u, 0, 1);
  }
  // Update number of changed vertices.
  __syncthreads();
  sumValuesBlockReduceCudU(ncomb, B, t);
  if (t==0) atomicAdd(ncom, ncomb[0]);
}


/**
 * Move each vertex to its best community, using thread-per-vertex approach.
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam BLIM maximum number of threads per block
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int HTYPE=3, int BLIM=32, class O, class K, class V, class W, class F>
inline void rakMoveIterationThreadCuU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  const int B = blockSizeCu(NE-NB, BLIM);
  const int G = gridSizeCu (NE-NB, B, GRID_LIMIT_MAP_CUDA);
  rakMoveIterationThreadCukU<HTYPE, BLIM><<<G, B>>>(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, NB, NE, PICKLESS);
}


/**
 * Move each vertex to its best community, using block-per-vertex approach [kernel].
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam BLIM maximum number of threads per block
 * @param ncom number of changed vertices (updated)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int HTYPE=3, int BLIM=128, class O, class K, class V, class W, class F>
void __global__ rakMoveIterationBlockCukU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  DEFINE_CUDA(t, b, B, G);
  // const int DMAX = BLIM;
  // __shared__ K shrk[2*DMAX];
  // __shared__ W shrw[2*DMAX];
  uint64_cu ncomb = 0;
  __shared__ bool vaffb;
  for (K u=NB+b; u<NE; u+=G) {
    if (t==0) vaffb = vaff[u];
    __syncthreads();
    if (!vaffb) continue;
    // if (!vaff[u]) continue;
    // Scan communities connected to u.
    K d = vcom[u];
    size_t EO = xoff[u];
    size_t EN = xoff[u+1] - xoff[u];
    size_t H  = nextPow2Cud(EN) - 1;
    size_t T  = nextPow2Cud(H)  - 1;
    K *hk = bufk + 2*EO;  // EN<=DMAX? shrk : bufk + 2*EO;
    W *hv = bufw + 2*EO;  // EN<=DMAX? shrw : bufw + 2*EO;
    hashtableClearCudW(hk, hv, H, t, B);
    __syncthreads();
    rakScanCommunitiesCudU<false, true, HTYPE>(hk, hv, H, T, xoff, xedg, xwei, u, vcom, t, B);
    __syncthreads();
    // Find best community for u.
    hashtableMaxCudU<true>(hk, hv, H, t, B);
    __syncthreads();
    if (t==0) vaff[u] = F(0);  // Mark u as unaffected (Use two buffers?)
    if  (!hk[0]) continue;     // No community found
    K c = hk[0] - 1;           // Best community
    if (c==d) continue;
    if (PICKLESS && c>d) continue;  // Pick smaller community-id (to avoid community swaps)
    // Change community of u.
    if (t==0) vcom[u] = c;
    if (t==0) ++ncomb;
    rakMarkNeighborsCudU(vaff, xoff, xedg, u, t, B);
  }
  // Update number of changed vertices.
  if (t==0) atomicAdd((uint64_cu*) ncom, ncomb);
}


/**
 * Move each vertex to its best community, using block-per-vertex approach.
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam BLIM maximum number of threads per block
 * @param ncom number of changed vertices (output)
 * @param vcom community each vertex belongs to (updated)
 * @param vaff vertex affected flags (updated)
 * @param bufk buffer for hashtable keys (updated)
 * @param bufw buffer for hashtable values (updated)
 * @param xoff offsets of original graph
 * @param xedg edge keys of original graph
 * @param xwei edge values of original graph
 * @param NB begin vertex (inclusive)
 * @param NE end vertex (exclusive)
 * @param PICKLESS allow only picking smaller community id?
 */
template <int HTYPE=3, int BLIM=128, class O, class K, class V, class W, class F>
inline void rakMoveIterationBlockCuU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K NB, K NE, bool PICKLESS) {
  const int B = blockSizeCu<true>(NE-NB, BLIM);
  const int G = gridSizeCu <true>(NE-NB, B, GRID_LIMIT_MAP_CUDA);
  rakMoveIterationBlockCukU<HTYPE, BLIM><<<G, B>>>(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, NB, NE, PICKLESS);
}
#pragma endregion




#pragma region COMPUTATION LOOP
/**
 * Perform RAK iterations.
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
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
 * @param E tolerance for convergence [0.05]
 * @param L maximum number of iterations [20]
 * @returns number of iterations performed
 */
template <int HTYPE=3, class O, class K, class V, class W, class F>
inline int rakLoopCuU(uint64_cu *ncom, K *vcom, F *vaff, K *bufk, W *bufw, const O *xoff, const K *xedg, const V *xwei, K N, K NL, double E, int L) {
  int l = 0;
  uint64_cu n = 0;
  const int PICKSTEP = 4;
  while (l<L) {
    bool PICKLESS = l % PICKSTEP == 0;
    fillValueCuW(ncom, 1, uint64_cu());
    rakMoveIterationThreadCuU<HTYPE>(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, K(), NL, PICKLESS);
    rakMoveIterationBlockCuU <HTYPE>(ncom, vcom, vaff, bufk, bufw, xoff, xedg, xwei, NL,  N,  PICKLESS); ++l;
    // rakCrossCheckCuU(ncom, vdom, vcom, K(), N); swap(vdom, vcom);
    TRY_CUDA( cudaMemcpy(&n, ncom, sizeof(uint64_cu), cudaMemcpyDeviceToHost) );
    if (!PICKLESS && double(n)/N <= E) break;
  }
  return l;
}
#pragma endregion




#pragma region PARTITION
/**
 * Partition vertices into low-degree and high-degree sets.
 * @param ks vertex keys (updated)
 * @param x original graph
 * @returns number of low-degree vertices
 */
template <class G, class K>
inline size_t rakPartitionVerticesCudaU(vector<K>& ks, const G& x) {
  const K SWITCH_DEGREE = 32;  // Switch to block-per-vertex approach if degree >= SWITCH_DEGREE
  const K SWITCH_LIMIT  = 64;  // Avoid switching if number of vertices < SWITCH_LIMIT
  size_t N = ks.size();
  auto  kb = ks.begin(), ke = ks.end();
  auto  ft = [&](K v) { return x.degree(v) < SWITCH_DEGREE; };
  partition(kb, ke, ft);
  size_t n = count_if(kb, ke, ft);
  if (n   < SWITCH_LIMIT) n = 0;
  if (N-n < SWITCH_LIMIT) n = N;
  return n;
}
#pragma endregion




#pragma region ENVIRONMENT SETUP
/**
 * Setup and perform the RAK algorithm.
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam HWEIGHT hashtable weight type
 * @param x original graph
 * @param o rak options
 * @param fi initialzing community membership (vcomD)
 * @param fm marking affected vertices (vaffD)
 * @returns rak result
 */
template <int HTYPE=3, class HWEIGHT=float, class G, class FI, class FM>
inline auto rakInvokeCuda(const G& x, const RakOptions& o, FI fi, FM fm) {
  using K = typename G::key_type;
  using V = typename G::edge_value_type;
  using W = HWEIGHT;
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
  K *bufkD = nullptr;  // Buffer for hashtable keys [device]
  W *bufwD = nullptr;  // Buffer for hashtable values [device]
  uint64_cu *ncomD = nullptr;  // Number of changed vertices [device]
  // Partition vertices into low-degree and high-degree sets.
  vector<K> ks = vertexKeys(x);
  size_t    NL = rakPartitionVerticesCudaU(ks, x);
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
  TRY_CUDA( cudaMalloc(&bufkD, (2*M) * sizeof(K)) );
  TRY_CUDA( cudaMalloc(&bufwD, (2*M) * sizeof(W)) );
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
    l = rakLoopCuU<HTYPE>(ncomD, vcomD, vaffD, bufkD, bufwD, xoffD, xedgD, xweiD, K(N), K(NL), E, L);
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
  TRY_CUDA( cudaFree(bufkD) );
  TRY_CUDA( cudaFree(bufwD) );
  TRY_CUDA( cudaFree(ncomD) );
  return RakResult<K>(vcom, l, t, tm/o.repeat, ti/o.repeat);
}
#pragma endregion




#pragma region STATIC
/**
 * Obtain the community membership of each vertex with Static RAK.
 * @tparam HTYPE hashtable type (0: linear, 1: quadratic, 2: double, 3: quadritic + double)
 * @tparam HWEIGHT hashtable weight type
 * @param x original graph
 * @param o rak options
 * @returns rak result
 */
template <int HTYPE=3, class HWEIGHT=float, class G>
inline auto rakStaticCuda(const G& x, const RakOptions& o={}) {
  using  K = typename G::key_type;
  using  F = char;
  size_t N = x.order();
  auto  fi = [&](K *vcomD, const auto& ks) { rakInitializeCuW(vcomD, K(), K(N)); };
  auto  fm = [&](F *vaffD, const auto& ks) { fillValueCuW(vaffD, N, F(1)); };
  return rakInvokeCuda<HTYPE, HWEIGHT>(x, o, fi, fm);
}
#pragma endregion
#pragma endregion
