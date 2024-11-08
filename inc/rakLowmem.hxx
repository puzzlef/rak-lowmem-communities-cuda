#pragma once
#include <utility>
#include <array>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "rak.hxx"

using std::array;
using std::vector;
using std::make_pair;
using std::swap;
using std::max;




#pragma region METHODS
#pragma region HASHTABLES
/**
 * Allocate a number of hashtables.
 * @param mcs majority communities vertex u is linked to (temporary buffer, updated)
 * @param mws total edge weight from vertex u to community C (temporary buffer, updated)
 */
template <class K, class V, size_t SLOTS>
inline void rakLowmemAllocateHashtablesW(vector<array<K, SLOTS>*>& mcs, vector<array<V, SLOTS>*>& mws) {
  size_t N = mcs.size();
  for (size_t i=0; i<N; ++i) {
    mcs[i] = new array<K, SLOTS>();
    mws[i] = new array<V, SLOTS>();
  }
}


/**
 * Free a number of hashtables.
 * @param mcs majority communities vertex u is linked to (temporary buffer, updated)
 * @param mws total edge weight from vertex u to community C (temporary buffer, updated)
 */
template <class K, class V, size_t SLOTS>
inline void rakLowmemFreeHashtablesW(vector<array<K, SLOTS>*>& mcs, vector<array<V, SLOTS>*>& mws) {
  size_t N = mcs.size();
  for (size_t i=0; i<N; ++i) {
    delete mcs[i];
    delete mws[i];
  }
}
#pragma endregion




#pragma region CHOOSE COMMUNITY
/**
 * Scan an edge community connected to a vertex.
 * @param mcs majority communities vertex u is linked to (temporary buffer, updated)
 * @param mws total edge weight from vertex u to community C (temporary buffer, updated)
 * @param u given vertex
 * @param v outgoing edge vertex
 * @param w outgoing edge weight
 * @param vcom community each vertex belongs to
 */
template <bool SELF=false, class K, class V, size_t SLOTS>
inline void rakLowmemScanCommunityU(array<K, SLOTS>& mcs, array<V, SLOTS>& mws, K u, K v, V w, const vector<K>& vcom) {
  if (!SELF && u==v) return;
  K c = vcom[v];
  // Add edge weight to community.
  for (int i=0; i<SLOTS; ++i)
    mws[i] += mcs[i]==c? w : V();
  // Check if community is already in the list.
  int has = 0;
  for (int i=0; i<SLOTS; ++i)
    has |= mcs[i]==c? -1 : 0;
  if (has) return;
  // Find empty slot.
  int f = -1;
  for (int i=0; i<SLOTS; ++i)
    if (mws[i]==V()) f = i;
  // Add community to list.
  if (f>=0) {
    mcs[f] = c;
    mws[f] = w;
  }
  // Subtract edge weight from non-matching communities.
  else {
    for (int i=0; i<SLOTS; ++i)
      mws[i] = max(mws[i] - w, V());
  }
}


/**
 * Scan communities connected to a vertex.
 * @param mcs majority communities vertex u is linked to (temporary buffer, updated)
 * @param mws total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 */
template <bool SELF=false, class G, class K, class V, size_t SLOTS>
inline void rakLowmemScanCommunitiesW(array<K, SLOTS>& mcs, array<V, SLOTS>& mws, const G& x, K u, const vector<K>& vcom) {
  x.forEachEdge(u, [&](auto v, auto w) { rakLowmemScanCommunityU<SELF>(mcs, mws, u, v, w, vcom); });
}


/**
 * Scan communities connected to a vertex.
 * @param mcs communities vertex u is linked to (temporary buffer, updated)
 * @param mws total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 * @returns [majority community, total edge weight to scommunity]
 */
template <bool SELF=false, class G, class K>
inline auto rakLowmemScanCommunitiesMajority(const G& x, K u, const vector<K>& vcom) {
  using V = typename G::edge_value_type;
  K mc = K();
  V mw = V();
  x.forEachEdge(u, [&](auto v, auto w) {
    if (!SELF && u==v) return;
    K c = vcom[v];
    if (c==mc)     mw += w;
    else if (mw>w) mw -= w;
    else { mc = c; mw  = w; }
  });
  return make_pair(mc, mw);
}


/**
 * Clear communities scan data.
 * @param mws communities vertex u is linked to (updated)
 */
template <class V, size_t SLOTS>
inline void rakLowmemClearScanW(array<V, SLOTS>& mws) {
  for (int i=0; i<SLOTS; ++i)
    mws[i] = V();
}


/**
 * Choose connected community with best delta modularity.
 * @param mws total edge weight from vertex u to community C (updated)
 * @param x original graph
 * @param u given vertex
 * @param mcs majority communities vertex u is linked to
 * @param vcom community each vertex belongs to
 * @returns [best community, delta modularity]
 */
template <bool SELF=false, bool RESCAN=true, class G, class K, class V, size_t SLOTS>
inline auto rakLowmemChooseCommunityW(array<V, SLOTS>& mws, const G& x, K u, const array<K, SLOTS>& mcs, const vector<K>& vcom) {
  // Compute total edge weight to communities.
  if (RESCAN) {
    rakLowmemClearScanW(mws);
    x.forEachEdge(u, [&](auto v, auto w) {
      if (!SELF && u==v) return;
      K c = vcom[v];
      for (int i=0; i<SLOTS; ++i)
        mws[i] += mcs[i]==c? w : V();
    });
  }
  // Choose community with best weight.
  K cmax = K();
  V wmax = V();
  for (int i=0; i<SLOTS; ++i) {
    K c = mcs[i];
    V w = mws[i];
    if (w>wmax) { wmax = w; cmax = c; }
  }
  return make_pair(cmax, wmax);
}
#pragma endregion




#pragma region MOVE ITERATION
/**
 * Move each vertex to its best community.
 * @param vcom community each vertex belongs to (updated)
 * @param vaff is vertex affected (updated)
 * @param mcs majority communities vertex u is linked to (temporary buffer, updated)
 * @param mws total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param fa is vertex allowed to be updated? (u)
 * @returns number of changed vertices
 */
template <bool MULTI=true, bool RESCAN=true, class G, class K, class V, class F, size_t SLOTS, class FA>
inline size_t rakLowmemMoveIterationOmpW(vector<K>& vcom, vector<F>& vaff, vector<array<K, SLOTS>*>& mcs, vector<array<V, SLOTS>*>& mws, const G& x, FA fa) {
  size_t a = K();
  size_t S = x.span();
  #pragma omp parallel for schedule(dynamic, 2048) reduction(+:a)
  for (K u=0; u<S; ++u) {
    int t = omp_get_thread_num();
    if (!x.hasVertex(u)) continue;
    if (!fa(u) || !vaff[u]) continue;
    K d = vcom[u];
    if (MULTI) {
      rakLowmemClearScanW(*mws[t]);
      rakLowmemScanCommunitiesW(*mcs[t], *mws[t], x, u, vcom);
      auto [c, w] = rakLowmemChooseCommunityW<false, RESCAN>(*mws[t], x, u, *mcs[t], vcom);
      if (w>0 && c!=d) { vcom[u] = c; ++a; x.forEachEdgeKey(u, [&](auto v) { vaff[v] = F(1); }); }
    }
    else {
      auto [c, w] = rakLowmemScanCommunitiesMajority(x, u, vcom);
      if (w>0 && c!=d) { vcom[u] = c; ++a; x.forEachEdgeKey(u, [&](auto v) { vaff[v] = F(1); }); }
    }
    vaff[u] = F();
  }
  return a;
}
#pragma endregion




#pragma region ENVIRONMENT SETUP
/**
 * Setup and perform the RAK algorithm.
 * @param x original graph
 * @param o rak options
 * @param fi initialzing community membership (vcom)
 * @param fm marking affected vertices (vaff, vcs, vcout, vcom)
 * @param fa is vertex allowed to be updated? (u)
 * @returns rak result
 */
template <bool MULTI=true, bool RESCAN=true, size_t SLOTS=8, class G, class FI, class FM, class FA>
inline auto rakLowmemInvokeOmp(const G& x, const RakOptions& o, FI fi, FM fm, FA fa) {
  using K = typename G::key_type;
  using V = typename G::edge_value_type;
  using W = RAK_WEIGHT_TYPE;
  using F = char;
  int l = 0;
  int T = omp_get_max_threads();
  // Get graph properties.
  size_t S = x.span();
  size_t N = x.order();
  // Allocate buffers.
  vector<F> vaff(S);  // Affected vertex flag
  vector<K> vcom(S);  // Community membership
  vector<array<K, SLOTS>*> mcs(T);    // Hashtable keys
  vector<array<V, SLOTS>*> mws(T);  // Hashtable values
  rakLowmemAllocateHashtablesW(mcs, mws);
  // Perform RAK algorithm.
  float tm = 0, ti = 0;  // Time spent in different phases
  float t  = measureDuration([&]() {
    // Initialize community membership.
    ti += measureDuration([&]() { fi(vcom); });
    // Mark affected vertices.
    tm += measureDuration([&]() { fm(vaff, mcs, mws, vcom); });
    // Perform iterations.
    for (l=0; l<o.maxIterations;) {
      size_t n = rakLowmemMoveIterationOmpW<MULTI, RESCAN>(vcom, vaff, mcs, mws, x, fa); ++l;
      if (double(n)/N <= o.tolerance) break;
    }
  }, o.repeat);
  rakLowmemFreeHashtablesW(mcs, mws);
  return RakResult<K>(vcom, l, t, tm/o.repeat, ti/o.repeat);
}
#pragma endregion




#pragma region STATIC
/**
 * Obtain the community membership of each vertex with Static RAK.
 * @param x original graph
 * @param o rak options
 * @returns rak result
 */
template <bool MULTI=true, bool RESCAN=true, size_t SLOTS=8, class G>
inline auto rakLowmemStaticOmp(const G& x, const RakOptions& o={}) {
  using FLAG = char;
  auto fi = [&](auto& vcom) { rakInitializeOmpW(vcom, x); };
  auto fm = [ ](auto& vaff, auto& vcs, auto& vcout, const auto& vcom) { fillValueOmpU(vaff, FLAG(1)); };
  auto fa = [ ](auto u) { return true; };
  return rakLowmemInvokeOmp<MULTI, RESCAN, SLOTS>(x, o, fi, fm, fa);
}
#pragma endregion
#pragma endregion
