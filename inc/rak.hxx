#pragma once
#include <utility>
#include <vector>
#include <cstdint>
#include "_main.hxx"
#include <omp.h>

using std::vector;
using std::make_pair;
using std::move;




#pragma region TYPES
/**
 * Options for RAK algorithm.
 */
struct RakOptions {
  #pragma region DATA
  /** Number of times to repeat the algorithm [1]. */
  int repeat;
  /** Tolerance for convergence [0.05]. */
  double tolerance;
  /** Maximum number of iterations [20]. */
  int maxIterations;
  #pragma endregion


  #pragma region CONSTRUCTORS
  /**
   * Define options for RAK algorithm.
   * @param repeat number of times to repeat the algorithm [1]
   * @param tolerance tolerance for convergence [0.05]
   * @param maxIterations maximum number of iterations [20]
   */
  RakOptions(int repeat=1, double tolerance=0.05, int maxIterations=20) :
  repeat(repeat), tolerance(tolerance), maxIterations(maxIterations) {}
  #pragma endregion
};


/** Weight to be used in hashtable. */
#define RAK_WEIGHT_TYPE double




/**
 * Result of RAK algorithm.
 * @tparam K key type (vertex-id)
 */
template <class K>
struct RakResult {
  #pragma region DATA
  /** Community membership each vertex belongs to. */
  vector<K> membership;
  /** Number of iterations performed. */
  int iterations;
  /** Time spent in milliseconds. */
  float time;
  /** Time spent in milliseconds for initial marking of affected vertices. */
  float markingTime;
  /** Time spent in milliseconds for initializing community memberships. */
  float initializationTime;
  #pragma endregion


  #pragma region CONSTRUCTORS
  /**
   * Result of RAK algorithm.
   * @param membership community membership each vertex belongs to
   * @param iterations number of iterations performed
   * @param time time spent in milliseconds
   * @param markingTime time spent in milliseconds for initial marking of affected vertices
   * @param initializationTime time spent in initializing community memberships
   */
  RakResult(vector<K>&& membership, int iterations=0, float time=0, float markingTime=0, float initializationTime=0) :
  membership(membership), iterations(iterations), time(time), markingTime(markingTime), initializationTime(initializationTime) {}


  /**
   * Result of RAK algorithm.
   * @param membership community membership each vertex belongs to (moved)
   * @param iterations number of iterations performed
   * @param time time spent in milliseconds
   * @param markingTime time spent in milliseconds for initial marking of affected vertices
   * @param initializationTime time spent in initializing community memberships
   */
  RakResult(vector<K>& membership, int iterations=0, float time=0, float markingTime=0, float initializationTime=0) :
  membership(move(membership)), iterations(iterations), time(time), markingTime(markingTime), initializationTime(initializationTime) {}
  #pragma endregion
};
#pragma endregion




#pragma region METHODS
#pragma region HASHTABLES
/**
 * Allocate a number of hashtables.
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param S size of each hashtable
 */
template <class K, class W>
inline void rakAllocateHashtablesW(vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, size_t S) {
  size_t N = vcs.size();
  for (size_t i=0; i<N; ++i) {
    vcs[i]   = new vector<K>();
    vcout[i] = new vector<W>(S);
  }
}


/**
 * Free a number of hashtables.
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 */
template <class K, class W>
inline void rakFreeHashtablesW(vector<vector<K>*>& vcs, vector<vector<W>*>& vcout) {
  size_t N = vcs.size();
  for (size_t i=0; i<N; ++i) {
    delete vcs[i];
    delete vcout[i];
  }
}
#pragma endregion




#pragma region INITIALIZE
/**
 * Initialize communities such that each vertex is its own community.
 * @param vcom community each vertex belongs to (updated, must be initialized)
 * @param x original graph
 */
template <class G, class K>
inline void rakInitializeOmpW(vector<K>& vcom, const G& x) {
  size_t S = x.span();
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    vcom[u] = u;
  }
}
#pragma endregion




#pragma region CHOOSE COMMUNITY
/**
 * Scan an edge community connected to a vertex.
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param u given vertex
 * @param v outgoing edge vertex
 * @param w outgoing edge weight
 * @param vcom community each vertex belongs to
 */
template <bool SELF=false, class K, class V, class W>
inline void rakScanCommunityU(vector<K>& vcs, vector<W>& vcout, K u, K v, V w, const vector<K>& vcom) {
  if (!SELF && u==v) return;
  K c = vcom[v];
  if (!vcout[c]) vcs.push_back(c);
  vcout[c] += w;
}


/**
 * Scan communities connected to a vertex.
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 */
template <bool SELF=false, class G, class K, class W>
inline void rakScanCommunitiesW(vector<K>& vcs, vector<W>& vcout, const G& x, K u, const vector<K>& vcom) {
  x.forEachEdge(u, [&](auto v, auto w) { rakScanCommunityU<SELF>(vcs, vcout, u, v, w, vcom); });
}


/**
 * Clear communities scan data.
 * @param vcs total edge weight from vertex u to community C (temporary buffer, updated)
 * @param vcout communities vertex u is linked to (updated)
 */
template <class K, class W>
inline void rakClearScanW(vector<K>& vcs, vector<W>& vcout) {
  for (K c : vcs)
    vcout[c] = W();
  vcs.clear();
}


/**
 * Choose connected community with most weight.
 * @param x original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 * @param vcs communities vertex u is linked to
 * @param vcout total edge weight from vertex u to community C
 * @returns [best community, best edge weight to community]
 */
template <class G, class K, class W>
inline auto rakChooseCommunity(const G& x, K u, const vector<K>& vcom, const vector<K>& vcs, const vector<W>& vcout) {
  K d = vcom[u];
  K cmax = K();
  W wmax = W();
  for (K c : vcs)
    if (vcout[c]>wmax) { cmax = c; wmax = vcout[c]; }
  return make_pair(cmax, wmax);
}
#pragma endregion




#pragma region MOVE ITERATION
/**
 * Move each vertex to its best community.
 * @param vcom community each vertex belongs to (updated)
 * @param vaff is vertex affected (updated)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param fa is vertex allowed to be updated? (u)
 * @returns number of changed vertices
 */
template <class G, class K, class W, class F, class FA>
inline size_t rakMoveIterationOmpW(vector<K>& vcom, vector<F>& vaff, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, FA fa) {
  size_t a = K();
  size_t S = x.span();
  #pragma omp parallel for schedule(dynamic, 2048) reduction(+:a)
  for (K u=0; u<S; ++u) {
    int t = omp_get_thread_num();
    if (!x.hasVertex(u)) continue;
    if (!fa(u) || !vaff[u]) continue;
    K d = vcom[u];
    rakClearScanW(*vcs[t], *vcout[t]);
    rakScanCommunitiesW(*vcs[t], *vcout[t], x, u, vcom);
    auto [c, w] = rakChooseCommunity(x, u, vcom, *vcs[t], *vcout[t]);
    if (c && c!=d) { vcom[u] = c; ++a; x.forEachEdgeKey(u, [&](auto v) { vaff[v] = F(1); }); }
    vaff[u] = F(0);
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
template <class G, class FI, class FM, class FA>
inline auto rakInvokeOmp(const G& x, const RakOptions& o, FI fi, FM fm, FA fa) {
  using K = typename G::key_type;
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
  vector<vector<K>*> vcs(T);    // Hashtable keys
  vector<vector<W>*> vcout(T);  // Hashtable values
  rakAllocateHashtablesW(vcs, vcout, S);
  // Perform RAK algorithm.
  float tm = 0, ti = 0;  // Time spent in different phases
  float t  = measureDuration([&]() {
    // Initialize community membership.
    ti += measureDuration([&]() { fi(vcom); });
    // Mark affected vertices.
    tm += measureDuration([&]() { fm(vaff, vcs, vcout, vcom); });
    // Perform iterations.
    for (l=0; l<o.maxIterations;) {
      size_t n = rakMoveIterationOmpW(vcom, vaff, vcs, vcout, x, fa); ++l;
      if (double(n)/N <= o.tolerance) break;
    }
  }, o.repeat);
  rakFreeHashtablesW(vcs, vcout);
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
template <class G>
inline auto rakStaticOmp(const G& x, const RakOptions& o={}) {
  using FLAG = char;
  auto fi = [&](auto& vcom) { rakInitializeOmpW(vcom, x); };
  auto fm = [ ](auto& vaff, auto& vcs, auto& vcout, const auto& vcom) { fillValueOmpU(vaff, FLAG(1)); };
  auto fa = [ ](auto u) { return true; };
  return rakInvokeOmp(x, o, fi, fm, fa);
}
#pragma endregion
#pragma endregion
