#include <cstdint>
#include <cstdio>
#include <utility>
#include <random>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include "inc/main.hxx"

using namespace std;




#pragma region CONFIGURATION
#ifndef TYPE
/** Type of edge weights. */
#define TYPE float
#endif
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 64
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 5
#endif
#pragma endregion




#pragma region METHODS
#pragma region HELPERS
/**
 * Obtain the modularity of community structure on a graph.
 * @param x original graph
 * @param a rak result
 * @param M sum of edge weights
 * @returns modularity
 */
template <class G, class K>
inline double getModularity(const G& x, const RakResult<K>& a, double M) {
  auto fc = [&](auto u) { return a.membership[u]; };
  return modularityBy(x, fc, M, 1.0);
}
#pragma endregion




#pragma region PERFORM EXPERIMENT
/**
 * Perform the experiment.
 * @param x original graph
 */
template <class G>
void runExperiment(const G& x) {
  double   M = edgeWeightOmp(x)/2;
  // Follow a specific result logging format, which can be easily parsed later.
  auto flog = [&](const auto& ans, const char *technique) {
    printf(
      "{%03d threads} -> "
      "{%09.1fms, %09.1fms mark, %09.1fms init, %04d iters, %01.9f modularity} %s\n",
      MAX_THREADS,
      ans.time, ans.markingTime, ans.initializationTime,
      ans.iterations, getModularity(x, ans, M), technique
    );
  };
  // Find static RAK, using OpenMP.
  {
    auto b0 = rakStaticOmp(x, {REPEAT_METHOD});
    flog(b0, "rakStaticOmp");
  }
  // Find static RAK, using CUDA.
  {
    auto b0 = rakStaticCuda(x, {REPEAT_METHOD});
    flog(b0, "rakStaticCuda");
  }
}


/**
 * Main function.
 * @param argc argument count
 * @param argv argument values
 * @returns zero on success, non-zero on failure
 */
int main(int argc, char **argv) {
  using K = uint32_t;
  using V = TYPE;
  install_sigsegv();
  char *file     = argv[1];
  bool symmetric = argc>2? stoi(argv[2]) : false;
  bool weighted  = argc>3? stoi(argv[3]) : false;
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  LOG("Loading graph %s ...\n", file);
  DiGraph<K, None, V> x;
  readMtxOmpW(x, file, weighted); LOG(""); println(x);
  if (!symmetric) { symmetrizeOmpU(x); LOG(""); print(x); printf(" (symmetrize)\n"); }
  runExperiment(x);
  printf("\n");
  return 0;
}
#pragma endregion
#pragma endregion
