#pragma once
#define BUILD  0  // 5 == BUILD_TRACE
#define OPENMP 1
#define CUDA   1
#include "_main.hxx"
#include "Graph.hxx"
#include "update.hxx"
#include "csr.hxx"
#include "mtx.hxx"
#include "snap.hxx"
#include "duplicate.hxx"
#include "symmetrize.hxx"
#include "selfLoop.hxx"
#include "properties.hxx"
#include "bfs.hxx"
#include "dfs.hxx"
#include "batch.hxx"
#include "hashtableCuda.hxx"
#include "sketchCuda.hxx"
#include "rak.hxx"
#include "rakLowmem.hxx"
#include "rakCuda.hxx"
#include "rakLowmemCuda.hxx"
