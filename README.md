Design of memory-efficient CUDA-based Parallel [Label Propagation Algorithm (LPA)], aka RAK, for [community detection].

Community detection involves grouping nodes in a graph with dense connections within groups, than between them. We previously proposed efficient multicore ([GVE-LPA]) and GPU-based ([ν-LPA]) implementations of Label Propagation Algorithm (LPA) for community detection. However, these methods incur high memory overhead due to their per-thread/per-vertex hashtables. This makes it challenging to process large graphs on shared memory systems. In this report, we introduce memory-efficient GPU-based LPA implementations, using weighted Boyer-Moore (BM) and Misra-Gries (MG) sketches. Our new implementation, νMG8-LPA, using an 8-slot MG sketch, reduces memory usage by `98x` and `44x` compared to GVE-LPA and ν-LPA, respectively. It is also `2.4x` faster than GVE-LPA and only `1.1x` slower than ν-LPA, with minimal quality loss (`4.7%`/`2.9%` drop compared to GVE-LPA/ν-LPA).


Below we plot the memory usage of [NetworKit] LPA, [GVE-LPA], [ν-LPA], 𝜈MG8-LPA, and 𝜈BM-LPA on 13 different graphs. 𝜈MG8-LPA and 𝜈BM-LPA achieve, on average, `2.2x`, `98x`, and `44x` lower memory usage than NetworKit LPA, GVE-LPA, and 𝜈-LPA.

[![](https://i.imgur.com/mTVkyBh.png)][sheets-o1]

Next, we plot the time taken by [NetworKit] LPA, [GVE-LPA], [ν-LPA], 𝜈MG8-LPA, and 𝜈BM-LPA on 13 different graphs. On average, 𝜈BM-LPA is `186x`, `9.0x`, `3.5x`, and `3.7x` faster than NetworKit LPA, GVE-LPA, 𝜈-LPA, and 𝜈MG8-LPA, respectively; while 𝜈MG8-LPA is `51x` and `2.4x` faster than NetworKit LPA and GVE-LPA, but `1.1x` and `3.7x` slower than 𝜈-LPA and 𝜈BM-LPA.

[![](https://i.imgur.com/wIJaFc1.png)][sheets-o1]

Further, we plot the speedup of 𝜈MG8-LPA over NetworKit LPA, GVE-LPA, and 𝜈-LPA.

[![](https://i.imgur.com/vQlweCL.png)][sheets-o1]

Finally, we plot the modularity of communities identified by NetworKit LPA, GVE-LPA, 𝜈-LPA, 𝜈MG8-LPA, and 𝜈BM-LPA. 𝜈BM-LPA identifies communities that are of `27%`, `24%`, `23%`, and `20%` lower quality, respectively. However, the communities identified by 𝜈MG8-LPA are only `8.4%`, `4.7%`, and `2.9%` lower in quality than NetworKit LPA, GVE-LPA, and 𝜈-LPA, and `25%` higher than 𝜈BM-LPA.

[![](https://i.imgur.com/c23ma3z.png)][sheets-o1]



Refer to our technical report for more details: \
[Memory Efficient GPU-based Label Propagation Algorithm (LPA) for Community Detection on Large Graphs][report].

<br>

> [!NOTE]
> You can just copy `main.sh` to your system and run it. \
> For the code, refer to `main.cxx`.


[Label Propagation Algorithm (LPA)]: https://arxiv.org/abs/0709.2938
[NetworKit]: https://github.com/networkit/networkit
[GVE-LPA]: https://github.com/puzzlef/rak-communities-openmp
[ν-LPA]: https://github.com/puzzlef/rak-communities-cuda
[community detection]: https://en.wikipedia.org/wiki/Community_search
[sheets-o1]: https://docs.google.com/spreadsheets/d/11NPJ-2I4ZIvCQSuL36vkwpQ-RU6t5NmHuIsZyQIRah4/edit?usp=sharing
[report]: https://arxiv.org/abs/2411.19901

<br>
<br>


### Code structure

The code structure of νMG-LPA / νBM-LPA is as follows:

```bash
# Main files
- main.sh: Shell script for running experiments
- process.js: Node.js script for processing output logs
- main.cxx: Experimentation code

# Key algorithms
- rakLowmemCuda.hxx: Memory-efficient GPU-based LPA, i.e., νMG-LPA, νBM-LPA
- rakLowmem.hxx: Memory-efficient CPU-based LPA, i.e., MG-LPA, BM-LPA
- rakCuda.hxx: GPU-based LPA, i.e., ν-LPA
- rak.hxx: CPU-based LPA, i.e., GVE-LPA
- sketchCuda.hxx: Misra-Gries sketch for νMG-LPA
- hashtableCuda.hxx: Hashtable for ν-LPA

# Common graph operations
- inc/main.hxx: Main header
- inc/Graph.hxx: Graph data structure functions
- inc/mtx.hxx: Graph file reading functions
- inc/update.hxx: Update functions
- inc/csr.hxx: Compressed Sparse Row (CSR) data structure functions
- inc/bfs.hxx: Breadth-first search algorithms
- inc/dfs.hxx: Depth-first search algorithms
- inc/duplicate.hxx: Graph duplicating functions
- inc/symmetrize.hxx: Graph Symmetrization functions
- inc/transpose.hxx: Graph transpose functions
- inc/selfLoop.hxx: Graph Self-looping functions
- inc/properties.hxx: Graph Property functions
- inc/batch.hxx: Batch update generation functions

# Support headers
- inc/_algorithm.hxx: Algorithm utility functions
- inc/_bitset.hxx: Bitset manipulation functions
- inc/_cmath.hxx: Math functions
- inc/_ctypes.hxx: Data type utility functions
- inc/_cuda.hxx: CUDA utility functions
- inc/_debug.hxx: Debugging macros (LOG, ASSERT, ...)
- inc/_iostream.hxx: Input/output stream functions
- inc/_iterator.hxx: Iterator utility functions
- inc/_main.hxx: Main program header
- inc/_mpi.hxx: MPI (Message Passing Interface) utility functions
- inc/_openmp.hxx: OpenMP utility functions
- inc/_queue.hxx: Queue utility functions
- inc/_random.hxx: Random number generation functions
- inc/_string.hxx: String utility functions
- inc/_utility.hxx: Runtime measurement functions
- inc/_vector.hxx: Vector utility functions
```

Note that each branch in this repository contains code for a specific experiment. The `main` branch contains code for the final experiment. If the intention of a branch in unclear, or if you have comments on our technical report, feel free to open an issue.

<br>
<br>


## References

- [Near linear time algorithm to detect community structures in large-scale networks; Raghavan et al. (2007)](https://arxiv.org/abs/0709.2938)
- [The University of Florida Sparse Matrix Collection; Davis et al. (2011)](https://doi.org/10.1145/2049662.2049663)
- [How to import VSCode keybindings into Visual Studio?](https://stackoverflow.com/a/62417446/1413259)
- [Configure X11 Forwarding with PuTTY and Xming](https://www.centlinux.com/2019/01/configure-x11-forwarding-putty-xming-windows.html)
- [Installing snap on CentOS](https://snapcraft.io/docs/installing-snap-on-centos)

<br>
<br>


[![](https://img.youtube.com/vi/M6npDdVGue4/maxresdefault.jpg)](https://www.youtube.com/watch?v=M6npDdVGue4)<br>
[![ORG](https://img.shields.io/badge/org-puzzlef-green?logo=Org)](https://puzzlef.github.io)
![](https://ga-beacon.deno.dev/G-KD28SG54JQ:hbAybl6nQFOtmVxW4if3xw/github.com/puzzlef/rak-lowmem-communities-cuda)

[Prof. Dip Sankar Banerjee]: https://sites.google.com/site/dipsankarban/
[Prof. Kishore Kothapalli]: https://faculty.iiit.ac.in/~kkishore/
[SuiteSparse Matrix Collection]: https://sparse.tamu.edu
