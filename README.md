# Hotspot Analysis and Platform Choice

**Project:** Locality-Aware Task Scheduling and Data Distribution for OpenMP Programs
on NUMA Systems and Manycore Processors (Muddukrishna, Jonsson, Brorsson, 2015)

**Author:** Siddharth Kancharla, S20240030402, IIIT Sri City

---

## 1. What the sequential program does

`matmul.c` computes `C = A x B` for dense `n x n` double matrices, two ways:

- `matmul_naive` walks rows of A against **columns** of B
- `matmul_blocked` splits all three matrices into `bs x bs` tiles and multiplies tile by tile

Both do exactly `2n^3` floating point operations. Same arithmetic, same answer
(verified to zero difference). The only thing that changes is the **order in which
memory is touched**, which is the entire point the paper is making.

---

## 2. Hotspot identification

### gprof, `-O0`, n = 1024

```
  %   cumulative   self              self     total
 time   seconds   seconds    calls   s/call   s/call  name
 75.02      9.49     9.49        1     9.49     9.49  matmul_naive
 24.98     12.65     3.16     4096     0.00     0.00  multiply_block
  0.00     12.65     0.00        4     0.00     0.00  alloc_matrix
  0.00     12.65     0.00        2     0.00     0.00  fill_random
```

**Hotspot: the two multiply kernels account for 100.00% of execution time.**
Everything else (allocation, initialisation, verification, CSV logging) rounds to 0%.

Sanity check on the call counts: `multiply_block` is called 4096 times, and
`(1024 / 64)^3 = 16^3 = 4096`. The profile is internally consistent, so these
numbers can be trusted.

### Effect of optimisation level on the profile

| Build | Dominant symbol | Self time | What the compiler did |
|---|---|---|---|
| `-O0` | `matmul_naive` | 75.02% | nothing inlined |
| `-O2` | `matmul_naive` | 90.86% | `multiply_block` inlined into `matmul_blocked` |
| `-O3` | `main` | 89.50% | `matmul_naive` itself inlined into `main` |

At `-O3` the naive kernel disappears as a separate symbol entirely. This is the same
inlining effect observed in Tutorial 0 and it is a known limitation of gprof:
once a function is inlined, its time is attributed to the caller.

### The decisive measurement

Optimisation flags do almost nothing for the naive kernel:

| Build | Naive | Blocked | Blocking speedup |
|---|---|---|---|
| `-O0` | 9.482 s | 2.898 s | **3.27x** |
| `-O2` | 9.758 s | 1.015 s | **9.62x** |
| `-O3` | 9.541 s | 0.979 s | **9.75x** |

Naive stays pinned near 9.5 seconds at every optimisation level. Blocked drops from
2.90 s to 0.98 s. **The compiler cannot optimise away a bad memory access pattern.**
Only restructuring the access order fixes it.

### The cache conflict cliff

Naive kernel, `-O2`, varying only `n`:

| n | 1020 | 1023 | **1024** | 1025 | 1028 |
|---|---|---|---|---|---|
| time | 1.08 s | 1.39 s | **9.40 s** | 1.47 s | 1.14 s |

Changing the matrix size by **one element** changes runtime by **6.4x**, while
doing slightly more arithmetic.

Cause: in the naive kernel the inner loop walks a column of B, stepping `n`
doubles at a time. At `n = 1024` that stride is `1024 x 8 = 8192` bytes, a power
of two. Addresses that far apart map to the same cache set, so each access in the
column evicts the previous one. The cache is effectively reduced to one usable
line per set. This is a classic conflict miss.

With blocking the cliff shrinks from 6.4x to 1.8x, because each tile of B is
small enough that the whole tile stays resident while it is reused.

This single experiment is the strongest evidence in the project that the bottleneck
is **memory placement, not computation**.

---

## 3. Are the hotspots parallelizable?

| Loop | Parallelizable | Reason |
|---|---|---|
| `ii` (block row of C) | **Yes** | Each `ii` writes a disjoint set of rows of C. No two iterations touch the same output element, so there is no race. |
| `jj` (block column of C) | **Yes** | Same argument, disjoint columns of C. `(ii, jj)` pairs are fully independent, giving `(n/bs)^2` independent units of work. |
| `kk` (accumulation) | **No, not without extra work** | Every `kk` iteration does `C[block] += ...` on the **same** block of C. Parallelizing it directly is a write-write race. It would need a reduction with private partial blocks, or atomics, both of which cost more than they gain here. |
| Inner `i, j, k` | Technically yes, not useful | The trip count is only `bs` (64). Thread creation overhead would dominate the work. Leave these serial so the compiler can vectorise them. |
| `fill_random` | No | `rand()` keeps internal state, so calls are order dependent. It is also 0% of runtime, so there is nothing to gain. |

**Conclusion:** parallelize the outer `(ii, jj)` block pairs, keep `kk` sequential
inside each pair. At `n = 1024, bs = 64` that gives `16 x 16 = 256` independent
tasks, which is plenty of parallelism for any desktop or server core count.

This maps directly onto the paper's model. Each `(ii, jj)` task has a known data
footprint (one block row of A, one block column of B, one block of C), which is
exactly the information the paper's `depend` clause feeds to its locality-aware
scheduler so it can place the task near its data.

---

## 4. Chosen platform: OpenMP

### Justification

**1. The paper is about OpenMP.** The whole contribution is a data distribution
scheme and a task scheduler for OpenMP's task model. Reproducing it in MPI or CUDA
would not be reproducing it.

**2. The problem is shared memory with a locality penalty.** All three matrices
sit in one address space. Every task reads overlapping parts of A and B. This is
precisely the case OpenMP is designed for, and precisely the case where the paper
says the default work-stealing scheduler fails.

**3. OpenMP exposes the knobs the paper argues for.** The paper asks for programmer
control over placement and scheduling. OpenMP already provides a usable subset:

- `#pragma omp task depend(in: ...) depend(inout: ...)` declares the data footprint,
  which is the exact input Algorithm 1 uses to compute the distribution `D`
- `OMP_PROC_BIND` and `OMP_PLACES` pin threads so a thread stops migrating away
  from its data, which is the practical stand-in for the paper's task-queue-per-node
- first-touch initialisation inside a parallel region reproduces the paper's
  `coarse` distribution policy without needing `omp_malloc`
- `schedule(static)` vs `schedule(dynamic)` is the load-balancing versus locality
  tradeoff the paper measures

**4. Incremental adoption.** The sequential code stays valid. Adding pragmas does
not restructure the program, which matches the paper's stated goal of an
"architecture-oblivious approach for programmers".

### Why not MPI

MPI assumes distributed memory and no shared address space. To use it we would have
to explicitly partition and send the matrices between ranks. That **hides** the
effect we are trying to study: once every rank has its own private copy there is no
shared-memory locality problem left to measure, because remote data is no longer
transparently accessible, it is an explicit message. We would be measuring network
bandwidth, not memory placement. MPI also cannot express the paper's central idea,
since there is no runtime scheduler deciding where a task runs.

MPI would be the right choice if the matrices were too large for one node's RAM.
They are not.

### Why not CUDA

Matrix multiplication is a famously good GPU workload, so this needs a careful answer.

- A GPU has its own memory hierarchy (global, shared, registers) and its own
  locality problem, but it is **not NUMA and not a cache-bank topology**. The
  paper's model of nodes, distances and home caches has no counterpart there.
- CUDA has no task scheduler the programmer can influence in the way the paper
  describes. Warp scheduling is fixed in hardware.
- The comparison would become "GPU beats CPU", which is true and uninteresting
  and says nothing about the paper's claim.

CUDA would be the right choice if the goal were raw matmul throughput rather than
studying scheduling and placement.

### Summary

| Platform | Fits the problem? | Verdict |
|---|---|---|
| **OpenMP** | Shared memory, task-based, exposes affinity and dependence control | **Chosen** |
| MPI | Distributed memory, explicit messaging, removes the shared-memory effect being studied | Rejected |
| CUDA | Different memory model, no programmer-visible task scheduler | Rejected |

---

## 5. What comes next

The parallel implementation (`matmul_omp.c`) will compare four configurations at
the same `n` and `bs`, to isolate scheduling and placement from raw thread count:

1. **Sequential baseline** (this file)
2. **`parallel for`, default placement** - naive parallelisation, no affinity
3. **OpenMP tasks with `depend`** - work stealing decides placement, the paper's baseline
4. **Tasks plus affinity** - `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, first-touch
   initialisation. This is the locality-aware configuration.

Expected result, following the paper: configuration 4 beats 3 on the same thread
count, and the gap widens as thread count rises, because work stealing increasingly
moves tasks away from their data.

### Honest limitation to state in the presentation

The test machine for this stage has **a single NUMA node** (`lscpu` reports
`NUMA node(s): 1`). The inter-node latency effect from Figure 2 of the paper
therefore cannot be reproduced here. What *is* reproduced is the same principle one
level down the memory hierarchy, at the cache level, which is exactly what the
paper's TILEPro64 half is about: home caches and banked shared caches rather than
NUMA nodes. The 6.4x conflict cliff and the 9.75x blocking speedup are both
measurements of that same effect.

---

## 6. Comparative results (completed)

The parallel implementation is `matmul_omp.c`. Four configurations, all
verified against the sequential reference to zero difference.

### Headline: locality and parallelism are independent, and they multiply

| n = 1024, -O2 | serial | 2 threads | thread speedup |
|---|---|---|---|
| naive | 9.585 s | 3.039 s | **3.15x** |
| blocked | 0.932 s | 0.497 s | **1.88x** |

- locality alone (naive serial -> blocked serial): **10.28x**
- threading alone (naive serial -> naive parallel): **3.15x**
- both together: **19.29x**

### The superlinear result is diagnostic, not an error

The naive kernel gains **3.15x from 2 cores**, which exceeds the core count.
A second core brings a second private cache. That only helps a kernel that was
starved of cache rather than of compute. The blocked kernel gains only 1.88x
precisely because it had already solved its memory problem, so the extra cache
had nothing left to fix.

This is independent confirmation that the hotspot was memory bound.

### The three parallel strategies, n = 1024, 2 threads

| Configuration | Time | Speedup |
|---|---|---|
| Sequential | 0.936 s | 1.00x |
| `omp parallel for` | 0.476 s | 1.97x |
| `omp task` (work stealing) | 0.488 s | 1.92x |
| `omp task` + bind + first touch | 0.479 s | 1.95x |

**Negative result, reported honestly:** all three land within 2.5% of each
other, and `OMP_PROC_BIND=close` versus `OMP_PROC_BIND=false` makes no
measurable difference (2.00x unbound vs 1.97x bound).

This does not contradict the paper. It confirms its framing. The test machine
has **one NUMA node and two cores**, so there is no node distance and no
placement decision to get right. The paper's mechanism needs a topology to
exploit; with no topology, correctly, there is nothing to win. Reproducing the
paper's reported +50% NUMA gain requires a multi-socket machine.

### Task overhead is real and measurable

At **one** thread the task version runs at **0.97x** of sequential, i.e. 3%
slower. That is pure task-creation and dependence-tracking overhead with no
parallelism to pay for it. Worth stating because it bounds how fine-grained
tasks can usefully be.

### Oversubscription

At 4 threads on 2 cores the task configurations still reach 2.00x and 2.01x,
while `parallel for` drops to 1.91x. Tasking degrades more gracefully under
oversubscription because the runtime can rebalance, which is the load-balancing
benefit the paper says often outweighs locality (their Algorithm 2 keeps work
stealing for exactly this reason).
