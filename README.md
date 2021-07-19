# Karlsruhe Distributed Sorting Library (KaDiS)

This is the implementation of several sorting algorithms presented in the papers \[[0](https://dl.acm.org/doi/10.1145/2755573.2755595),[1](https://epubs.siam.org/doi/abs/10.1137/1.9781611974768.7),[2](https://doi.org/10.1109/IPDPS.2018.00035)\],
which contains an in-depth description of their inner workings, as well as an extensive experimental performance evaluation:

* **AmsSort**: An implementation of our algorithm *Adaptive Multi-level Samplesort* (AMS-sort) [[0](https://dl.acm.org/doi/10.1145/2755573.2755595),[1](https://epubs.siam.org/doi/abs/10.1137/1.9781611974768.7)]. It works for arbitrary number of processes, guarantees that the data imbalance is limited by a constant factor &epsilon; with &epsilon; being a tuning parameter, and has a very good isoefficiency function.
* **RQuick**: Our robust version of Hypercube Quicksort [[1]](https://epubs.siam.org/doi/abs/10.1137/1.9781611974768.7) with polylogarithmic latency and advanced pivot selection.
* **RFis**: Implementation of our *Robust Fast Work-Inefficient Sorting* algorithm [[0](https://dl.acm.org/doi/10.1145/2755573.2755595),[1](https://epubs.siam.org/doi/abs/10.1137/1.9781611974768.7)] with logarithmic latency.
* **Rlm**: A prototypical implementation of our algorithm *Recursive Last Multiway Mergesort* (RLM-sort) [[0]](https://dl.acm.org/doi/10.1145/2755573.2755595). It works for arbitrary number of processes, guarantees that the data imbalance is limited by a constant factor &epsilon; with &epsilon; being a tuning parameter, and has a slightly worse isoefficiency function than AmsSort.

Additionally, the repository contains our implementations of third-party algorithms (Bitonic and MiniSort) as well as hybrid algorithms (Hss):

* **Bitonic**: An own implementation of bitonic sort that works for arbitrary number of processes.
* **MiniSort**: An own implementation of the *Scalable Algorithm* from Siebert and Wolf [[3]](https://doi.org/10.1007/978-3-642-24449-0_20) that sorts one element per process.
* **JanusSort**: An implementation of our perfectly balanced algorithm *Janus Quicksort* [[2]](https://doi.org/10.1109/IPDPS.2018.00035).
* **Hss**: Our implementation of AMS-sort [0,1] that uses the *Histogram Sort with Sampling* splitter selection routine proposed by Harsh et al. [4](https://doi.org/10.1145/3323165.3323184). Hss uses the algorithmic framework and the data redistribution routine of AmsSort, and thus, works for arbitrary number of processes.


All algorithms are implemented with RBC communicators [[2]](https://doi.org/10.1109/IPDPS.2018.00035). RBC communicators, implemented in the repository [RBC](https://github.com/MichaelAxtmann/RBC.git), support range-based communicator creation in constant time and support (non)blocking point-to-point operations as well as many (non)blocking collective operations. The interface accepts RBC communicators as well as MPI communicators. If an algorithm is invoked with an MPI communicator, RBC communicators are used internally.

[0] [Practical Massively Parallel Sorting](https://dl.acm.org/doi/10.1145/2755573.2755595)

[1] [Robust Massively Parallel Sorting](https://epubs.siam.org/doi/abs/10.1137/1.9781611974768.7)

[2] [Lightweight MPI Communicators with Applications to Perfectly Balanced Janus Quicksort](https://doi.org/10.1109/IPDPS.2018.00035)

[3] [Parallel Sorting with Minimal Data](https://doi.org/10.1007/978-3-642-24449-0_20)

[4] [Histogram Sort with Sampling](https://doi.org/10.1145/3323165.3323184)

## Sorting Examples

```C++
#include <vector>

#include "AmsSort/AmsSort.hpp"
#include "Bitonic/Bitonic.hpp"
#include "HSS/Hss.hpp"
#include "MiniSort/MiniSort.hpp"
#include "RFis/RFis.hpp"
#include "RLM/Rlm.hpp"
#include "RQuick/RQuick.hpp"
#include "JanusSort/JanusSort.hpp"

MPI_Comm comm  = MPI_COMM_WORLD;
std::vector<double> data = ...;
double el = ...;
const int kway = 64;
const int num_levels = 3;

// Create random generator with different seeds for each process.
std::random_device rd;
std::mt19937_64& gen(rd());

Ams::sort(MPI_DOUBLE, data, kway, gen, comm[, comp = std::less<>(), imbalance = 1.10,...]);
Ams::sortLevel(MPI_DOUBLE, data, num_levels, gen, comm[, comp = std::less<>(), imbalance = 1.10,...]);

RQuick::sort(MPI_DOUBLE, data, tag, gen, comm[, comp = std::less<>(),...]);

RFis::Sort(MPI_DOUBLE, data, comm[, comp = std::less<>(),...]);

MiniSort::sort(MPI_DOUBLE, el, comm[, comp = std::less<>(),...]);

JanusSort::sort(comm, data, MPI_DOUBLE[, comp = std::less<>(),...]);

Bitonic::Sort(data, MPI_DOUBLE, tag, comm[, comp = std::less<>{ }, is_equal_input_size = false,...]);
Hss::sort(MPI_DOUBLE, data, kway, gen, comm[, comp = std::less<>(), imbalance = 1.10,...]);
Hss::sortLevel(MPI_DOUBLE, data, num_levels, gen, comm[, comp = std::less<>(), imbalance = 1.10,...]);
Rlm::sort(MPI_DOUBLE, data, kway, gen, comm[, comp = std::less<>(), imbalance = 1.10,...]);
Rlm::sortLevel(MPI_DOUBLE, data, num_levels, gen, comm[, comp = std::less<>(), imbalance = 1.10,...]);
```

For a full description of the interface, see files 
[include/AmsSort/AmsSort.hpp](include/AmsSort/AmsSort.hpp) [include/RQuick/RQuick.hpp](include/RQuick/RQuick.hpp)  [include/RFis/RFis.hpp](include/RFis/RFis.hpp) [include/RLM/Rlm.hpp](include/RLM/Rlm.hpp) [include/Bitonic/Bitonic.hpp](include/Bitonic/Bitonic.hpp) [include/MiniSort/MiniSort.hpp](include/MiniSort/MiniSort.hpp) [include/JanusSort/JanusSort.hpp](include/JanusSort/JanusSort.hpp) and [include/HSS/Hss.hpp](include/HSS/Hss.hpp). Please compile and execute the example with the following commands:

```
git submodule update --init --recursive
mkdir Release
cd Release
cmake .. -DCMAKE_BUILD_TYPE=Release -DMPI_C_COMPILER=mpicc -DMPI_CXX_COMPILER=mpic++
make kadisexample
mpirun -np 10 ./kadisexample
```

In the case that you use the sorting algorithms in your source code, link against the KaDiS library:
```
add_subdirectory(<path-to-KaDis>)
target_link_libraries(<your-target> kadis)
```

Use ```make rbcsorting``` to compile the sorting library.

## Details

KaDiS uses the RBC library for communication. User of this library must not use the tags reserved by [RBC](https://github.com/MichaelAxtmann/RBC.git). Some sorting algorithms use additional tags which are reserved during the execution of the algorithm:

* AmsSort, Hss, and Rlm: 10000-10002 (see [include/AmsSort/Tags.hpp](include/AmsSort/Tags.hpp))
* RFis: 10003-10004 (see [include/RFis/Tags.hpp](include/RFis/Tags.hpp))
* MiniSort: 10005-10006 (see [include/MiniSort/Tags.hpp](include/MiniSort/Tags.hpp))
* RQuick: User provided tag
* JanusSort: 1000050-1000057

The tags of the algorithms AmsSort, Hss, Rlm, RFis, and MiniSort can be adapted to your needs via preprocessor defines or via the algorithms' interface.
