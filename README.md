# Karlsruhe Distributed Sorting Library (KaDiS)

This is the implementation of several sorting algorithms presented in the papers [1,2],
which contains an in-depth description of its inner workings, as well as an extensive experimental performance evaluation. Currently, the repository contains the following algorithms (many more algorithms will follow):

* Janus Quicksort (JQuick): a perfectly balanced distributed sorting algorithm (also see [1])
* Rubust Quicksort on Hypercubes (RQuick) [2]: a robust implementation of Hypercube Quicksort with minimal latency and advanced pivot selection.

All algorithms are implemented with RBC communicators [1] as well as MPI communicators. RBC communicators, implemented in the repository [RBC](https://github.com/MichaelAxtmann/RBC.git), support range-based communcator creation in constant time and support (non)blocking point-to-point operations as well as many (non)blocking collective operations.

[1] [Lightweight MPI Communicators with Applications to Perfectly Balanced Janus Quicksort](https://ieeexplore.ieee.org/abstract/document/8425179)

[2] [Robust Massively Parallel Sorting](https://epubs.siam.org/doi/abs/10.1137/1.9781611974768.7)


## RBC Example

```C++
#include <vector>

#include "JanusSort.hpp"
#include "RQuick.hpp"

std::vector<double> data = ...;

JanusSort::sort(comm, data, MPI_DOUBLE[, std::greater<double>(),...]);

// Create random generator with different seeds for each process.
std::random_device rd;
std::mt19937_64& gen(rd());
RQuick::sort(gen, data, MPI_DOUBLE, tag, comm[, std::greater<double>(),...]);
```

For a full description of the interface, see files [src/RQuick/RQuick.hpp](src/RQuick/RQuick.hpp) and [src/JanusSort/JanusSort.hpp](src/JanusSort/JanusSort.hpp). Please compile and execute the example with the following commands:

```
git submodule update --init --recursive
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=mpic++
make sortingexample
mpirun -np 10 ./sortingexample
```

Use ```make rbcsorting``` to compile the sorting library.

## Details

KaDiS uses the RBC library for communication. User of this library must not use the tags reserved by [RBC](https://github.com/MichaelAxtmann/RBC.git). Some sorting algorithms use additional tags which are reserved during the execution of the algorithm:

* JanusSort: 1000050-1000057
* RQuick: User provided tag