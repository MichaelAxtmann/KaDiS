/*****************************************************************************
 * This file is part of the Project JanusSortRBC
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * All rights reserved. Published under the BSD-2 license in the LICENSE file.
 ******************************************************************************/

#include <random>
#include <vector>

#include "JanusSort.hpp"
#include "RQuick.hpp"

#define PRINT_ROOT(msg) if (rank == 0) std::cout << msg << std::endl;

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Create random input elements
  PRINT_ROOT("Create random input elements");
  std::mt19937_64 generator;
  int data_seed = 3469931 + rank;
  generator.seed(data_seed);
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  std::vector<double> data;
  for (int i = 0; i < 10; ++i)
    data.push_back(dist(generator));

  {
    /* JanusSort */

    // Sort data descending
    auto data1 = data;
    PRINT_ROOT("Start sorting algorithm JanusSort with MPI_Comm. " <<
               "RBC::Communicators are used internally.");
    JanusSort::sort(comm, data1, MPI_DOUBLE, std::greater<double>());
    PRINT_ROOT("Elements have been sorted");

    PRINT_ROOT("Start sorting algorithm JanusSort with RBC::Comm.");
    RBC::Comm rcomm;
    RBC::Create_Comm_from_MPI(comm, &rcomm);
    auto data2 = data;
    JanusSort::sort(rcomm, data2, MPI_DOUBLE, std::greater<double>());
    PRINT_ROOT("Elements have been sorted");

    PRINT_ROOT("Start sorting algorithm JanusSort with RBC::Comm. " <<
               "MPI communicators and MPI collectives are used.");
    RBC::Comm rcomm1;
    RBC::Create_Comm_from_MPI(comm, &rcomm1, true, true);
    auto data3 = data;
    JanusSort::sort(rcomm1, data3, MPI_DOUBLE);
    PRINT_ROOT("Elements have been sorted");
  }

  {
    /* RQuick */
    int tag = 11111;

    // Sort data descending
    auto data1 = data;
    PRINT_ROOT("Start sorting algorithm RQuick with MPI_Comm. " <<
               "RBC::Communicators are used internally.");
    RQuick::sort(generator, data1, MPI_DOUBLE, tag, comm, std::greater<double>());
    PRINT_ROOT("Elements have been sorted");

    PRINT_ROOT("Start sorting algorithm RQuick with RBC::Comm.");
    RBC::Comm rcomm;
    RBC::Create_Comm_from_MPI(comm, &rcomm);
    auto data2 = data;
    RQuick::sort(generator, data2, MPI_DOUBLE, tag, rcomm, std::greater<double>());
    PRINT_ROOT("Elements have been sorted");

    PRINT_ROOT("Start sorting algorithm RQuick with RBC::Comm. " <<
               "MPI communicators and MPI collectives are used.");
    RBC::Comm rcomm1;
    RBC::Create_Comm_from_MPI(comm, &rcomm1, true, true);
    auto data3 = data;
    RQuick::sort(generator, data3, MPI_DOUBLE, tag, rcomm1);
    PRINT_ROOT("Elements have been sorted");
  }

  // Finalize the MPI environment
  MPI_Finalize();
}
