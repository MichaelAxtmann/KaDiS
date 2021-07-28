/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiS).
 *
 * Copyright (c) 2019, Michael Axtmann <michael.axtmann@kit.edu>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#include <random>
#include <vector>

#include "AmsSort/AmsSort.hpp"
#include "RQuick/RQuick.hpp"
#include "RFis/RFis.hpp"
#include "Bitonic/Bitonic.hpp"
#include "HSS/Hss.hpp"

#define PRINT_ROOT(msg) if (rank == 0) std::cout << msg << std::endl;

std::mt19937_64 createGen(int rank) {
  std::mt19937_64 gen;
  int data_seed = 3469931 + rank;
  gen.seed(data_seed);
  return gen;
}

std::vector<double> createData(std::mt19937_64* gen) {
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  std::vector<double> data;
  for (int i = 0; i < 10; ++i) {
    data.push_back(dist(*gen));
  }
  return data;
}

struct MyStruct {
  MyStruct(double v) : v(v){
  }
  MyStruct() {}
  double v;
};

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, size;
  const int tag = 12345;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const int num_levels = 2;

  // Create random input elements
  PRINT_ROOT("Create random input elements");
  auto gen = createGen(rank);

  // Create RBC communicator.
  RBC::Comm rcomm;
  // Does use RBC collectives and RBC sub-communicators internally.
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  // Sort data descending with RQuick.
  auto data = createData(&gen);
  PRINT_ROOT("Start sorting algorithm RQuick with MPI_Comm. " <<
             "RBC communicators are used internally.");
  RQuick::sort(MPI_DOUBLE, data, tag, gen, comm, std::greater<double>());
  // RQuick::sort(gen, data, MPI_DOUBLE, tag, comm);
  // RQuick::sort(gen, data, MPI_DOUBLE, tag, rcomm, std::less<double>());
  PRINT_ROOT("Elements have been sorted");

  PRINT_ROOT("Start sorting algorithm AMS-sort with RBC::Comm.");
  data = createData(&gen);
  std::vector<MyStruct> str;
  for (const auto& d : data) str.emplace_back(d);
  auto mycomp = [](const MyStruct& l, const MyStruct&r) {
		  return l.v < r.v;
		};
  Ams::sortLevel(MPI_DOUBLE, str, num_levels, gen, rcomm, mycomp);
  // int k = 2;
  // Ams::sort(MPI_DOUBLE, data, k, gen, rcomm);
  PRINT_ROOT("Elements have been sorted");

  PRINT_ROOT("Start sorting algorithm Bitonic Sort.");
  data = createData(&gen);
  const bool is_equal_input_size = true;
  Bitonic::Sort(data, MPI_DOUBLE, tag, rcomm, std::less<>{ }, is_equal_input_size);
  // Bitonic::Sort(data, MPI_DOUBLE, tag, rcomm);
  // if (rank == 0) {
  //   data.pop_back();
  // }
  // Bitonic::Sort(data, MPI_DOUBLE, tag, rcomm, std::less<>{}, !is_equal_input_size);
  PRINT_ROOT("Elements have been sorted");

  PRINT_ROOT("Start sorting algorithm Robust Fast Work-Inefficient Sort.");
  data = createData(&gen);
  RFis::Sort(MPI_DOUBLE, data, rcomm);
  PRINT_ROOT("Elements have been sorted");

  PRINT_ROOT("Start sorting algorithm HSS-sort with RBC::Comm.");
  data = createData(&gen);
  Hss::sortLevel(MPI_DOUBLE, data, num_levels, gen, rcomm);
  // Hss::sortLevel(MPI_DOUBLE, data, num_levels, gen, rcomm);
  // int k = 2;
  // Hss::sort(MPI_DOUBLE, data, k, gen, rcomm);
  PRINT_ROOT("Elements have been sorted");


  // Finalize the MPI environment
  MPI_Finalize();
}
