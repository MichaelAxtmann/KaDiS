/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiSo).
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
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

#pragma once

#include <functional>
#include <utility>
#include <vector>

#include "../../src/JanusSort/JanusSort.hpp"
#include "RBC.hpp"

namespace JanusSort {
/**
 * Helper function for creating a reusable parallel sorter.
 */
template <class value_type>
Sorter<value_type> make_sorter(MPI_Datatype mpi_type, int seed = 1, int64_t min_samples = 64) {
  return Sorter<value_type>(mpi_type, seed, min_samples, false);
}

/**
 * Configurable interface.
 *
 * Sorts the input data with a custom compare operator
 * @param mpi_comm MPI commuicator (all ranks have to call the function)
 * @param data_vec Vector that contains the input data
 * @param global_elements The total number of elements on all PEs.
 *          Set the parameter to -1 if unknown or if the global input
 *          is not evenly distributed, i.e., x, ..., x, x-1, ..., x-1.
 *          If the parameter is not set to -1, the algorithm runs
 *          slightly faster for small inputs.
 * @param comp The compare operator
 */
template <class value_type, class Compare = std::less<value_type> >
void sort(std::vector<value_type>& data, MPI_Datatype mpi_type,
          MPI_Comm mpi_comm, Compare&& comp = Compare(),
          int64_t global_elements = -1) {
  make_sorter<value_type>(mpi_type).sort(mpi_comm, data, std::forward<Compare>(comp),
                                         global_elements);
}

/**
 * Configurable interface.
 *
 * Sorts the input data with a custom compare operator
 * @param rbc_comm rbc commuicator (all ranks have to call the function)
 * @param data_vec Vector that contains the input data
 * @param global_elements The total number of elements on all PEs.
 *          Set the parameter to -1 if unknown or if the global input
 *          is not evenly distributed, i.e., x, ..., x, x-1, ..., x-1.
 *          If the parameter is not set to -1, the algorithm runs
 *          slightly faster for small inputs.
 * @param comp The compare operator
 */
template <class value_type, class Compare = std::less<value_type> >
void sort(std::vector<value_type>& data, MPI_Datatype mpi_type,
          RBC::Comm rbc_comm, Compare&& comp = Compare(),
          int64_t global_elements = -1) {
  make_sorter<value_type>(mpi_type).sort_range(rbc_comm, data, std::forward<Compare>(comp),
                                               global_elements);
}
}  // namespace JanusSort
