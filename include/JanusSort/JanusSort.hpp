/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiSo).
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2019, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************************/

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
void sort(MPI_Comm mpi_comm, std::vector<value_type>& data, MPI_Datatype mpi_type,
          Compare&& comp = Compare(), int64_t global_elements = -1) {
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
void sort(RBC::Comm rbc_comm, std::vector<value_type>& data, MPI_Datatype mpi_type,
          Compare&& comp = Compare(), int64_t global_elements = -1) {
  make_sorter<value_type>(mpi_type).sort_range(rbc_comm, data, std::forward<Compare>(comp),
                                               global_elements);
}
}  // namespace JanusSort
