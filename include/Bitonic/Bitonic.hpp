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

#pragma once

#include <functional>
#include <random>
#include <vector>

#include "Configuration.hpp"

#include "RBC.hpp"

namespace Bitonic {
/* @brief bitonic sort algorithm
 *
 * Bitonic sort implementation which runs for any number of PEs. Also
 * the number of input elements on the PEs must not be the same.
 *
 * @param comm MPI communicator
 *
 * @param is_equal_input_size If this value is 'true', any PE must
 *    privide the same number of elements. In this case, the algorithm
 *    is slightly faster. If this value is 'false', the algorithm gets
 *    more complicated and we need to compute the maximum number of
 *    input elements. This calculation requires an additional
 *    allreduce operation.
 *
 * @param partition_strategy The user can choose two strategies:
 *    Merging and BinSearch. With Merging, we collect our own data and
 *    our partner's data and select the small (large) elements. With
 *    BinSearch, we use a distributed algorithm to determine the small
 *    (large) elements collectively and just exchange the requested
 *    elements. Merging is faster for small inputs and BinSearch is
 *    faster for large inputs.
 */
template <class T, class Comp = std::less<> >
void Sort(std::vector<T>& data, MPI_Datatype mpi_type, int tag,
          MPI_Comm comm, Comp comp = Comp(),
          bool is_equal_input_size = false,
          Bitonic::PartitionBy partition_strategy = Bitonic::PartitionBy::Merging);

/* @brief Sorts a bitonic sequence for any number of PEs and any
 *   number of local input sizes.
 *
 * Globally sorts a distributed array which contains a decreasing
 * sequence at the beginning followed by an increasing sequence at the
 * end.
 *
 * @param comm MPI communicator
 *
 * @param data Local elements. The global elements must be a bitonic
 *    sequnece.  The input must be locally sorted according to
 *    <comp>. Thus, if a PE stores a part of the decreasing sequence,
 *    its input is still sorting increasing according to <comp>.  This
 *    constraint avoids negative peaks in the local input.  E.g., the
 *    three local arrays <1, 1, 1> <1, 0, 1> <1, 1, 1> must be passed
 *    to this function in locally sorted order, e.g., <1, 1, 1> <0, 1,
 *    1> <1, 1, 1>.
 *
 * @param sort_increasing Determines whether we sort the input increasing or
 *    decreasing according to the comparator comp.

 * @param is_equal_input_size Defines whether each PE passes the same
 *    number of elements.  If the PEs pass different number of
 *    elements, we calculate the maximum number of local elements
 *    which has been passed and implicitly pad each local input with
 *    infinite large elements.
 */
template <class T, class Comp = std::less<> >
void SortBitonicSequence(std::vector<T>& data, MPI_Datatype mpi_type, int tag,
                         MPI_Comm comm, Comp comp = Comp(),
                         bool sort_increasing = true,
                         bool is_equal_input_size = false,
                         Bitonic::PartitionBy partition_strategy = Bitonic::PartitionBy::Merging);

/* @brief bitonic sort algorithm
 *
 * Bitonic sort implementation which runs for any number of PEs. Also
 * the number of input elements on the PEs must not be the same.
 *
 * @param comm MPI communicator
 *
 * @param is_equal_input_size If this value is 'true', any PE must
 *    privide the same number of elements. In this case, the algorithm
 *    is slightly faster. If this value is 'false', the algorithm gets
 *    more complicated and we need to compute the maximum number of
 *    input elements. This calculation requires an additional
 *    allreduce operation.
 *
 * @param partition_strategy The user can choose two strategies:
 *    Merging and BinSearch. With Merging, we collect our own data and
 *    our partner's data and select the small (large) elements. With
 *    BinSearch, we use a distributed algorithm to determine the small
 *    (large) elements collectively and just exchange the requested
 *    elements. Merging is faster for small inputs and BinSearch is
 *    faster for large inputs.
 */
template <class T, class Comp = std::less<> >
void Sort(std::vector<T>& data, MPI_Datatype mpi_type, int tag,
          const RBC::Comm& comm, Comp comp = Comp(),
          bool is_equal_input_size = false,
          Bitonic::PartitionBy partition_strategy = Bitonic::PartitionBy::Merging);

/* @brief Sorts a bitonic sequence for any number of PEs and any
 *   number of local input sizes.
 *
 * Globally sorts a distributed array which contains a decreasing
 * sequence at the beginning followed by an increasing sequence at the
 * end.
 *
 * @param comm RBC communicator
 *
 * @param data Local elements. The global elements must be a bitonic
 *    sequnece.  The input must be locally sorted according to
 *    <comp>. Thus, if a PE stores a part of the decreasing sequence,
 *    its input is still sorting increasing according to <comp>.  This
 *    constraint avoids negative peaks in the local input.  E.g., the
 *    three local arrays <1, 1, 1> <1, 0, 1> <1, 1, 1> must be passed
 *    to this function in locally sorted order, e.g., <1, 1, 1> <0, 1,
 *    1> <1, 1, 1>.
 *
 * @param sort_increasing Determines whether we sort the input increasing or
 *    decreasing according to the comparator comp.

 * @param is_equal_input_size Defines whether each PE passes the same
 *    number of elements.  If the PEs pass different number of
 *    elements, we calculate the maximum number of local elements
 *    which has been passed and implicitly pad each local input with
 *    infinite large elements.
 */
template <class T, class Comp = std::less<> >
void SortBitonicSequence(std::vector<T>& data, MPI_Datatype mpi_type, int tag,
                         const RBC::Comm& comm, Comp comp = Comp(),
                         bool sort_increasing = true,
                         bool is_equal_input_size = false,
                         Bitonic::PartitionBy partition_strategy = Bitonic::PartitionBy::Merging);
}  // namespace Bitonic

#include "../../src/Bitonic/Bitonic.hpp"
