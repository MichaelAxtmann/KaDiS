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

#include "Tags.hpp"

#include <RBC.hpp>

namespace MiniSort {
/* @brief MiniSort algorithm.
 *
 * Implementation of the "Scalable Algorithm" from the publication
 * "Parallel Sorting with Minimal Data" by Siebert and Wolf;
 * https://doi.org/10.1007/978-3-642-24449-0_20).
 *
 * @param comm MPI communicator
 *
 * @param mpi_type MPI datatype of element type T
 *
 * @param el Input value
 *
 * @param comp Comparator
 *
 * @return Output value
 */
template <class T, class Comp = std::less<>, class Tags = MiniSort::Tags<> >
T sort(MPI_Datatype mpi_datatype,
       T el, MPI_Comm comm, Comp comp = Comp{ });

/* @brief MiniSort algorithm.
 *
 * Implementation of the "Scalable Algorithm" from the publication
 * "Parallel Sorting with Minimal Data" by Siebert and Wolf;
 * https://doi.org/10.1007/978-3-642-24449-0_20).
 *
 * @param comm RBC communicator
 *
 * @param mpi_type MPI datatype of element type T
 *
 * @param el Input value
 *
 * @param comp Comparator
 *
 * @return Output value
 */
template <class T, class Comp = std::less<>, class Tags = MiniSort::Tags<> >
T sort(MPI_Datatype mpi_datatype,
       T el, const RBC::Comm& comm, Comp comp = Comp{ });
}  // end namespace MiniSort

#include "../../src/MiniSort/MiniSort.hpp"
