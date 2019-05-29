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

#include "../Tools/MpiTuple.hpp"
#include "Tags.hpp"

#include "RBC.hpp"

namespace RFis {
/* @brief Robust fast work-inefficient sorting algorithm.
 *

 * The algorithm first performs a fast work-inefficient ranking
 * routine (RankRobust) followed by a redistribution phase.
 *
 * @param comm MPI communcator
 *
 * @param els Local input and output. If the global number of elements
 * is less than prev_pow_of_two(comm size), it is expected that
 * processes [0, global number of elements) store one input element
 * each. The algorithm works without this condition but may be slower
 * than expected. The output is evenly distributed to processes
 * prev_pow_of_two(comm size). If the global number of elements
 * is less than prev_pow_of_two(comm size), it is expected that
 * processes [0, global number of elements).
 *
 */
template <class T, class CompLess = std::less<>, class RFisTags = RFis::Tags<> >
void Sort(MPI_Datatype mpi_datatype,
                std::vector<T>& els,
                MPI_Comm comm,
                CompLess comp = std::less<>{ });

/* @brief Ranks elements
 *
 * Arranges processes in a grid. If the number of processes is a
 * square root, the height and the width are the same. Otherwise,
 * the grid contains one additional row which contains the
 * remaining processes.
 *
 * 1. Sort input and perform column-wise All-gather-merge ('column data').
 *
 * 2. Transpose data; process (i, j) sends element to process (j, i) ('row data').
 * Last row may not do anything.
 *
 * 3. Rank row data into column data.
 *
 * 4. Allreduce ranks. The processes of a row now store the same elements and ranks.
 *
 * 5. The i'th process of a row returns the i'th largest ranked row data.
 *
 * @param comm MPI communcator
 *
 * @param els Local input. If the global number of elements is less
 *   than comm size, it is expected that processes [0, global
 *   number of elements) store one element each. The algorithm works
 *   without this condition but may be slower than expected.
 *
 * @return Returns a vector of tuples storing pairs of elements and
 *   their rank value. The type RankType has to be a integral type.
 */
template <class T, class RankType, class CompLess = std::less<>, class RFisTags = RFis::Tags<> >
std::vector<Tools::Tuple<T, RankType> > Rank(MPI_Datatype mpi_datatype,
                                                   MPI_Datatype rank_type,
                                                   std::vector<T> els,
                                                   MPI_Comm comm,
                                                   CompLess comp = std::less<>{ });

/* @brief Ranks elements
 *
 * Same as function above with RankType = int.
 */
template <class T, class CompLess = std::less<>, class RFisTags = RFis::Tags<> >
std::vector<Tools::Tuple<T, int> > Rank(MPI_Datatype mpi_datatype,
                                              std::vector<T> els,
                                              MPI_Comm comm,
                                              CompLess comp = std::less<>{ });

/* @brief Robust fast work-inefficient sorting algorithm.
 *
 * The algorithm first performs a fast work-inefficient ranking
 * routine (RankRobust) followed by a redistribution phase.
 *
 * @param comm RBC communcator
 *
 * @param els Local input and output. If the global number of elements
 *   is less than prev_pow_of_two(comm.getSize()), it is expected that
 *   processes [0, global number of elements) store one input element
 *   each. The algorithm works without this condition but may be
 *   slower than expected. The output is evenly distributed to
 *   processes prev_pow_of_two(comm.getSize()). If the global number
 *   of elements is less than prev_pow_of_two(comm.getSize()), it is
 *   expected that processes [0, global number of elements).
 *
 */
template <class T, class CompLess = std::less<>, class RFisTags = RFis::Tags<> >
void Sort(MPI_Datatype mpi_datatype,
                std::vector<T>& els,
                const RBC::Comm& comm,
                CompLess comp = std::less<>{ });

/* @brief Ranks elements
 *
 * Arranges processes in a grid. If the number of processes is a
 * square root, the height and the width are the same. Otherwise,
 * the grid contains one additional row which contains the
 * remaining processes.
 *
 * 1. Sort input and perform column-wise All-gather-merge ('column data').
 *
 * 2. Transpose data; process (i, j) sends element to process (j, i) ('row data').
 * Last row may not do anything.
 *
 * 3. Rank row data into column data.
 *
 * 4. Allreduce ranks. The processes of a row now store the same elements and ranks.
 *
 * 5. The i'th process of a row returns the i'th largest ranked row data.
 *
 * @param comm RBC communcator
 *
 * @param els Local input. If the global number of elements is less
 *   than comm.getSize(), it is expected that processes [0, global
 *   number of elements) store one element each. The algorithm works
 *   without this condition but may be slower than expected.
 *
 * @return Returns a vector of tuples storing pairs of elements and
 *   their rank value. The type RankType has to be a integral type.
 */
template <class T, class RankType, class CompLess = std::less<>, class RFisTags = RFis::Tags<> >
std::vector<Tools::Tuple<T, RankType> > Rank(MPI_Datatype mpi_datatype,
                                                   MPI_Datatype rank_type,
                                                   std::vector<T> els,
                                                   const RBC::Comm& comm,
                                                   CompLess comp = std::less<>{ });

/* @brief Ranks elements
 *
 * Same as function above with RankType = int.
 */
template <class T, class CompLess = std::less<>, class RFisTags = RFis::Tags<> >
std::vector<Tools::Tuple<T, int> > Rank(MPI_Datatype mpi_datatype,
                                              std::vector<T> els,
                                              const RBC::Comm& comm,
                                              CompLess comp = std::less<>{ });
}  // namespace RFis

#include "../../src/RFis/RFis.hpp"
