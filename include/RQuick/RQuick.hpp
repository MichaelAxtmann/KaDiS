/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiSo).
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

#include "RBC.hpp"

namespace RQuick {
/**
 * Robust Quicksort with RBC communicators
 *
 * Sorts the input data with a custom compare operator
 * @param tracker The tracker to store times of subroutines
 * @param async_gen Random generator with different seeds for each process
 * @param v Vector that contains the input data
 * @param tag The tag which is used by Robust Quicksort
 * @param comm MPI commuicator
 * @param comp The compare operator
 * @param is_robust Algorithm is executed robustly if set to true. We
 *  recommend to execute the algorithm robustly. Otherwise, data
 *  imbalances may occur. Use the nonrobust version only if the input does
 *  not contain duplicate keys and if the elements are randomly
 *  distributed.
 */template <class Tracker, class T, class Comp = std::less<T> >
void sort(Tracker&& tracker,
          MPI_Datatype mpi_type,
          std::vector<T>& v,
                    int tag,
          std::mt19937_64& async_gen,
MPI_Comm comm,
          Comp comp = Comp(),
          bool is_robust = true);

/**
 * Robust Quicksort with RBC communicators
 *
 * Sorts the input data with a custom compare operator
 * @param async_gen Random generator with different seeds for each process
 * @param v Vector that contains the input data
 * @param tag The tag which is used by Robust Quicksort
 * @param comm MPI commuicator
 * @param comp The compare operator
 * @param is_robust Algorithm is executed robustly if set to true. We
 *  recommend to execute the algorithm robustly. Otherwise, data
 *  imbalances may occur. Use the nonrobust version only if the input does
 *  not contain duplicate keys and if the elements are randomly
 *  distributed.
 */
template <class T, class Comp = std::less<T> >
void sort(MPI_Datatype mpi_type,
          std::vector<T>& v,
          int tag,
          std::mt19937_64& async_gen,
          MPI_Comm comm,
          Comp comp = Comp(),
          bool is_robust = true);

/**
 * Robust Quicksort with RBC communicators
 *
 * Sorts the input data with a custom compare operator
 * @param tracker The tracker to store times of subroutines
 * @param async_gen Random generator with different seeds for each process
 * @param v Vector that contains the input data
 * @param tag The tag which is used by Robust Quicksort
 * @param comm RBC commuicator
 * @param comp The compare operator
 * @param is_robust Algorithm is executed robustly if set to true. We
 *  recommend to execute the algorithm robustly. Otherwise, data
 *  imbalances may occur. Use the nonrobust version only if the input does
 *  not contain duplicate keys and if the elements are randomly
 *  distributed.
 */
template <class Tracker, class T, class Comp = std::less<T> >
void sort(Tracker&& tracker,
          MPI_Datatype mpi_type,
          std::vector<T>& v,
          int tag,
          std::mt19937_64& async_gen,
          const RBC::Comm& comm,
          Comp comp = Comp(),
          bool is_robust = true);

/**
 * Robust Quicksort with RBC communicators
 *
 * Sorts the input data with a custom compare operator
 * @param async_gen Random generator with different seeds for each process
 * @param v Vector that contains the input data
 * @param tag The tag which is used by Robust Quicksort
 * @param comm RBC commuicator
 * @param comp The compare operator
 * @param is_robust Algorithm is executed robustly if set to true. We
 *  recommend to execute the algorithm robustly. Otherwise, data
 *  imbalances may occur. Use the nonrobust version only if the input does
 *  not contain duplicate keys and if the elements are randomly
 *  distributed.
 */
template <class T, class Comp = std::less<T> >
void sort(MPI_Datatype mpi_type,
          std::vector<T>& v,
          int tag,
          std::mt19937_64& async_gen,
          const RBC::Comm& comm,
          Comp comp = Comp(),
          bool is_robust = true);
}  // namespace RQuick

#include "../../src/RQuick/RQuick.hpp"
