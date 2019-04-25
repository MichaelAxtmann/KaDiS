/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiSo).
 *
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
 * @param comm MPI commuicator (all ranks have to call the function)
 * @param comp The compare operator
 * @param is_robust Algorithm is executed robustly if set to true. We
 *  recommend to execute the algorithm robustly. Otherwise, data
 *  imbalances may occur. Use the nonrobust version only if the input does
 *  not contain duplicate keys and if the elements are randomly
 *  distributed.
 */template <class Tracker, class T, class Comp = std::less<T> >
void sort(Tracker&& tracker,
          std::mt19937_64& async_gen,
          std::vector<T>& v,
          MPI_Datatype mpi_type,
          int tag,
          MPI_Comm comm,
          Comp&& comp = Comp(),
          bool is_robust = true);

/**
 * Robust Quicksort with RBC communicators
 *
 * Sorts the input data with a custom compare operator
 * @param async_gen Random generator with different seeds for each process
 * @param v Vector that contains the input data
 * @param tag The tag which is used by Robust Quicksort
 * @param comm MPI commuicator (all ranks have to call the function)
 * @param comp The compare operator
 * @param is_robust Algorithm is executed robustly if set to true. We
 *  recommend to execute the algorithm robustly. Otherwise, data
 *  imbalances may occur. Use the nonrobust version only if the input does
 *  not contain duplicate keys and if the elements are randomly
 *  distributed.
 */
template <class T, class Comp = std::less<T> >
void sort(std::mt19937_64& async_gen,
          std::vector<T>& v,
          MPI_Datatype mpi_type,
          int tag,
          MPI_Comm comm,
          Comp&& comp = Comp(),
          bool is_robust = true);

/**
 * Robust Quicksort with RBC communicators
 *
 * Sorts the input data with a custom compare operator
 * @param tracker The tracker to store times of subroutines
 * @param async_gen Random generator with different seeds for each process
 * @param v Vector that contains the input data
 * @param tag The tag which is used by Robust Quicksort
 * @param comm RBC commuicator (all ranks have to call the function)
 * @param comp The compare operator
 * @param is_robust Algorithm is executed robustly if set to true. We
 *  recommend to execute the algorithm robustly. Otherwise, data
 *  imbalances may occur. Use the nonrobust version only if the input does
 *  not contain duplicate keys and if the elements are randomly
 *  distributed.
 */
template <class Tracker, class T, class Comp = std::less<T> >
void sort(Tracker&& tracker,
          std::mt19937_64& async_gen,
          std::vector<T>& v,
          MPI_Datatype mpi_type,
          int tag,
          RBC::Comm& comm,
          Comp&& comp = Comp(),
          bool is_robust = true);

/**
 * Robust Quicksort with RBC communicators
 *
 * Sorts the input data with a custom compare operator
 * @param async_gen Random generator with different seeds for each process
 * @param v Vector that contains the input data
 * @param tag The tag which is used by Robust Quicksort
 * @param comm RBC commuicator (all ranks have to call the function)
 * @param comp The compare operator
 * @param is_robust Algorithm is executed robustly if set to true. We
 *  recommend to execute the algorithm robustly. Otherwise, data
 *  imbalances may occur. Use the nonrobust version only if the input does
 *  not contain duplicate keys and if the elements are randomly
 *  distributed.
 */
template <class T, class Comp = std::less<T> >
void sort(std::mt19937_64& async_gen,
          std::vector<T>& v,
          MPI_Datatype mpi_type,
          int tag,
          RBC::Comm& comm,
          Comp&& comp = Comp(),
          bool is_robust = true);
}  // namespace RQuick

#include "../../src/RQuick/RQuick.hpp"
