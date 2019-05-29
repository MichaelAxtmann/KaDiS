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

#include <algorithm>
#include <cassert>
#include <vector>

#include "../../include/Tools/MpiTuple.hpp"
#include "../../include/MiniSort/Tags.hpp"
#include "../Tools/CommonMpi.hpp"

#include <RBC.hpp>

namespace MiniSort {
template <class T, class Comp, class Tags>
T sort(MPI_Datatype mpi_datatype,
       T el, const RBC::Comm& comm, Comp comp);

namespace _internal {
template <class T>
using TieBreakerType = Tools::Tuple<T, int>;

inline
int intLog3Floor(int i) {
  int p = 0;

  i /= 3;
  while (i) {
    i /= 3;
    ++p;
  }

  return p;
}

inline
int intPowOfThree(int i) {
  int p = 1;
  for (int j = 1; j <= i; ++j) {
    p *= 3;
  }
  return p;
}

inline
std::vector<int> tarnaryLevelSizes(int rank, int num_levels, int next_pow_of_three) {
  std::vector<int> res(num_levels);
  for (int it = num_levels - 1; it >= 0; --it) {
    next_pow_of_three /= 3;
    res[it] = rank / next_pow_of_three;
    rank = rank % next_pow_of_three;
  }
  return res;
}

// Ternary-tree median-selection. The algorithm guarantees that
// there is always one element less or equal to the return value.
template <class Tags, class T, class Comp>
TieBreakerType<T> pickPivot(int size, int rank,
                            MPI_Datatype mpi_datatype,
                            TieBreakerType<T> el,
                            Comp tb_comp, const RBC::Comm& comm) {
  assert(size > 1);
  assert(size == comm.getSize());
  assert(rank == comm.getRank());

  if (size == 2) {
    TieBreakerType<T> recv;
    int partner = (rank + 1) % 2;
    RBC::Sendrecv(&el,
                  1,
                  mpi_datatype,
                  partner,
                  Tags::kSecondTag,
                  &recv,
                  1,
                  mpi_datatype,
                  partner,
                  Tags::kSecondTag,
                  comm,
                  MPI_STATUSES_IGNORE);
    return std::min(el, recv, tb_comp);
  }

  int num_levels = intLog3Floor(size);
  int prev_pow_of_three = intPowOfThree(num_levels);

  if (rank < prev_pow_of_three) {
    RBC::Request requests[2];
    int off = 1;
    TieBreakerType<T> a, b;
    const auto size_three_base = tarnaryLevelSizes(rank, num_levels, prev_pow_of_three);

    // Receiving
    int round = 0;
    for ( ; round < num_levels; ++round) {
      if (size_three_base[round] == 0) {
        RBC::Irecv(&a, 1, mpi_datatype, rank + off * 1, Tags::kSecondTag, comm, requests);
        RBC::Irecv(&b, 1, mpi_datatype, rank + off * 2, Tags::kSecondTag, comm, requests + 1);
        RBC::Waitall(2, requests, MPI_STATUSES_IGNORE);

        el = std::max(
          std::min(a, b, tb_comp),
          std::min(el, std::max(a, b, tb_comp), tb_comp), tb_comp);
        off *= 3;
      } else {
        break;
      }
    }

    if (rank != 0) {
      int target = rank - off * size_three_base[round];
      RBC::Send(&el, 1, mpi_datatype, target, Tags::kSecondTag, comm);
    }
  }
  RBC::Bcast(&el, 1, mpi_datatype, 0, comm);
  return el;
}

template <class Tags, class T, class Comp>
TieBreakerType<T> sort(int size,
                       int rank,
                       MPI_Datatype mpi_datatype,
                       TieBreakerType<T> el,
                       Comp comp,
                       const RBC::Comm& comm) {
  if (size == 1) {
    return el;
  }

    const auto pivot = pickPivot<Tags>(size, rank,
                               mpi_datatype, el,
                               comp, comm);

  // The pivot selection guarantees that there is always one
  // element smaller or equal to the pivot. Thus, we use the
  // comparison pivot < el which guarantees that each process
  // group has at least one process.
  int right_group_el = comp(pivot, el);

  int right_scan = 0;
  int right_group_size = 0;
  RBC::ScanAndBcast(&right_group_el, &right_scan, &right_group_size,
                    1, Common::getMpiType(right_group_el), MPI_SUM, comm);
  const int right_exscan = right_scan - right_group_el;
  const int left_exscan = rank - right_exscan;
  const int left_group_size = size - right_group_size;

  TieBreakerType<T> recv;
  if (right_group_el) {
    int target = left_group_size + right_exscan;

    RBC::Sendrecv(&el,
                  1,
                  mpi_datatype,
                  target,
                  Tags::kFirstTag,
                  &recv,
                  1,
                  mpi_datatype,
                  MPI_ANY_SOURCE,
                  Tags::kFirstTag,
                  comm,
                  MPI_STATUSES_IGNORE);
  } else {
    int target = left_exscan;

    RBC::Sendrecv(&el,
                  1,
                  mpi_datatype,
                  target,
                  Tags::kFirstTag,
                  &recv,
                  1,
                  mpi_datatype,
                  MPI_ANY_SOURCE,
                  Tags::kFirstTag,
                  comm,
                  MPI_STATUSES_IGNORE);
  }

  RBC::Comm sub_comm;

  if (rank < left_group_size) {
    RBC::Comm_create_group(comm, &sub_comm, 0, left_group_size - 1);
    return sort<Tags>(left_group_size, rank, mpi_datatype,
                recv, comp, sub_comm);
  } else {
    RBC::Comm_create_group(comm, &sub_comm, left_group_size, size - 1);
    return sort<Tags>(right_group_size, rank - left_group_size, mpi_datatype,
                recv, comp, sub_comm);
  }
}
}  // namespace _internal

template <class T, class Comp, class Tags>
T sort(MPI_Datatype mpi_datatype,
       T el,
       const RBC::Comm& comm,
       Comp comp) {
  const int size = comm.getSize();
  const int rank = comm.getRank();

  if (size == 1) return el;

  using Tb = _internal::TieBreakerType<T>;
  MPI_Datatype second_type = Common::getMpiType<typename Tb::second_type>();

  const MPI_Datatype mpi_tb_type = Tb::MpiType(mpi_datatype, second_type);
  auto tb_comp = [&comp](const Tb& a, const Tb& b) {
                   return comp(a.first, b.first) ||
                          (!comp(b.first, a.first) &&
                            a.second < b.second);
                 };
  Tb tb_el;
  tb_el.first = el;
  tb_el.second = rank;
  const auto val = _internal::sort<Tags>(size, rank,
                                   mpi_tb_type, tb_el,
                                   tb_comp, comm);
  return val.first;
}

template <class T, class Comp, class Tags>
T sort(MPI_Datatype mpi_datatype,
       T el,
       MPI_Comm comm,
       Comp comp) {
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  return sort<T, Comp, Tags>(mpi_datatype,
              el,
              rcomm,
              comp);
}
}  // end namespace MiniSort
