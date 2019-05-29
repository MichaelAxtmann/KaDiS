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
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "RandomBitStore.hpp"

#include <tlx/math.hpp>

#include <RBC.hpp>

namespace BinTreeMedianSelection {
template <class Iterator, class Comp>
typename std::iterator_traits<Iterator>::value_type select(Iterator begin, Iterator end,
                                                           size_t n, Comp comp,
                                                           MPI_Datatype mpi_type,
                                                           std::mt19937_64& async_gen,
                                                           RandomBitStore& bit_gen,
                                                           int tag, RBC::Comm& comm);

namespace _internal {
template <class T, class Comp>
void selectMedians(std::vector<T>& recv_v,
                   std::vector<T>& tmp_v,
                   std::vector<T>& vs,
                   size_t n,
                   Comp comp,
                   std::mt19937_64& async_gen,
                   RandomBitStore& bit_gen) {
  assert(recv_v.size() <= n);
  assert(vs.size() <= n);

  tmp_v.resize(recv_v.size() + vs.size());
  std::merge(recv_v.begin(), recv_v.end(), vs.begin(), vs.end(),
             tmp_v.begin(), std::forward<Comp>(comp));

  if (tmp_v.size() <= n) {
    vs.swap(tmp_v);
    assert(std::is_sorted(vs.begin(), vs.end(), std::forward<Comp>(comp)));
    return;
  } else {
    if ((tmp_v.size() - n) % 2 == 0) {
      const auto offset = (tmp_v.size() - n) / 2;
      assert(offset + n < tmp_v.size());
      const auto begin = tmp_v.begin() + offset;
      const auto end = begin + n;
      vs.clear();
      vs.insert(vs.end(), begin, end);
      return;
    } else {
      // We cannot remove the same number of elements at
      // the right and left end.
      const auto offset = (tmp_v.size() - n) / 2;
      const auto padding_cnt = bit_gen.getNextBit(async_gen);
      assert(padding_cnt <= 1);
      assert(offset + padding_cnt + n <= tmp_v.size());
      auto begin = tmp_v.begin() + offset + padding_cnt;
      auto end = begin + n;
      vs.clear();
      vs.insert(vs.end(), begin, end);
      return;
    }
  }
}

template <class T>
T selectMedian(const std::vector<T>& v,
               std::mt19937_64& async_gen,
               RandomBitStore& bit_gen) {
  if (v.size() == 0) {
    return T { };
  }

  assert(v.size() > 0);
  if (v.size() % 2 == 0) {
    if (bit_gen.getNextBit(async_gen)) {
      return v[v.size() / 2];
    } else {
      return v[(v.size() / 2) - 1];
    }
  } else {
    return v[v.size() / 2];
  }
}
}     // namespace _internal

template <class Iterator, class Comp>
typename std::iterator_traits<Iterator>::value_type select(Iterator begin, Iterator end,
                                                           size_t n, Comp comp,
                                                           MPI_Datatype mpi_type,
                                                           std::mt19937_64& async_gen,
                                                           RandomBitStore& bit_gen,
                                                           int tag, const RBC::Comm& comm) {
  using T = typename std::iterator_traits<Iterator>::value_type;

  const auto myrank = comm.getRank();

  assert(static_cast<size_t>(end - begin) <= n);

  std::vector<T> v(begin, end);
  std::vector<T> recv_v;
  std::vector<T> tmp_v;

  v.reserve(2 * n);
  recv_v.reserve(2 * n);
  tmp_v.reserve(2 * n);

  assert(std::is_sorted(begin, end, std::forward<Comp>(comp)));

  // Reduce.
  const int tailing_zeros = tlx::ffs(comm.getRank()) - 1;
  const int iterations = comm.getRank() > 0 ?
                         tailing_zeros : tlx::integer_log2_ceil(comm.getSize());

  for (int it = 0; it != iterations; ++it) {
    const auto source = myrank + (1 << it);

    MPI_Status status;
    RBC::Probe(source, tag, comm, &status);
    int count = 0;
    MPI_Get_count(&status, mpi_type, &count);
    assert(static_cast<size_t>(count) <= n);
    recv_v.resize(count);
    RBC::Recv(recv_v.data(), count, mpi_type, source, tag, comm, MPI_STATUS_IGNORE);

    _internal::selectMedians(recv_v, tmp_v, v, n,
                             std::forward<Comp>(comp), async_gen, bit_gen);
  }
  if (myrank == 0) {
    auto median = _internal::selectMedian(v, async_gen, bit_gen);
    RBC::Bcast(&median, 1, mpi_type, 0, comm);
    return median;
  } else {
    int target = myrank - (1 << tailing_zeros);
    assert(v.size() <= n);
    RBC::Send(v.data(), v.size(), mpi_type, target, tag, comm);

    T median;
    RBC::Bcast(&median, 1, mpi_type, 0, comm);
    return median;
  }
}
}  // namespace BinTreeMedianSelection
