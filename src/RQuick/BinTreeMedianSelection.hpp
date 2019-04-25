/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiS).
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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include <tlx/math.hpp>

#include "../../include/RBC/RBC.hpp"

#include "./RandomBitStore.hpp"

namespace BinTreeMedianSelection {
template <class Iterator, class Comp>
typename std::iterator_traits<Iterator>::value_type select(Iterator begin, Iterator end,
                                                           size_t n, Comp&& comp,
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
                   Comp&& comp,
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
                                                           size_t n, Comp&& comp,
                                                           MPI_Datatype mpi_type,
                                                           std::mt19937_64& async_gen,
                                                           RandomBitStore& bit_gen,
                                                           int tag, RBC::Comm& comm) {
  using T = typename std::iterator_traits<Iterator>::value_type;

  const auto myrank = comm.getRank();
  const auto nprocs = comm.getSize();

  assert(static_cast<size_t>(end - begin) <= n);

  std::vector<T> v(begin, end);
  std::vector<T> recv_v;
  std::vector<T> tmp_v;

  v.reserve(2 * n);
  recv_v.reserve(2 * n);
  tmp_v.reserve(2 * n);

  assert(std::is_sorted(begin, end, std::forward<Comp>(comp)));

  const auto tailing_zeros = static_cast<unsigned>(tlx::ffs(myrank)) - 1;
  const auto logp = tlx::integer_log2_floor(nprocs);
  const auto iterations = std::min(tailing_zeros, logp);

  for (size_t it = 0; it != iterations; ++it) {
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
