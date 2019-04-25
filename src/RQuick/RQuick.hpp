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

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "../../include/RBC/RBC.hpp"
#include "ips4o.hpp"
#include "tlx/algorithm.hpp"
#include "tlx/math.hpp"

#include "../Tools/DummyTimer.hpp"
#include "./BinTreeMedianSelection.hpp"
#include "./RandomBitStore.hpp"

namespace RQuick {
namespace _internal {
inline
void split(RBC::Comm& comm, RBC::Comm* subcomm) {
  const auto nprocs = comm.getSize();
  const auto myrank = comm.getRank();

  bool is_left_group = myrank < (nprocs / 2);

  int first = 0, last = 0;
  if (is_left_group) {
    first = 0;
    last = nprocs / 2 - 1;
  } else {
    first = nprocs / 2;
    last = nprocs - 1;
  }

  RBC::Comm_create_group(comm, subcomm, first, last);
}

/*
 * @brief Returns the k middle most elements.
 *
 * Returns the k middle most elements of v if v contains at least k
 * elements. Otherwise, v is returned. This function uses
 * randomization to decide which elements are the 'middle most'
 * elements if v contains an odd number of elements and k is an even
 * number or if v contains an even number of elements and k is an odd
 * number.
 */
template <class T>
std::vector<T> middleMostElements(const std::vector<T>& v, size_t k,
                                  std::mt19937_64& async_gen, RandomBitStore& bit_gen) {
  if (v.size() <= k) {
    return std::vector<T>(v.begin(), v.end());
  }

  const auto offset = (v.size() - k) / 2;

  if (((v.size() % 2) == 0 && ((k % 2) == 0)) ||
      (v.size() % 2 == 1 && ((k % 2) == 1))) {
    return std::vector<T>(
      v.data() + offset,
      v.data() + offset + k);
  } else {
    // Round down or round up randomly.

    const auto shift = bit_gen.getNextBit(async_gen);
    return std::vector<T>(
      v.data() + offset + shift,
      v.data() + offset + k + shift);
  }
}

/*
 * @brief Distributed splitter selection with a binary reduction tree.
 *
 * @param v Local input. The input must be sorted.
 */
template <class T, class Comp>
T selectSplitter(std::mt19937_64& async_gen,
                 RandomBitStore& bit_gen, const std::vector<T>& v,
                 MPI_Datatype mpi_type, Comp&& comp, int tag,
                 RBC::Comm& comm) {
  assert(std::is_sorted(v.begin(), v.end(), comp));

  const auto local_medians = middleMostElements(v, 2, async_gen, bit_gen);

  const auto s = BinTreeMedianSelection::select(
    local_medians.data(), local_medians.data() + local_medians.size(), 2,
    std::forward<Comp>(comp), mpi_type, async_gen, bit_gen, tag, comm);

  return s;
}

/*
 * @brief Split vector according to a given splitter with or without tie-breaking.
 *
 * No tie-breaking: Returns a pointer to the first element which is
 * larger or equal to the splitter.
 *
 * With tie-breaking: Splits the vector according to a given splitter
 * such that the split is as close to the middle of the vector as
 * possible.
 *
 * @param v Local input. The input must be sorted.
 */
template <class T, class Comp>
const T* locateSplitter(const std::vector<T>& v, Comp&& comp, const T& splitter,
                        std::mt19937_64& gen, RandomBitStore& bit_store,
                        bool is_robust) {
  const auto begin_equal_els = std::lower_bound(v.data(), v.data() + v.size(),
                                                splitter, std::forward<Comp>(comp));

  if (!is_robust) {
    return begin_equal_els;
  }

  const auto end_equal_els = std::upper_bound(begin_equal_els, v.data() + v.size(),
                                              splitter, std::forward<Comp>(comp));

  // Round down or round up randomly.
  const auto opt_split = v.data()
                         + v.size() / 2
                         + ((v.size() % 2) == 1 && bit_store.getNextBit(gen));

  if (begin_equal_els < opt_split) {
    return std::min(opt_split, end_equal_els);
  } else {
    return begin_equal_els;
  }
}

/*
 * @brief Partitions two sequences according to a specific rank.
 *
 * Split two sequences into a left and a right part each. The left
 * parts contain 'rank' elements in total and the left parts only
 * contain elements which are smaller than elements of the right
 * parts. Tie-breaking is used meaning that elements of the first
 * sequence are smaller than the same elements of the second sequence
 * and equal elements of the same sequence use their index as
 * tie-breaker.
 *
 * @param begin1 Begin of the first sorted sequence.
 * @param end1 End of the first sorted sequence.
 * @param begin2 Begin of the second sorted sequence.
 * @param end2 End of the second sorted sequence.
 * @param rank Total size of the left part. Rank must be a positive
 *  number smaller of equal to the total size of both input sequences.
 * @param comp The comparator.
 */
template <class T, class Comp>
std::pair<const T*, const T*> twoSequenceSelection(const T* begin1, const T* end1,
                                                   const T* begin2, const T* end2,
                                                   int64_t rank, Comp&& comp) {
  assert(std::is_sorted(begin1, end1, std::forward<Comp>(comp)));
  assert(std::is_sorted(begin2, end2, std::forward<Comp>(comp)));

  assert(static_cast<size_t>((end1 - begin1) + (end2 - begin2)) >= rank);

  // Shrink sequences by taking splitters from the first
  // sequence until first sequence does not contain any
  // elements.
  while (end1 != begin1) {
    assert(end1 > begin1);

    const auto offset1 = (end1 - begin1) / 2;
    const auto splitter1 = begin1 + offset1;
    assert(begin1 <= splitter1);
    assert(splitter1 < end1);

    // We use tie-breaking. Meaning than an element 'a' in
    // the first sequence is smaller than the same element
    // 'a' in the second sequence. Thus, the function
    // 'lower_bound' breaks ties implicitly.
    const auto splitter2 = std::lower_bound(begin2, end2, *splitter1,
                                            std::forward<Comp>(comp));
    const auto offset2 = splitter2 - begin2;

    const auto new_rank = offset1 + offset2;

    if (rank < new_rank) {
      // Continue on left part.

      // Size of first sequence decreases by at least
      // one as 'splitter1' is now excluded.

      end1 = splitter1;
      end2 = splitter2;
    } else if (rank > new_rank) {
      // Continue on right part.

      // Size of the first sequence decreases by at least one
      // (even if the frist element is selected as 'splitter1')
      // as 'splitter1' is now excluded. We can exclude
      // 'splitter1' from the first sequence, as all elements in
      // [splitter2, end2) as they are larger than 'splitter1'
      // (ties are already broken). However, as 'rank >
      // new_rank', we can remove the smallest element.

      rank -= new_rank + 1;
      begin1 = splitter1 + 1;
      begin2 = splitter2;
    } else {
      return {
               splitter1, splitter2
      };
    }
  }

  // Position in first sequence has been found. We still
  // need 'rank' elements. Those 'rank' elements must be the
  // first elements of the second sequence.
  return {
           begin1, begin2 + rank
  };
}

template <class T>
void exchange(const T* send_begin, const T* send_end,
              std::vector<T>& v_recv,
              int target, MPI_Datatype mpi_type,
              int tag, RBC::Comm& comm) {
  const auto send_size = send_end - send_begin;

  RBC::Request requests[2];
  RBC::Isend(send_begin, send_size, mpi_type, target,
             tag, comm, requests);

  int recv_size = 0;
  MPI_Status status;
  RBC::Probe(target, tag, comm, &status);
  MPI_Get_count(&status, mpi_type, &recv_size);

  v_recv.resize(recv_size);
  RBC::Irecv(v_recv.data(), recv_size, mpi_type, target,
             tag, comm, requests + 1);
  RBC::Waitall(2, requests, MPI_STATUSES_IGNORE);
}

template <class T, class Comp>
void merge(const T* begin1, const T* end1,
           const T* begin2, const T* end2,
           T* tbegin, Comp&& comp) {
  assert(std::is_sorted(begin1, end1, std::forward<Comp>(comp)));
  assert(std::is_sorted(begin2, end2, std::forward<Comp>(comp)));

#ifdef _OPENMP

  const auto num_elements = (end1 - begin1) + (end2 - begin2);

#pragma omp parallel
  {
    const auto num_threads = omp_get_num_threads();
    const auto id = omp_get_thread_num();

    const auto stripe = tlx::div_ceil(num_elements, num_threads);

    const auto rank_begin = std::min(id * stripe, num_elements);
    const auto rank_end = std::min((id + 1) * stripe, num_elements);

    auto[s1_begin, s2_begin] = twoSequenceSelection(begin1, end1, begin2, end2,
                                                    rank_begin, std::forward<Comp>(comp));
    assert((s1_begin - begin1) + (s2_begin - begin2) == rank_begin);

    auto[s1_end, s2_end] = twoSequenceSelection(s1_begin, end1, s2_begin, end2,
                                                rank_end - rank_begin, std::forward<Comp>(comp));
    assert(s1_begin <= s1_end);
    assert(s2_begin <= s2_end);
    assert((s1_end - begin1) + (s2_end - begin2) == rank_end);

    std::merge(s1_begin, s1_end, s2_begin, s2_end, tbegin + rank_begin, std::forward<Comp>(comp));
  }

#else

  std::merge(begin1, end1, begin2, end2, tbegin, std::forward<Comp>(comp));

#endif
}

template <class T, class Tracker, class Comp>
void sortRec(std::mt19937_64& gen, RandomBitStore& bit_store,
             std::vector<T>& v, std::vector<T>& v_merge, std::vector<T>& v_recv,
             Comp&& comp, MPI_Datatype mpi_type, bool is_robust,
             Tracker&& tracker, int tag, RBC::Comm& comm) {
  tracker.median_select_t.start(comm);

  const auto nprocs = comm.getSize();
  const auto myrank = comm.getRank();

  assert(nprocs >= 2);
  assert(std::is_sorted(v.begin(), v.end(), std::forward<Comp>(comp)));
  assert(tlx::integer_log2_floor(nprocs));

  const auto is_left_group = myrank < nprocs / 2;

  // Select pivot globally with binary tree median selection.
  const auto pivot = selectSplitter(gen, bit_store, v,
                                    mpi_type,
                                    std::forward<Comp>(comp),
                                    tag, comm);

  tracker.median_select_t.stop();

  // Partition data into small elements and large elements.

  tracker.partition_t.start(comm);
  const auto* separator = locateSplitter(v, std::forward<Comp>(comp), pivot,
                                         gen, bit_store, is_robust);

  const auto* send_begin = v.data();
  const auto* send_end = separator;
  const auto* own_begin = separator;
  const auto* own_end = v.data() + v.size();
  if (is_left_group) {
    std::swap(send_begin, own_begin);
    std::swap(send_end, own_end);
  }

  tracker.partition_t.stop();

  // Move elements to partner and receive elements for own group.
  tracker.exchange_t.start(comm);

  const auto partner = (myrank + (nprocs / 2)) % nprocs;
  exchange(send_begin, send_end, v_recv, partner,
           mpi_type, tag, comm);

  tracker.exchange_t.stop();

  // Merge received elements with own elements.
  tracker.merge_t.start(comm);

  const auto num_elements = v_recv.size() + (own_end - own_begin);
  v_merge.resize(num_elements);
  merge(own_begin, own_end, v_recv.begin(), v_recv.end(),
        v_merge.begin(), std::forward<Comp>(comp));
  v.swap(v_merge);
  assert(std::is_sorted(v.begin(), v.end(), std::forward<Comp>(comp)));

  tracker.merge_t.stop();

  if (nprocs >= 4) {
    // Split communicator and solve subproblems.
    tracker.comm_split_t.start(comm);

    RBC::Comm subcomm;
    split(comm, &subcomm);

    tracker.comm_split_t.stop();

    sortRec(gen, bit_store, v, v_merge, v_recv, std::forward<Comp>(comp),
            mpi_type, is_robust, std::forward<Tracker>(tracker), tag, subcomm);
  }
}

template <class T>
void shuffle(std::mt19937_64& async_gen,
             std::vector<T>& v,
             std::vector<T>& v_tmp,
             MPI_Datatype mpi_type,
             int tag,
             RBC::Comm& comm) {
  // Just used for OpenMP
  std::ignore = v_tmp;

  const size_t nprocs = comm.getSize();
  const size_t myrank = comm.getRank();

  // Generate a random bit generator for each thread. Using pointers
  // is faster than storing the generators in one vector due to
  // cache-line sharing.
#ifdef _OPENMP
  int num_threads = 1;
  std::vector<std::unique_ptr<std::mt19937_64> > omp_async_gens;
  std::vector<size_t> seeds;
#pragma omp parallel
  {
#pragma omp single
    {
      num_threads = omp_get_num_threads();
      omp_async_gens.resize(num_threads);
      for (int i = 0; i != num_threads; ++i) {
        seeds.push_back(async_gen());
      }
    }

    const int id = omp_get_thread_num();

    omp_async_gens[id].reset(new std::mt19937_64 { seeds[id] });
  }

  std::vector<size_t> sizes(2 * num_threads);
  std::vector<size_t> prefix_sum(2 * num_threads + 1);

#endif

  const size_t comm_phases = tlx::integer_log2_floor(nprocs);

  for (size_t phase = 0; phase != comm_phases; ++phase) {
    const size_t mask = 1 << phase;
    const size_t partner = myrank ^ mask;

#ifdef _OPENMP

    int send_cnt = 0;
    int own_cnt = 0;
    int recv_cnt = 0;

#pragma omp parallel
    {
      const int id = omp_get_thread_num();

      const size_t stripe = tlx::div_ceil(v.size(), num_threads);
      std::unique_ptr<T[]> partitions[2] =
      { std::unique_ptr<T[]>(new T[stripe]),
        std::unique_ptr<T[]>(new T[stripe]) };

      const auto begin = v.data() + id * stripe;
      const auto end = v.data() + std::min((id + 1) * stripe, v.size());

      auto& mygen = *omp_async_gens[id].get();
      T* partptrs[2] = { partitions[0].get(), partitions[1].get() };

      const auto size = end - begin;
      const auto remaining = size % (8 * sizeof(std::mt19937_64::result_type));
      auto ptr = begin;
      while (ptr < end - remaining) {
        auto rand = mygen();
        for (size_t j = 0; j < 8 * sizeof(std::mt19937_64::result_type); ++j) {
          const auto bit = rand & 1;
          rand = rand >> 1;
          *partptrs[bit] = *ptr;
          ++partptrs[bit];
          ++ptr;
        }
      }

      auto rand = mygen();
      while (ptr < end) {
        // for (size_t i = 0; i != remaining; ++i) {
        const auto bit = rand & 1;
        rand = rand >> 1;
        *partptrs[bit] = *ptr;
        ++partptrs[bit];
        ++ptr;
      }

      sizes[id] = partptrs[0] - partitions[0].get();
      sizes[num_threads + id] = partptrs[1] - partitions[1].get();

#pragma omp barrier
#pragma omp master
      {
        tlx::exclusive_scan(sizes.begin(), sizes.end(),
                            prefix_sum.begin(), size_t { 0 });

        send_cnt = prefix_sum[2 * num_threads] - prefix_sum[num_threads];
        own_cnt = prefix_sum[num_threads];
        recv_cnt = 0;
        RBC::Sendrecv(&send_cnt, 1, MPI_INT, partner, tag,
                      &recv_cnt, 1, MPI_INT, partner, tag,
                      comm, MPI_STATUS_IGNORE);

        v.resize(own_cnt + recv_cnt);
        v_tmp.resize(send_cnt);
      }
#pragma omp barrier

      std::copy(partitions[0].get(), partitions[0].get() + sizes[id],
                v.data() + prefix_sum[id]);
      std::copy(partitions[1].get(), partitions[1].get() + sizes[num_threads + id],
                v_tmp.data() + prefix_sum[id + num_threads] - prefix_sum[num_threads]);
    }

    RBC::Sendrecv(v_tmp.data(), send_cnt, mpi_type, partner, tag,
                  v.data() + own_cnt, recv_cnt, mpi_type, partner, tag,
                  comm, MPI_STATUS_IGNORE);

#else

    std::unique_ptr<T[]> partition(new T[v.size()]);
    T* begins[2] = { v.data(), partition.get() };

    const auto size = v.size();
    const auto remaining = size % (8 * sizeof(std::mt19937_64::result_type));
    auto ptr = v.begin();
    const auto end = v.end();
    while (ptr < end - remaining) {
      auto rand = async_gen();
      for (size_t j = 0; j < 8 * sizeof(std::mt19937_64::result_type); ++j) {
        const auto bit = rand & 1;
        rand = rand >> 1;
        *begins[bit] = *ptr;
        ++begins[bit];
        ++ptr;
      }
    }

    auto rand = async_gen();
    while (ptr < end) {
      const auto bit = rand & 1;
      rand = rand >> 1;
      *begins[bit] = *ptr;
      ++begins[bit];
      ++ptr;
    }

    RBC::Request requests[2];

    RBC::Isend(partition.get(), (begins[1] - partition.get()),
               mpi_type, partner, tag, comm, requests);

    int count = 0;
    MPI_Status status;
    RBC::Probe(partner, tag, comm, &status);
    MPI_Get_count(&status, mpi_type, &count);

    v.resize(begins[0] - v.data() + count);

    RBC::Irecv(v.data() + v.size() - count, count, mpi_type,
               partner, tag, comm, requests + 1);
    RBC::Waitall(2, requests, MPI_STATUSES_IGNORE);

#endif
  }
}

template <class Iterator, class Comp>
void sortLocally(Iterator begin, Iterator end, Comp&& comp) {
#ifdef _OPENMP
  ips4o::parallel::sort(begin, end, std::forward<Comp>(comp));
#else
  ips4o::sort(begin, end, std::forward<Comp>(comp));
#endif
}

template <class Tracker, class T, class Comp>
void sort(std::mt19937_64& async_gen,
          std::vector<T>& v,
          MPI_Datatype mpi_type,
          int tag,
          RBC::Comm comm,
          Tracker&& tracker,
          Comp&& comp,
          bool is_robust) {
  if (comm.getSize() == 1) {
    tracker.local_sort_t.start(comm);
    sortLocally(v.begin(), v.end(), std::forward<Comp>(comp));
    tracker.local_sort_t.stop();

    return;
  }


  tracker.move_to_pow_of_two_t.start(comm);

  const auto pow = tlx::round_down_to_power_of_two(comm.getSize());

  // Send data to a smaller hypercube if the number of processes is
  // not a power of two.
  if (comm.getRank() < comm.getSize() - pow) {
    // Not a power of two but we are part of the smaller hypercube
    // and receive elements.

    MPI_Status status;
    const auto source = pow + comm.getRank();

    RBC::Probe(source, tag, comm, &status);
    int recv_cnt = 0;
    MPI_Get_count(&status, mpi_type, &recv_cnt);

    // Avoid reallocations later.
    v.reserve(2 * (v.size() + recv_cnt));

    v.resize(v.size() + recv_cnt);
    MPI_Request request;
    RBC::Irecv(v.data() + v.size() - recv_cnt, recv_cnt, mpi_type,
               source, tag, comm, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    RBC::Comm sub_comm;
    RBC::Comm_create_group(comm, &sub_comm, 0, pow - 1);
    comm = sub_comm;
  } else if (comm.getRank() >= pow) {
    // Not a power of two and we are not part of the smaller
    // hypercube.

    const auto target = comm.getRank() - pow;
    RBC::Send(v.data(), v.size(), mpi_type, target, tag, comm);
    v.clear();

    // This process is not part of 'sub_comm'. We call
    // this function to support MPI implementations
    // without MPI_Comm_create_group.
    RBC::Comm sub_comm;
    RBC::Comm_create_group(comm, &sub_comm, 0, pow - 1);
    comm = sub_comm;

    tracker.move_to_pow_of_two_t.stop();

    return;
  } else if (pow != comm.getSize()) {
    // Not a power of two but we are part of the smaller hypercube
    // and do not receive elements.

    RBC::Comm sub_comm;
    RBC::Comm_create_group(comm, &sub_comm, 0, pow - 1);
    comm = sub_comm;

    // Avoid reallocations later.
    v.reserve(3 * v.size());
  } else {
    // The number of processes is a power of two.

    // Avoid reallocations later.
    v.reserve(2 * v.size());
  }

  tracker.move_to_pow_of_two_t.stop();

  assert(tlx::is_power_of_two(comm.getSize()));

  tracker.parallel_shuffle_t.start(comm);

  // Vector is used to store about the same number of elements as v.
  std::vector<T> tmp1;
  tmp1.reserve(v.capacity());

  // Vector is used to store about half the number of elements as
  // v. This vector is used to receive data.
  std::vector<T> tmp2;
  tmp2.reserve(v.capacity() / 2);

  if (is_robust) {
    shuffle(async_gen, v, tmp2, mpi_type, tag, comm);
  }

  tracker.parallel_shuffle_t.stop();

  tracker.local_sort_t.start(comm);
  sortLocally(v.begin(), v.end(), std::forward<Comp>(comp));
  tracker.local_sort_t.stop();

  RandomBitStore bit_store;
  _internal::sortRec(async_gen, bit_store, v, tmp1, tmp2, std::forward<Comp>(comp),
                     mpi_type, is_robust, std::forward<Tracker>(tracker), tag, comm);
}

class DummyTracker {
 public:
  Tools::DummyTimer local_sort_t;
  Tools::DummyTimer exchange_t;
  Tools::DummyTimer parallel_shuffle_t;
  Tools::DummyTimer merge_t;
  Tools::DummyTimer median_select_t;
  Tools::DummyTimer partition_t;
  Tools::DummyTimer comm_split_t;
  Tools::DummyTimer move_to_pow_of_two_t;
};
}   // namespace _internal

template <class Tracker, class T, class Comp>
void sort(Tracker&& tracker,
          std::mt19937_64& async_gen,
          std::vector<T>& v,
          MPI_Datatype mpi_type,
          int tag,
          MPI_Comm mpi_comm,
          Comp&& comp,
          bool is_robust) {
  RBC::Comm comm;
  RBC::Create_Comm_from_MPI(mpi_comm, &comm);
  _internal::sort(async_gen, v, mpi_type,
                  tag, comm, std::forward<Tracker>(tracker),
                  std::forward<Comp>(comp), is_robust);
}

template <class T, class Comp>
void sort(std::mt19937_64& async_gen,
          std::vector<T>& v,
          MPI_Datatype mpi_type,
          int tag,
          MPI_Comm mpi_comm,
          Comp&& comp,
          bool is_robust) {
  _internal::DummyTracker tracker;
  sort(tracker, async_gen, v, mpi_type, tag, mpi_comm,
       comp, is_robust);
}

template <class Tracker, class T, class Comp>
void sort(Tracker&& tracker,
          std::mt19937_64& async_gen,
          std::vector<T>& v,
          MPI_Datatype mpi_type,
          int tag,
          RBC::Comm& comm,
          Comp&& comp,
          bool is_robust) {
  _internal::sort(async_gen, v, mpi_type,
                  tag, comm, std::forward<Tracker>(tracker),
                  std::forward<Comp>(comp), is_robust);
}

template <class T, class Comp>
void sort(std::mt19937_64& async_gen,
          std::vector<T>& v,
          MPI_Datatype mpi_type,
          int tag,
          RBC::Comm& comm,
          Comp&& comp,
          bool is_robust) {
  _internal::DummyTracker tracker;
  sort(tracker, async_gen, v, mpi_type, tag, comm,
       comp, is_robust);
}
}  // namespace RQuick
