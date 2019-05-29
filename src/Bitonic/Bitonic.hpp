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
#include <cstdint>
#include <functional>
#include <vector>

#include "../../include/Bitonic/Configuration.hpp"
#include "../Tools/CommonMpi.hpp"

#include <ips4o.hpp>
#include <RBC.hpp>
#include <tlx/math.hpp>

namespace Bitonic {
template <class T, class Comp>
void Sort(std::vector<T>& data, MPI_Datatype mpi_type, int tag, MPI_Comm comm,
          Comp comp, bool is_equal_input_size,
          Bitonic::PartitionBy partition_strategy);

template <class T, class Comp>
void SortBitonicSequence(std::vector<T>& data, MPI_Datatype mpi_type, int tag,
                         MPI_Comm comm, Comp comp, bool sort_increasing, bool
                         is_equal_input_size,
                         Bitonic::PartitionBy partition_strategy);

template <class T, class Comp>
void Sort(std::vector<T>& data, MPI_Datatype mpi_type, int tag, const RBC::Comm& comm,
          Comp comp, bool is_equal_input_size,
          Bitonic::PartitionBy partition_strategy);

template <class T, class Comp>
void SortBitonicSequence(std::vector<T>& data, MPI_Datatype mpi_type, int tag,
                         const RBC::Comm& comm, Comp comp, bool sort_increasing, bool
                         is_equal_input_size,
                         Bitonic::PartitionBy partition_strategy);


namespace _internal {
// Select the smallest elements.
template <class It, class Comp>
void MergeFirstElements(It first1, It last1, It first2, It last2, It result,
                        size_t el_cnt, Comp comp) {
  assert((last1 - first1) + (last2 - first2) >= static_cast<std::ptrdiff_t>(el_cnt));

  size_t processed_el_cnt = 0;
  while (processed_el_cnt < el_cnt) {
    if (first1 == last1) {
      std::copy(first2, first2 + (el_cnt - processed_el_cnt), result);
      return;
    }
    if (first2 == last2) {
      std::copy(first1, first1 + (el_cnt - processed_el_cnt), result);
      return;
    }
    *result++ = (comp(*first2, *first1)) ? *first2++ : *first1++;
    ++processed_el_cnt;
  }
}

// Select the largest elements.
template <class It, class Comp>
void MergeLastElements(It first1, It last1, It first2, It last2, It result,
                       size_t el_cnt, Comp comp) {
  assert((last1 - first1) + (last2 - first2) >= static_cast<std::ptrdiff_t>(el_cnt));

  auto it = result + el_cnt;
  size_t processed_el_cnt = 0;
  while (processed_el_cnt < el_cnt) {
    if (first1 == last1) {
      std::copy(last2 - (el_cnt - processed_el_cnt), last2,
                it - (el_cnt - processed_el_cnt));
      return;
    }
    if (first2 == last2) {
      std::copy(last1 - (el_cnt - processed_el_cnt), last1,
                it - (el_cnt - processed_el_cnt));
      return;
    }
    *(--it) = comp(*(last2 - 1), *(last1 - 1)) ? *(--last1) : *(--last2);
    ++processed_el_cnt;
  }
}

/* Bitonic sort with merging to determine own data partition */
template <class T, class Comp>
void SortBitonicSequenceMerge(std::vector<T>& data,
                              std::vector<T>& tmp_data,
                              MPI_Datatype mpi_type,
                              int& my_size,
                              size_t max_input_size,
                              Comp comp,
                              bool sort_increasing,
                              int tag,
                              const RBC::Comm& comm) {
  assert(std::is_sorted(data.begin(), data.begin() + my_size, comp));
  // We may have to store our own data plus our partner's data in worst case.
  assert(data.size() == 2 * max_input_size);
  assert(tmp_data.size() == 2 * max_input_size);

  int nprocs, myrank;
  RBC::Comm_size(comm, &nprocs);
  RBC::Comm_rank(comm, &myrank);

  const int logp = tlx::integer_log2_ceil(nprocs);

  for (int bit_idx = logp - 1; bit_idx >= 0; --bit_idx) {
    const bool right_group = (myrank >> bit_idx) & 0x1;
    const bool large_el_group = right_group == sort_increasing;
    const int partner = myrank ^ (1 << bit_idx);

    if (partner < nprocs) {
      // Receive elements from partner.
      int send_size = my_size;
      int recv_size = 0;
      RBC::Sendrecv(&send_size, 1, Common::getMpiType(send_size), partner, tag,
                    &recv_size, 1, Common::getMpiType(recv_size), partner,
                    tag, comm, MPI_STATUS_IGNORE);
      RBC::Sendrecv(data.data(), send_size, mpi_type, partner,
                    tag, data.data() + send_size,
                    recv_size, mpi_type, partner, tag,
                    comm, MPI_STATUS_IGNORE);

      const int64_t total_size = send_size + recv_size;

      const int num_small_els = std::min<int64_t>(total_size, max_input_size);
      const int num_large_els = total_size - num_small_els;

      // The PE in large group selects the smallest
      // 'num_small_els' elements and the PE in the small group
      // selects the remaining elements.
      assert(num_small_els + num_large_els == total_size);
      if (large_el_group) {
        // Partner data first, own data second to avoid that
        // this PE and the partner PE selects the same element
        // 'e'. Two PEs cound select the same element if both
        // pass their own data first and other elements exist
        // that have the same key as 'e'.
        MergeLastElements(data.begin() + send_size, data.begin() + total_size,
                          data.begin(), data.begin() + send_size,
                          tmp_data.begin(), num_large_els, comp);
        data.swap(tmp_data);
        my_size = num_large_els;
      } else {
        // Own data first, partner data second -> stable
        MergeFirstElements(data.begin(), data.begin() + send_size,
                           data.begin() + send_size, data.begin() + total_size,
                           tmp_data.begin(), num_small_els, comp);
        data.swap(tmp_data);
        my_size = num_small_els;
      }
    }
  }
}

/* Bitonic sort which uses std::lower_bound to determine own data partition */
template <class T, class Comp>
void SortBitonicSequenceBinSearch(std::vector<T>& data,
                                  std::vector<T>& tmp_data,
                                  MPI_Datatype mpi_type,
                                  int& my_size,
                                  size_t max_input_size,
                                  Comp comp,
                                  bool sort_increasing,
                                  int tag,
                                  RBC::Comm comm) {
  assert(std::is_sorted(data.begin(), data.begin() + my_size, comp));
  // We may have to store our own data plus our partner's data in worst case.
  assert(data.size() == 2 * max_input_size);
  assert(tmp_data.size() == 2 * max_input_size);

  int nprocs, myrank;
  RBC::Comm_size(comm, &nprocs);
  RBC::Comm_rank(comm, &myrank);

  const int logp = tlx::integer_log2_ceil(nprocs);

  for (int bit_idx = logp - 1; bit_idx >= 0; --bit_idx) {
    const bool right_group = (myrank >> bit_idx) & 0x1;
    const bool large_el_group = right_group == sort_increasing;
    const int partner = myrank ^ (1 << bit_idx);

    if (partner < nprocs) {
      // Receive elements from partner.
      int partner_size = 0;
      RBC::Sendrecv(const_cast<int*>(&my_size), 1, Common::getMpiType(my_size), partner, tag,
                    &partner_size, 1, Common::getMpiType(partner_size), partner,
                    tag, comm, MPI_STATUS_IGNORE);

      // Number of elements of the PE in the small group
      const int num_small_els = std::min<int64_t>(
        static_cast<int64_t>(my_size + partner_size),
        max_input_size);
      const int num_large_els = my_size + partner_size - num_small_els;

      int partner_range_size = partner_size;
      int my_range_size = my_size;
      int partner_l = 0;
      int my_l = 0;
      int my_r = my_size;
#ifndef NDEBUG
      int partner_r = partner_size;
#endif
      while (partner_range_size > 0 || my_range_size > 0) {
        assert(my_r - my_l == my_range_size);
        assert(partner_r - partner_l == partner_range_size);

        /* Determine pivot and pivot offset in both ranges. */
        T pivot;
        // Position of pivot relative to 'l'.
        int my_pivot_offset = 0;
        // Distribute pivot. The PE, which stores more
        // elements, will select the pivot.
        bool partner_selects_pivot = true;
        if (my_range_size > partner_range_size ||
            (partner_range_size == my_range_size &&
             large_el_group)) {
          assert(my_range_size > 0);
          my_pivot_offset = (my_r - my_l) / 2;
          pivot = data[my_l + my_pivot_offset];
          RBC::Send(&pivot, 1, mpi_type,
                    partner, tag, comm);
          partner_selects_pivot = false;
        } else {
          RBC::Recv(&pivot, 1, mpi_type,
                    partner, tag, comm, MPI_STATUS_IGNORE);
          const auto ptr = std::lower_bound(data.begin() + my_l, data.begin() + my_r, pivot, comp);
          my_pivot_offset = ptr - (data.begin() + my_l);
        }

        /* Calculate new range. */
        int partner_pivot_offset = 0;
        RBC::Sendrecv(&my_pivot_offset, 1,
                      Common::getMpiType(my_pivot_offset),
                      partner, tag,
                      &partner_pivot_offset, 1,
                      Common::getMpiType(partner_pivot_offset),
                      partner, tag,
                      comm, MPI_STATUSES_IGNORE);

        const int64_t pivot_range_size = my_l + partner_l
                                         + my_pivot_offset + partner_pivot_offset;
        // Total number of elements which are smaller than the pivot is too large.
        if (num_small_els == pivot_range_size) {
          partner_range_size = 0;
          my_range_size = 0;
          partner_l = partner_l + partner_pivot_offset;
#ifndef NDEBUG
          partner_r = partner_l;
#endif
          my_l = my_l + my_pivot_offset;
          my_r = my_l;
        } else if (pivot_range_size < num_small_els) {
          // + 1 to discard pivot in subsequent rounds
          if (partner_selects_pivot) {
            partner_range_size -= partner_pivot_offset + 1;
            my_range_size -= my_pivot_offset;
            partner_l += partner_pivot_offset + 1;
            // partner_r = partner_r;
            my_l += my_pivot_offset;
          } else {
            partner_range_size -= partner_pivot_offset;
            my_range_size -= my_pivot_offset + 1;
            partner_l += partner_pivot_offset;
            // partner_r = partner_r;
            my_l += my_pivot_offset + 1;
            // my_r = my_r;
          }
        } else {
          partner_range_size = partner_pivot_offset;
          my_range_size = my_pivot_offset;
#ifndef NDEBUG
          partner_r = partner_l + partner_pivot_offset;
#endif
          // my_l = my_l;
          my_r = my_l + my_pivot_offset;
        }
      }
      assert(my_l == my_r);
      assert(partner_l == partner_r);

      // Merge local elements and elements from partner and select the
      // first/second half.
      // Swap left and right group if sort_increasing == false
      if (large_el_group) {
        assert(partner_size - partner_l + my_size - my_l == num_large_els);
        const int send_cnt = my_l;
        const int recv_cnt = partner_size - partner_l;
        RBC::Sendrecv(data.data(),
                      send_cnt,
                      mpi_type,
                      partner,
                      tag,
                      data.data() + my_size,
                      recv_cnt,
                      mpi_type,
                      partner,
                      tag,
                      comm,
                      MPI_STATUSES_IGNORE);

        std::merge(data.begin() + send_cnt, data.begin() + my_size,
                   data.begin() + my_size, data.begin() + my_size + recv_cnt,
                   tmp_data.begin(), comp);
        my_size = num_large_els;
        data.swap(tmp_data);
      } else {
        assert(partner_l + my_l == num_small_els);
        const int send_size = my_size - my_l;
        const int recv_size = partner_l;
        RBC::Sendrecv(data.data() + my_l,
                      send_size,
                      mpi_type,
                      partner,
                      tag,
                      data.data() + my_size,
                      recv_size,
                      mpi_type,
                      partner,
                      tag,
                      comm,
                      MPI_STATUSES_IGNORE);

        std::merge(data.begin(), data.begin() + my_l,
                   data.begin() + my_size, data.begin() + my_size + recv_size,
                   tmp_data.begin(), comp);
        my_size = num_small_els;
        data.swap(tmp_data);
      }
    }
  }
}

template <class T, class Comp>
void RecSort(std::vector<T>& data,
             std::vector<T>& tmp_data,
             MPI_Datatype mpi_type,
             int& my_size,
             size_t max_input_size,
             Comp comp,
             bool asc,
             Bitonic::PartitionBy partition_strategy,
             int tag,
             const RBC::Comm& comm) {
  // We may have to store our own data plus our partner's data in worst case.
  assert(data.size() == 2 * max_input_size);
  assert(tmp_data.size() == 2 * max_input_size);

  int nprocs, myrank;
  RBC::Comm_size(comm, &nprocs);
  RBC::Comm_rank(comm, &myrank);

  if (nprocs == 1) {
    ips4o::sort(data.begin(), data.begin() + my_size, comp);
    return;
  }

  int left_size = nprocs / 2;
  if (myrank < left_size) {
    RBC::Comm sub_comm;
    RBC::Comm_create_group(comm, &sub_comm, 0, left_size - 1);
    RecSort(data, tmp_data, mpi_type, my_size, max_input_size,
            comp, !asc, partition_strategy, tag, sub_comm);
  } else {
    RBC::Comm sub_comm;
    RBC::Comm_create_group(comm, &sub_comm, left_size, nprocs - 1);
    RecSort(data, tmp_data, mpi_type, my_size, max_input_size,
            comp, asc, partition_strategy, tag, sub_comm);
  }

  if (partition_strategy == Bitonic::PartitionBy::Merging) {
    SortBitonicSequenceMerge(data, tmp_data, mpi_type,
                             my_size, max_input_size, comp, asc, tag, comm);
  } else {
    SortBitonicSequenceBinSearch(data, tmp_data, mpi_type,
                                 my_size, max_input_size, comp, asc, tag, comm);
  }
}
}      // end namespace _internal

template <class T,
          class Comp>
void Sort(std::vector<T>& data, MPI_Datatype mpi_type, int tag, const RBC::Comm& comm,
          Comp comp, bool is_equal_input_size,
          Bitonic::PartitionBy partition_strategy) {
  int nprocs, myrank;
  RBC::Comm_size(comm, &nprocs);
  RBC::Comm_rank(comm, &myrank);

  if (nprocs == 1) {
    ips4o::sort(data.begin(), data.end(), comp);
    return;
  }

#ifndef NDEBUG
  if (is_equal_input_size) {
    size_t num_els = data.size();
    size_t max = 0;
    RBC::Allreduce(&num_els, &max, 1, Common::getMpiType(num_els), MPI_MAX, comm);
    assert(num_els == max);
  }
#endif

  int max_input_size = data.size();
  if (!is_equal_input_size) {
    int num_els = data.size();
    RBC::Allreduce(&num_els, &max_input_size, 1, Common::getMpiType(num_els), MPI_MAX, comm);
  }

  int my_size = data.size();
  std::vector<T> tmp_data(2 * max_input_size);
  data.resize(2 * max_input_size);

  const bool asc = true;
  _internal::RecSort(data, tmp_data, mpi_type, my_size, max_input_size,
                     comp, asc, partition_strategy, tag, comm);

  data.resize(my_size);
}

template <class T,
          class Comp>
void SortBitonicSequence(std::vector<T>& data,
                         MPI_Datatype mpi_type,
                         int tag,
                         const RBC::Comm& comm,
                         Comp comp,
                         bool asc,
                         bool is_equal_input_size,
                         Bitonic::PartitionBy partition_strategy) {
  int nprocs, myrank;
  RBC::Comm_size(comm, &nprocs);
  RBC::Comm_rank(comm, &myrank);

  if (nprocs == 1) {
    return;
  }

#ifndef NDEBUG
  if (is_equal_input_size) {
    size_t num_els = data.size();
    size_t max = 0;
    RBC::Allreduce(&num_els, &max, 1, Common::getMpiType(num_els), MPI_MAX, comm);
    assert(num_els == max);
  }
#endif

  int max_input_size = data.size();
  if (!is_equal_input_size) {
    int num_els = data.size();
    RBC::Allreduce(&num_els, &max_input_size, 1, Common::getMpiType(num_els), MPI_MAX, comm);
  }

  int my_size = data.size();
  std::vector<T> tmp_data(2 * max_input_size);
  data.resize(2 * max_input_size);

  if (partition_strategy == Bitonic::PartitionBy::Merging) {
    _internal::SortBitonicSequenceMerge(data, tmp_data, mpi_type,
                                        my_size, max_input_size,
                                        comp, asc, tag, comm);
  } else {
    _internal::SortBitonicSequenceBinSearch(data, tmp_data, mpi_type,
                                            my_size, max_input_size,
                                            comp, asc, tag, comm);
  }

  data.resize(my_size);
}
template <class T, class Comp>
void Sort(std::vector<T>& data, MPI_Datatype mpi_type, int tag, MPI_Comm comm,
          Comp comp, bool is_equal_input_size,
          Bitonic::PartitionBy partition_strategy) {
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  Sort(data,
       mpi_type, tag,
       rcomm,
       comp,
       is_equal_input_size,
       partition_strategy);
}

template <class T, class Comp>
void SortBitonicSequence(std::vector<T>& data,
                         MPI_Datatype mpi_type,
                         int tag,
                         MPI_Comm comm,
                         Comp comp,
                         bool sort_increasing,
                         bool is_equal_input_size,
                         Bitonic::PartitionBy partition_strategy) {
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  SortBitonicSequence(data,
                      mpi_type,
                      tag,
                      rcomm,
                      comp,
                      sort_increasing,
                      is_equal_input_size,
                      partition_strategy);
}
}  // end namespace Bitonic
