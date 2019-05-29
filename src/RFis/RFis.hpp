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
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <ips4o.hpp>
#include <RBC.hpp>
#include <tlx/math.hpp>

#include "../../include/Tools/MpiTuple.hpp"
#include "../Tools/Common.hpp"
#include "../Tools/CommonMpi.hpp"
#include "../../include/RFis/Tags.hpp"

namespace RFis {

namespace _internal {
template <class ForwardIterator, class T, class Compare>
ForwardIterator linear_upper_bound(ForwardIterator first, ForwardIterator last,
                                   const T& val, Compare comp) {
  while (first != last && !comp(val, *first)) {
    ++first;
  }
  return first;
}

template <class ForwardIterator, class T, class Compare>
ForwardIterator linear_lower_bound(ForwardIterator first, ForwardIterator last,
                                   const T& val, Compare comp) {
  while (first != last && comp(*first, val)) {
    ++first;
  }
  return first;
}

template <class RFisTags, class T>
void SendRecv(size_t partner, T* out_begin, size_t send_size, std::vector<T>& in,
              MPI_Datatype mpi_datatype, const RBC::Comm& comm) {
  MPI_Request requests[2];

  RBC::Isend(out_begin, send_size, mpi_datatype, partner,
             RFisTags::kRedistribution, comm, requests);

  MPI_Status recv_status;
  RBC::Probe(partner, RFisTags::kRedistribution, comm, &recv_status);

  int recv_size;
  MPI_Get_count(&recv_status, mpi_datatype, &recv_size);

  in.clear();
  in.resize(recv_size);

  RBC::Irecv(in.data(), in.size(), mpi_datatype, partner, RFisTags::kRedistribution,
             comm, requests + 1);

  MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
}

/* @brief Hypercube routing algorithm which routes elements by their rank.
 *
 * Routes elements to processes by their rank.
 *
 * @param comm Communicator
 * @param rank Rank of the process.
 * @pram size Number of processes of comm calling this routine.
 *            'size' must be a power of two.
 *            The first 'size' processes of comm must call this routine.
 * @param ranked_els Elements which ranks. There are no rank
 *        duplicates and the ranks are in range [0, num_total_els).
 *
 * @return Each process returns at most ceil(num_total_els / comm.getSize()) elements.
 */
template <class RFisTags, class T, class RankType>
std::vector<T> HypercubeRedistributeByElementRank(const int rank,
                                                  const int size,
                                                  std::vector<Tools::Tuple<T,
                                                                           RankType> >& ranked_els,
                                                  MPI_Datatype mpi_ranked_type,
                                                  size_t num_total_els,
                                                  const RBC::Comm& comm) {
  assert(tlx::is_power_of_two(size));

  const size_t els_per_pe = tlx::div_ceil(num_total_els, size);
  int rank_splitter = els_per_pe * size / 2;

  const size_t comm_phases = tlx::integer_log2_ceil(size);

  std::vector<Tools::Tuple<T, RankType> > recv_els, merge_els;
  recv_els.reserve(2 * els_per_pe);
  merge_els.reserve(2 * els_per_pe);

  for (int phase = comm_phases; phase > 0; --phase) {
    const int num_cube_pes = 1 << (phase - 1);
    const int partner = rank ^ num_cube_pes;
    const bool is_bit_set = rank & num_cube_pes;

    auto split_ptr = std::lower_bound(ranked_els.begin(), ranked_els.end(),
                                      rank_splitter,
                                      [](const Tools::Tuple<T, RankType>& el,
                                         const int rank_splitter) {
          return el.second < rank_splitter;
        });

    size_t send_pos, send_size, remain_pos, remain_size = 0;
    if (is_bit_set) {
      send_pos = 0;
      send_size = split_ptr - ranked_els.begin();
      remain_pos = send_size;
      remain_size = ranked_els.size() - send_size;
    } else {
      remain_pos = 0;
      remain_size = split_ptr - ranked_els.begin();
      send_pos = remain_size;
      send_size = ranked_els.size() - remain_size;
    }

    SendRecv<RFisTags>(partner, ranked_els.data() + send_pos, send_size,
             recv_els, mpi_ranked_type, comm);

    merge_els.resize(recv_els.size() + remain_size);
    std::merge(ranked_els.begin() + remain_pos,
               ranked_els.begin() + remain_pos + remain_size, recv_els.begin(),
               recv_els.end(), merge_els.begin(),
               [](const Tools::Tuple<T, RankType>& e1,
                  const Tools::Tuple<T, RankType>& e2) {
          return e1.second < e2.second;
        });
    ranked_els.swap(merge_els);

    if (is_bit_set) {
      rank_splitter += num_cube_pes * els_per_pe / 2;
    } else {
      rank_splitter -= num_cube_pes * els_per_pe / 2;
    }
    recv_els.clear();
    merge_els.clear();
  }

  std::vector<T> values(ranked_els.size());
  for (size_t i = 0; i != ranked_els.size(); ++i) {
    values[i] = ranked_els[i].first;
  }
  return values;
}

// Moves elements to the first 'num_target_pes' PEs. The total
// number of PEs must be smaller than 2 * 'num_target_pe.'
template <class RFisTags, class T>
std::vector<T> RedistributeByPeId(MPI_Datatype mpi_datatype,
                                  int num_target_pes,
                                  std::vector<T> els,
                                  const RBC::Comm& comm) {
  const int size = comm.getSize();
  const int rank = comm.getRank();

  assert(num_target_pes * num_target_pes >= size);

  if (rank >= num_target_pes) {
    // Send elements to first PEs of the subcube.

    RBC::Send(els.data(), els.size(), mpi_datatype,
              rank - num_target_pes, RFisTags::kRedistribution, comm);
    els.clear();
  } else if (rank < size - num_target_pes) {
    // Receive elements from the last PEs of the subcube.

    int count = 0;
    MPI_Status status;

    RBC::Probe(rank + num_target_pes, RFisTags::kRedistribution, comm, &status);
    MPI_Get_count(&status, mpi_datatype, &count);

    const int prev_size = els.size();
    els.resize(els.size() + count);
    RBC::Recv(els.data() + prev_size,
              count,
              mpi_datatype,
              rank + num_target_pes,
              RFisTags::kRedistribution,
              comm,
              MPI_STATUS_IGNORE);
  }

  return els;
}

/*
 * @brief Calculates ranks of input elements.
 *
 * This is a helper function! The function is not robust meaning that
 * the elements must be stored at process [0..comm.getSize() ) if the
 * global input is smaller than comm.getSize() to guarantee a fast
 * execution.
 *
 * This function does only work with more than three processes as we
 * need a grid of at least 2x2 processes. The code is not able to
 * handle a grid of one process (1x1 for 1, 2, and 3 input processes).
 */
template <class RFisTags, class T, class RankType, class Comp>
std::vector<Tools::Tuple<T, RankType> >
Rank(MPI_Datatype mpi_datatype,
     MPI_Datatype rank_type,
     std::vector<T> els, Comp comp, const RBC::Comm& comm) {
  const int size = comm.getSize();
  const int rank = comm.getRank();

  assert(size >= 4);

  // Sort input.
  ips4o::sort(els.begin(), els.end(), comp);

  const int num_cols = Common::int_floor_sqrt(size);
  const int row_id = rank / num_cols;
  const int col_id = rank % num_cols;

  // Create row communicators and column communicators. The row
  // communicators contain num_cols processes. Only the last row
  // communicator may contain less processes. There are num_cols,
  // num_cols + 1, or num_cols + 2 row communicators. The column
  // communicator extended_col_comm of column i contains the contains
  // the i'th process of each row communicator. The column
  // communicator col_comm of column i contains the i'th process of
  // the first num_cols row communicators.
  RBC::Comm row_comm, extended_col_comm, col_comm;
  RBC::Comm_create_group(comm, &row_comm, row_id * num_cols, std::min((row_id + 1) * num_cols - 1,
                                                                      size - 1));
  RBC::Comm_create_group(comm, &extended_col_comm, col_id, size - 1, num_cols);
  RBC::Comm_create_group(comm, &col_comm, col_id, num_cols * num_cols - 1, num_cols);

  // Calculate number of elements in my column.
  RankType num_loc_els = els.size();
  RankType num_col_els = 0;
  RBC::Allreduce(&num_loc_els, &num_col_els, 1, rank_type, MPI_SUM, extended_col_comm);

  /*
   * Allgather-merge column elements.
   */
  std::unique_ptr<T[]> col_els(new T[num_col_els]);
  auto merger = [&comp](void* a_begin, void* a_end, void* b_begin, void* b_end, void* out) {
                  std::merge(
                    static_cast<T*>(a_begin),
                    static_cast<T*>(a_end),
                    static_cast<T*>(b_begin),
                    static_cast<T*>(b_end),
                    static_cast<T*>(out),
                    comp);
                };
  RBC::Gatherm(els.data(), els.size(), mpi_datatype, col_els.get(), num_col_els,
               0, merger, extended_col_comm);

  if (rank < num_cols * num_cols) {
    RBC::_internal::optimized::BcastTwotree(col_els.get(), num_col_els, mpi_datatype, 0, col_comm);

    /*
     * Transpose matrix.
     */
    RankType num_row_els = 0;
    const int source = col_id * num_cols + row_id;
    const int target = num_cols * col_id + row_id;
    RBC::Sendrecv(&num_col_els,
                  1,
                  rank_type,
                  target,
                  RFisTags::kGeneral,
                  &num_row_els,
                  1,
                  rank_type,
                  source,
                  RFisTags::kGeneral,
                  comm,
                  MPI_STATUSES_IGNORE);

    std::unique_ptr<T[]> row_els(new T[num_row_els]);

    RBC::Sendrecv(col_els.get(),
                  num_col_els,
                  mpi_datatype,
                  target,
                  RFisTags::kGeneral,
                  row_els.get(),
                  num_row_els,
                  mpi_datatype,
                  source,
                  RFisTags::kGeneral,
                  comm,
                  MPI_STATUSES_IGNORE);

    /*
     * Calculate rank of column elements in row elements.
     */
    std::unique_ptr<RankType[]> ranks(new RankType[num_col_els]);
    std::unique_ptr<RankType[]> reduced_ranks(new RankType[num_col_els]);

    if (col_id == row_id) {
      // Rank of the i'th element is i.

      auto rank_begin = ranks.get();
      const auto rank_end = rank_begin + num_col_els;
      for (auto rank = rank_begin; rank != rank_end; ++rank) {
        *rank = rank - rank_begin;
      }
    } else if (col_id > row_id) {
      // Use < operation

      const auto col_begin = col_els.get();
      const auto row_begin = row_els.get();
      auto row_ptr = row_begin;
      const auto row_end = row_ptr + num_row_els;
      const auto rank_begin = ranks.get();
      for (RankType i = 0; i != num_col_els; ++i) {
        auto it = _internal::linear_upper_bound(
          row_ptr, row_end, col_begin[i], comp);
        rank_begin[i] = it - row_begin;
        row_ptr = it;
      }
    } else {
      // Use <= operation

      assert(col_id < row_id);
      const auto col_begin = col_els.get();
      const auto row_begin = row_els.get();
      auto row_ptr = row_begin;
      const auto row_end = row_ptr + num_row_els;
      const auto rank_begin = ranks.get();

      for (RankType i = 0; i != num_col_els; ++i) {
        auto it = _internal::linear_lower_bound(
          row_ptr, row_end, col_begin[i], comp);
        rank_begin[i] = it - row_begin;
        row_ptr = it;
      }
    }

    /*
     * Calculate global ranks.
     */
    RBC::_internal::optimized::AllreduceTwotree(ranks.get(), reduced_ranks.get(), num_col_els,
                                                rank_type, MPI_SUM, col_comm);

    RankType num_max_return_els = tlx::div_ceil(num_col_els, col_comm.getSize());
    const auto begin_idx = std::min(
      num_col_els,
      row_id * num_max_return_els);
    const auto end_idx = std::min(
      num_col_els,
      (row_id + 1) * num_max_return_els);
    const std::vector<T> ret_els(
      col_els.get() + begin_idx,
      col_els.get() + end_idx);
    const std::vector<RankType> ret_ranks(
      reduced_ranks.get() + begin_idx,
      reduced_ranks.get() + end_idx);

    const RankType num_ret_els = end_idx - begin_idx;
    auto col_els_begin = col_els.get() + begin_idx;
    auto reduced_ranks_begin = reduced_ranks.get() + begin_idx;
    std::vector<Tools::Tuple<T, RankType> > ranked_els(num_ret_els);
    for (RankType i = 0; i != num_ret_els; ++i) {
      ranked_els[i].first = col_els_begin[i];
      ranked_els[i].second = reduced_ranks_begin[i];
    }

    return ranked_els;
  } else {
    return std::vector<Tools::Tuple<T, RankType> >();
  }
}

/*
 * @brief Calculates ranks of input elements.
 *
 * This is a helper function! The function is not robust meaning that
 * the elements must be stored at process [0..comm.getSize() ) if the
 * global input is smaller than comm.getSize() to guarantee a fast
 * execution.
 *
 * This function does only work with more than three processes as we
 * need a grid of at least 2x2 processes. The code is not able to
 * handle a grid of one process (1x1 for 1, 2, and 3 input processes).
 */
template <class RFisTags, class T, class RankType, class Comp>
std::vector<Tools::Tuple<T, RankType> >
Rank(MPI_Datatype mpi_datatype,
     MPI_Datatype rank_type,
     std::vector<T> els,
     Comp comp,
     RankType* ret_num_total_elements,
     const RBC::Comm& comm) {
  const int size = comm.getSize();
  const int rank = comm.getRank();

  assert(size >= 4);

  // Sort input.
  ips4o::sort(els.begin(), els.end(), comp);

  const int num_cols = Common::int_floor_sqrt(size);
  const int row_id = rank / num_cols;
  const int col_id = rank % num_cols;

  // Create row communicators and column communicators. The row
  // communicators contain num_cols processes. Only the last row
  // communicator may contain less processes. There are num_cols,
  // num_cols + 1, or num_cols + 2 row communicators. The column
  // communicator extended_col_comm of column i contains
  // the i'th process of each row communicator. The column
  // communicator col_comm of column i contains the i'th process of
  // the first num_cols row communicators.
  RBC::Comm row_comm, extended_col_comm, col_comm;
  RBC::Comm_create_group(comm, &row_comm, row_id * num_cols, std::min((row_id + 1) * num_cols - 1,
                                                                      size - 1));
  RBC::Comm_create_group(comm, &extended_col_comm, col_id, size - 1, num_cols);
  RBC::Comm_create_group(comm, &col_comm, col_id, num_cols * num_cols - 1, num_cols);

  // Calculate number of elements in my column.
  RankType num_loc_els = els.size();
  RankType num_col_els = 0;
  RBC::Allreduce(&num_loc_els, &num_col_els, 1, rank_type, MPI_SUM, extended_col_comm);

  // Calculate total number of elements.
  const int num_remaining_pes = comm.getSize() - num_cols * num_cols;
  if (rank < num_cols * num_cols) {
    // Calculate total number of elements row-wise.
    RBC::Allreduce(&num_col_els, ret_num_total_elements, 1, rank_type, MPI_SUM, row_comm);

    if (rank >= num_cols * num_cols - num_remaining_pes) {
      // Send total number of element to the remaining processes.
      RBC::Send(ret_num_total_elements, 1, rank_type, rank + num_remaining_pes,
                RFisTags::kGeneral, comm);
    }
  } else {
    // Receive total number of elements.
    RBC::Recv(ret_num_total_elements, 1, rank_type, rank - num_remaining_pes,
              RFisTags::kGeneral, comm, MPI_STATUS_IGNORE);
  }


  /*
   * Allgather-merge column elements.
   */
  std::unique_ptr<T[]> col_els(new T[num_col_els]);
  auto merger = [&comp](void* a_begin, void* a_end, void* b_begin, void* b_end, void* out) {
                  std::merge(
                    static_cast<T*>(a_begin),
                    static_cast<T*>(a_end),
                    static_cast<T*>(b_begin),
                    static_cast<T*>(b_end),
                    static_cast<T*>(out),
                    comp);
                };
  RBC::Gatherm(els.data(), els.size(), mpi_datatype, col_els.get(), num_col_els,
               0, merger, extended_col_comm);

  if (rank < num_cols * num_cols) {
    RBC::_internal::optimized::BcastTwotree(col_els.get(), num_col_els, mpi_datatype, 0, col_comm);

    /*
     * Transpose matrix.
     */
    RankType num_row_els = 0;
    const int source = col_id * num_cols + row_id;
    const int target = num_cols * col_id + row_id;
    RBC::Sendrecv(&num_col_els,
                  1,
                  rank_type,
                  target,
                  RFisTags::kGeneral,
                  &num_row_els,
                  1,
                  rank_type,
                  source,
                  RFisTags::kGeneral,
                  comm,
                  MPI_STATUSES_IGNORE);

    std::unique_ptr<T[]> row_els(new T[num_row_els]);

    RBC::Sendrecv(col_els.get(),
                  num_col_els,
                  mpi_datatype,
                  target,
                  RFisTags::kGeneral,
                  row_els.get(),
                  num_row_els,
                  mpi_datatype,
                  source,
                  RFisTags::kGeneral,
                  comm,
                  MPI_STATUSES_IGNORE);

    /*
     * Calculate rank of column elements in row elements.
     */
    std::unique_ptr<RankType[]> ranks(new RankType[num_col_els]);
    std::unique_ptr<RankType[]> reduced_ranks(new RankType[num_col_els]);

    if (col_id == row_id) {
      // Rank of the i'th element is i.

      auto rank_begin = ranks.get();
      const auto rank_end = rank_begin + num_col_els;
      for (auto rank = rank_begin; rank != rank_end; ++rank) {
        *rank = rank - rank_begin;
      }
    } else if (col_id > row_id) {
      // Use < operation

      const auto col_begin = col_els.get();
      const auto row_begin = row_els.get();
      auto row_ptr = row_begin;
      const auto row_end = row_ptr + num_row_els;
      const auto rank_begin = ranks.get();
      for (RankType i = 0; i != num_col_els; ++i) {
        auto it = _internal::linear_upper_bound(
          row_ptr, row_end, col_begin[i], comp);
        rank_begin[i] = it - row_begin;
        row_ptr = it;
      }
    } else {
      // Use <= operation

      assert(col_id < row_id);
      const auto col_begin = col_els.get();
      const auto row_begin = row_els.get();
      auto row_ptr = row_begin;
      const auto row_end = row_ptr + num_row_els;
      const auto rank_begin = ranks.get();

      for (RankType i = 0; i != num_col_els; ++i) {
        auto it = _internal::linear_lower_bound(
          row_ptr, row_end, col_begin[i], comp);
        rank_begin[i] = it - row_begin;
        row_ptr = it;
      }
    }

    /*
     * Calculate global ranks.
     */
    RBC::_internal::optimized::AllreduceTwotree(ranks.get(), reduced_ranks.get(), num_col_els,
                                                rank_type, MPI_SUM, col_comm);

    RankType num_max_return_els = tlx::div_ceil(num_col_els, col_comm.getSize());
    const auto begin_idx = std::min(
      num_col_els,
      row_id * num_max_return_els);
    const auto end_idx = std::min(
      num_col_els,
      (row_id + 1) * num_max_return_els);
    const std::vector<T> ret_els(
      col_els.get() + begin_idx,
      col_els.get() + end_idx);
    const std::vector<RankType> ret_ranks(
      reduced_ranks.get() + begin_idx,
      reduced_ranks.get() + end_idx);

    const RankType num_ret_els = end_idx - begin_idx;
    auto col_els_begin = col_els.get() + begin_idx;
    auto reduced_ranks_begin = reduced_ranks.get() + begin_idx;
    std::vector<Tools::Tuple<T, RankType> > ranked_els(num_ret_els);
    for (RankType i = 0; i != num_ret_els; ++i) {
      ranked_els[i].first = col_els_begin[i];
      ranked_els[i].second = reduced_ranks_begin[i];
    }

    return ranked_els;
  } else {
    return std::vector<Tools::Tuple<T, RankType> >();
  }
}
}  // namespace _internal

/* @brief Ranks elements.
 *
 * @return All processes of a row return the same elements. The
 *         processes of each column return the global input.
 *         Returns the column communicator.
 */
template <class RFisTags, class T, class RankType, class Comp>
std::tuple<std::vector<T>, std::vector<RankType>, RBC::Comm>
RankRowwise(MPI_Datatype mpi_datatype,
                  MPI_Datatype rank_type,
                  std::vector<T> els,
                  Comp comp,
                  const RBC::Comm& comm) {
  const int size = comm.getSize();
  const int rank = comm.getRank();

  ips4o::sort(els.begin(), els.end(), comp);

  // For the ranking algorithm, we need a grid of at least 2 * 2 processes.
  if (size <= 3) {
    RankType num_loc_els = els.size();
    RankType num_total_els = 0;
    RBC::Allreduce(&num_loc_els, &num_total_els, 1, rank_type, MPI_SUM, comm);
    std::vector<T> gathered_els(num_total_els);
    RBC::Gatherm(els.data(), els.size(), mpi_datatype,
                 gathered_els.data(), num_total_els, 0,
                 [&comp](
                   void* a_begin, void* a_end,
                   void* b_begin, void* b_end,
                   void* out
                   ) {
        std::merge(
          static_cast<T*>(a_begin),
          static_cast<T*>(a_end),
          static_cast<T*>(b_begin),
          static_cast<T*>(b_end),
          static_cast<T*>(out),
          comp);
      },
                 comm);

    if (rank > 0) return { std::vector<T>{ }, std::vector<RankType>{ }, comm };

    std::tuple<std::vector<T>, std::vector<RankType>, RBC::Comm> ret{
      std::move(gathered_els), std::vector<RankType>(num_total_els), comm
    };
    std::iota(std::get<1>(ret).begin(), std::get<1>(ret).end(), RankType{ 0 });

    return ret;
  }

  const int num_cols = Common::int_floor_sqrt(size);
  const int row_id = rank / num_cols;
  const int col_id = rank % num_cols;

  RBC::Comm row_comm, col_comm;
  RBC::Comm_create_group(comm, &row_comm, row_id * num_cols, std::min((row_id + 1) * num_cols - 1,
                                                                      size - 1));
  RBC::Comm_create_group(comm, &col_comm, col_id, size - 1, num_cols);

  RankType num_loc_els = els.size();
  RankType num_col_els = 0;
  RBC::Allreduce(&num_loc_els, &num_col_els, 1, rank_type, MPI_SUM, col_comm);

  std::unique_ptr<T[]> col_els(new T[num_col_els]);
  auto merger = [&comp](void* a_begin, void* a_end, void* b_begin, void* b_end, void* out) {
                  std::merge(
                    static_cast<T*>(a_begin),
                    static_cast<T*>(a_end),
                    static_cast<T*>(b_begin),
                    static_cast<T*>(b_end),
                    static_cast<T*>(out),
                    comp);
                };
  RBC::Gatherm(els.data(), els.size(), mpi_datatype, col_els.get(), num_col_els,
               0, merger, col_comm);

  RBC::_internal::optimized::BcastTwotree(col_els.get(), num_col_els, mpi_datatype, 0, col_comm);

  if (rank < num_cols * num_cols) {
    RankType num_row_els = 0;
    const int source = col_id * num_cols + row_id;
    const int target = num_cols * col_id + row_id;
    RBC::Sendrecv(&num_col_els,
                  1,
                  rank_type,
                  target,
                  RFisTags::kGeneral,
                  &num_row_els,
                  1,
                  rank_type,
                  source,
                  RFisTags::kGeneral,
                  comm,
                  MPI_STATUSES_IGNORE);

    std::unique_ptr<T[]> row_els(new T[num_row_els]);

    RBC::Sendrecv(col_els.get(),
                  num_col_els,
                  mpi_datatype,
                  target,
                  RFisTags::kGeneral,
                  row_els.get(),
                  num_row_els,
                  mpi_datatype,
                  source,
                  RFisTags::kGeneral,
                  comm,
                  MPI_STATUSES_IGNORE);

    // Calculate rank of row elements in column elements.
    std::unique_ptr<RankType[]> ranks(new RankType[num_row_els]);
    std::unique_ptr<RankType[]> reduced_ranks(new RankType[num_row_els]);

    if (row_id == col_id) {
      // Rank of the i'th element is i.

      auto rank_begin = ranks.get();
      const auto rank_end = rank_begin + num_row_els;
      for (auto rank = rank_begin; rank != rank_end; ++rank) {
        *rank = rank - rank_begin;
      }
    } else if (row_id > col_id) {
      // Use < operation

      const auto row_begin = row_els.get();
      const auto col_begin = col_els.get();
      auto col_ptr = col_begin;
      const auto col_end = col_ptr + num_col_els;
      const auto rank_begin = ranks.get();

      for (RankType i = 0; i != num_row_els; ++i) {
        auto it = _internal::linear_upper_bound(
          col_ptr, col_end, row_begin[i], comp);
        rank_begin[i] = it - col_begin;
        col_ptr = it;
      }
    } else {
      // Use <= operation

      assert(row_id < col_id);
      const auto row_begin = row_els.get();
      const auto col_begin = col_els.get();
      auto col_ptr = col_begin;
      const auto col_end = col_ptr + num_col_els;
      const auto rank_begin = ranks.get();

      for (RankType i = 0; i != num_row_els; ++i) {
        auto it = _internal::linear_lower_bound(
          col_ptr, col_end, row_begin[i], comp);
        rank_begin[i] = it - col_begin;
        col_ptr = it;
      }
    }

    RBC::_internal::optimized::AllreduceTwotree(ranks.get(), reduced_ranks.get(), num_row_els,
                                                rank_type, MPI_SUM, row_comm);

    std::vector<T> v_row_els{ row_els.get(), row_els.get() + num_row_els };
    std::vector<RankType> v_reduced_ranks{ reduced_ranks.get(), reduced_ranks.get() + num_row_els };

    return std::tuple<std::vector<T>, std::vector<RankType>, RBC::Comm>{ std::move(v_row_els),
                                                                         std::move(v_reduced_ranks),
                                                                         std::move(col_comm) };
  } else {
    return std::tuple<std::vector<T>, std::vector<RankType>, RBC::Comm>{ std::vector<T>(),
                                                                         std::vector<RankType>(),
                                                                         std::move(col_comm) };
  }
}


template <class T, class RankType, class Comp, class RFisTags>
std::vector<Tools::Tuple<T, RankType> >
Rank(MPI_Datatype mpi_datatype,
           MPI_Datatype rank_type,
           std::vector<T> els,
	   const RBC::Comm& comm,
	   Comp comp) {
  // Rearrange element to guarantee running times of FIR

  RankType num_loc_els = els.size();
  const int size = comm.getSize();
  const int rank = comm.getRank();

  // Use Gatherm as and Rank() does not work for three PEs.
  if (size <= 3) {
    ips4o::sort(els.begin(), els.end(), comp);

    RankType num_total_els = 0;
    RBC::Allreduce(&num_loc_els, &num_total_els, 1, rank_type, MPI_SUM, comm);

    std::vector<T> gathered_els(num_total_els);
    RBC::Gatherm(els.data(), els.size(), mpi_datatype,
                 gathered_els.data(), num_total_els, 0,
                 [&comp](
                   void* a_begin, void* a_end,
                   void* b_begin, void* b_end,
                   void* out
                   ) {
        std::merge(
          static_cast<T*>(a_begin),
          static_cast<T*>(a_end),
          static_cast<T*>(b_begin),
          static_cast<T*>(b_end),
          static_cast<T*>(out),
          comp);
      },
                 comm);

    if (rank > 0) return std::vector<Tools::Tuple<T, RankType> >();

    std::vector<Tools::Tuple<T, RankType> > ret(num_total_els);
    for (size_t i = 0; i != num_total_els; ++i) {
      ret[i].first = gathered_els[i];
      ret[i].second = i;
    }

    return ret;
  }

  // FIR

  return _internal::Rank<RFisTags, T, RankType, decltype(comp)>(
    mpi_datatype, rank_type, els, comp, comm);
}

template <class T, class Comp, class RFisTags>
void Sort(MPI_Datatype mpi_datatype,
                std::vector<T>& els, const RBC::Comm& comm, Comp comp) {
  using RankType = int;
  MPI_Datatype rank_type = Common::getMpiType<RankType>();

  // Rearrange element to guarantee running times of FIR

  const int size = comm.getSize();
  const int rank = comm.getRank();

  RankType num_loc_els = els.size();

  // Use Gatherm as and Rank() does not work for three PEs.
  if (size <= 3) {
    ips4o::sort(els.begin(), els.end(), comp);

    RankType num_total_els = 0;
    RBC::Allreduce(&num_loc_els, &num_total_els, 1, rank_type, MPI_SUM, comm);
    std::vector<T> ret(num_total_els);

    RBC::Gatherm(els.data(), els.size(), mpi_datatype,
                 ret.data(), num_total_els, 0,
                 [&comp](
                   void* a_begin, void* a_end,
                   void* b_begin, void* b_end,
                   void* out
                   ) {
        std::merge(
          static_cast<T*>(a_begin),
          static_cast<T*>(a_end),
          static_cast<T*>(b_begin),
          static_cast<T*>(b_end),
          static_cast<T*>(out),
          comp);
      },
                 comm);

    if (rank > 0) {
      els = std::vector<T>();
    } else {
      els = std::move(ret);
    }

    return;
  }

  // FIR

  using RankedElement = Tools::Tuple<T, RankType>;
  using RankedElements = std::vector<RankedElement>;
  const auto mpi_ranked_type = RankedElement::MpiType(mpi_datatype,
                                                      rank_type);

  RankType num_total_els = 0;
  RankedElements ranked_els = _internal::Rank<RFisTags, T, RankType, decltype(comp)>(
    mpi_datatype, rank_type, els, comp, &num_total_els, comm);

  const int pow = tlx::round_down_to_power_of_two(size);
  assert(pow * pow > size);

  // Route elements to processes such that they are globally
  // sorted. If the number of elements is smaller than pow, each
  // process [0, num_total_els] stores one ranked element. Otherwise,
  // the elements are evenly distributed. This guarantees that the
  // hypercube exchange guarantees the required running times bounds
  // even for sparse inputs.

  // Move elements to power-of-two subcube
  {
    const RankType prev_size = ranked_els.size();
    ranked_els = _internal::RedistributeByPeId<RFisTags>(
      mpi_ranked_type, pow, ranked_els, comm);

    // Received elements and own elements are merged.
    if (static_cast<RankType>(ranked_els.size()) > prev_size) {
      std::vector<RankedElement> target(ranked_els.size());

      std::merge(ranked_els.begin(), ranked_els.begin() + prev_size,
                 ranked_els.begin() + prev_size, ranked_els.end(),
                 target.begin(),
                 [](const RankedElement& a,
                    const RankedElement& b) {
          return a.second < b.second;
        });

      ranked_els = std::move(target);
    }
  }

  // Merge received elements

  if (rank < pow) {
    els = _internal::HypercubeRedistributeByElementRank<RFisTags>(
      comm.getRank(), pow, ranked_els, mpi_ranked_type,
      num_total_els, comm);

    assert(std::is_sorted(els.begin(), els.end(), comp));
  } else {
    els.clear();
  }
}

template <class T, class Comp, class RFisTags>
std::vector<Tools::Tuple<T, int> >
Rank(MPI_Datatype mpi_datatype,
           std::vector<T> els, const RBC::Comm& comm, Comp comp) {
  using RankType = int;
  return Rank<T, RankType, decltype(comp), RFisTags>(
    mpi_datatype, Common::getMpiType<RankType>(),
    els, comm, comp);
}

template <class T, class Comp, class RFisTags>
void Sort(MPI_Datatype mpi_datatype,
                std::vector<T>& els,
                MPI_Comm comm,
                Comp comp) {
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  Sort<T, Comp, RFisTags> (mpi_datatype,
             els,
             rcomm,
             comp);
}

template <class T, class Comp, class RFisTags>
std::vector<Tools::Tuple<T, int> >
Rank(MPI_Datatype mpi_datatype,
           std::vector<T> els, MPI_Comm comm, Comp comp) {
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  return Rank<T, Comp, RFisTags>(mpi_datatype,
                    els,
                    rcomm,
                    comp);
}

template <class T, class RankType, class Comp, class RFisTags>
std::vector<Tools::Tuple<T, RankType> >
Rank(MPI_Datatype mpi_datatype,
           MPI_Datatype rank_type,
           std::vector<T> els, MPI_Comm comm, Comp comp) {
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  return Rank<T, RankType, Comp, RFisTags>(mpi_datatype,
                    rank_type,
                    els,
                    rcomm,
                    comp);
}
}  // end namespace RFis
