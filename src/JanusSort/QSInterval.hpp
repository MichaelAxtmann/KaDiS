/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiS).
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (C) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
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

#include <vector>

#include <RBC.hpp>

namespace JanusSort {
/*
 * This struct represents a interval of data used in a Quicksort call
 */
template <typename T>
struct QSInterval_SQS {
  QSInterval_SQS() { }

  QSInterval_SQS(std::vector<T>* data, T* buffer, int64_t split_size, int64_t extra_elements,
                 int64_t local_start, int64_t local_end,
                 RBC::Comm comm, int64_t offset_first_PE, int64_t offset_last_PE,
                 MPI_Datatype mpi_type, int seed, int64_t min_samples, bool add_pivot,
                 bool blocking_priority, bool evenly_distributed = true) :
    m_data(data),
    m_buffer(buffer),
    m_split_size(split_size),
    m_extra_elements(extra_elements),
    m_local_start(local_start),
    m_local_end(local_end),
    m_missing_first_pe(offset_first_PE),
    m_missing_last_pe(offset_last_PE),
    m_min_samples(min_samples),
    m_seed(seed),
    m_comm(comm),
    m_mpi_type(mpi_type),
    m_evenly_distributed(evenly_distributed),
    m_add_pivot(add_pivot),
    m_blocking_priority(blocking_priority) {
    this->m_seed = seed * 48271 % 2147483647;

    if (!comm.isEmpty()) {
      RBC::Comm_size(comm, &m_number_of_pes);
      RBC::Comm_rank(comm, &m_rank);
    } else {
      m_number_of_pes = -1;
      m_rank = -1;
    }

    m_start_pe = 0;
    m_end_pe = m_number_of_pes - 1;
    m_local_elements = local_end - local_start;
  }

  int getRankFromIndex(int64_t global_index) const {
    int64_t idx = m_extra_elements * (m_split_size + 1);
    if (global_index < idx) {
      return global_index / (m_split_size + 1);
    } else {
      if (m_split_size == 0) {
        return m_extra_elements;
      } else {
        int64_t idx_dif = global_index - idx;
        int r = idx_dif / m_split_size;
        return m_extra_elements + r;
      }
    }
  }

  int getOffsetFromIndex(int64_t global_index) const {
    int64_t idx = m_extra_elements * (m_split_size + 1);
    if (global_index < idx) {
      return global_index % (m_split_size + 1);
    } else {
      int64_t idx_dif = global_index - idx;
      return idx_dif % m_split_size;
    }
  }

  int64_t getIndexFromRank(int rank) const {
    if (rank <= m_extra_elements)
      return rank * (m_split_size + 1);
    else
      return m_extra_elements * (m_split_size + 1) + (rank - m_extra_elements) * m_split_size;
  }

  int64_t getSplitSize() const {
    return getSplitSize(m_rank);
  }

  int64_t getSplitSize(int rank) const {
    if (rank < m_extra_elements)
      return m_split_size + 1;
    else
      return m_split_size;
  }

  int64_t getLocalElements() const {
    return getLocalElements(m_rank);
  }

  int64_t getLocalElements(int rank) const {
    int elements = getSplitSize(rank);
    if (rank == 0)
      elements -= m_missing_first_pe;
    if (rank == m_number_of_pes - 1)
      elements -= m_missing_last_pe;
    return elements;
  }

  std::vector<T>* m_data;
  T* m_buffer;
  int64_t m_split_size, m_extra_elements, m_local_start, m_local_end,
    m_missing_first_pe, m_missing_last_pe,
    m_local_elements, m_presum_small, m_presum_large,
    m_local_small_elements, m_local_large_elements,
    m_global_small_elements, m_global_large_elements, m_global_elements,
    m_bound1, m_split, m_bound2, m_min_samples;
  int m_seed, m_number_of_pes, m_rank, m_start_pe, m_end_pe;
  RBC::Comm m_comm;
  MPI_Datatype m_mpi_type;
  bool m_evenly_distributed, m_add_pivot, m_blocking_priority;
};
}  // namespace JanusSort
