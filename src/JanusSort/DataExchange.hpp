/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiS).
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (C) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
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
#include <cstring>

#include "../../include/RBC/RBC.hpp"
#include "Constants.hpp"
#include "QSInterval.hpp"
#include "RequestVector.hpp"

namespace JanusSort {
template <typename T>
class DataExchange_SQS {
 public:
/*
 * Exchange the data with other PEs
 */
  static void exchangeData(QSInterval_SQS<T>& ival) {
    int64_t recv_small = 0, recv_large = 0, recv_count_small = 0, recv_count_large = 0;
    RequestVector requests;
    // copy current data (that will be send) into buffer
    copyDataToBuffer(ival);

    // calculate how much data need to be received
    getRecvCount(ival, recv_small, recv_large);

    int64_t recv_elements = recv_small + recv_large;
    assert(ival.getLocalElements() == recv_elements);
    if (!ival.m_evenly_distributed)
      ival.m_data->resize(recv_elements);

    sendData(ival, requests, recv_small,
             recv_count_small, recv_count_large);

    T* data_ptr = ival.m_data->data();
    // receive data
    while (recv_count_small < recv_small || recv_count_large < recv_large) {
      if (recv_count_small < recv_small) {
        receiveData(ival.m_comm, requests,
                    data_ptr + ival.m_local_start + recv_count_small,
                    recv_count_small, recv_small, Constants::EXCHANGE_SMALL, ival.m_mpi_type);
      }
      if (recv_count_large < recv_large) {
        receiveData(ival.m_comm, requests,
                    data_ptr + ival.m_local_start + recv_small + recv_count_large,
                    recv_count_large, recv_large, Constants::EXCHANGE_LARGE, ival.m_mpi_type);
      }
    }
    requests.waitAll();
  }

/*
 * Exchange the data with other PEs on both intervals simultaneously
 */
  static void exchangeDataJanus(QSInterval_SQS<T>& ival_left, QSInterval_SQS<T>& ival_right) {
    int64_t recv_small_l = 0, recv_large_l = 0, recv_count_small_l = 0, recv_count_large_l = 0;
    int64_t recv_small_r = 0, recv_large_r = 0, recv_count_small_r = 0, recv_count_large_r = 0;
    RequestVector requests;

    // copy current data (that will be send) into buffer
    copyDataToBuffer(ival_left);
    copyDataToBuffer(ival_right);

    // calculate how much data need to be send and received, then start non-blocking sends
    getRecvCount(ival_left, recv_small_l, recv_large_l);
    getRecvCount(ival_right, recv_small_r, recv_large_r);
    sendData(ival_left, requests, recv_small_l,
             recv_count_small_l, recv_count_large_l);
    sendData(ival_right, requests, recv_small_r,
             recv_count_small_r, recv_count_large_r);

    T* data_ptr_left = ival_left.m_data->data();
    T* data_ptr_right = ival_right.m_data->data();
    // receive data
    while ((recv_count_small_l < recv_small_l) || (recv_count_large_l < recv_large_l) ||
           (recv_count_small_r < recv_small_r) || (recv_count_large_r < recv_large_r)) {
      if (recv_count_small_l < recv_small_l) {
        receiveData(ival_left.m_comm, requests,
                    data_ptr_left + ival_left.m_local_start + recv_count_small_l,
                    recv_count_small_l, recv_small_l, Constants::EXCHANGE_SMALL, ival_left.m_mpi_type);
      }
      if (recv_count_large_l < recv_large_l) {
        receiveData(ival_left.m_comm, requests,
                    data_ptr_left + ival_left.m_local_start + recv_small_l + recv_count_large_l,
                    recv_count_large_l, recv_large_l, Constants::EXCHANGE_LARGE, ival_left.m_mpi_type);
      }
      if (recv_count_small_r < recv_small_r) {
        receiveData(ival_right.m_comm, requests,
                    data_ptr_right + ival_right.m_local_start + recv_count_small_r,
                    recv_count_small_r, recv_small_r, Constants::EXCHANGE_SMALL, ival_right.m_mpi_type);
      }
      if (recv_count_large_r < recv_large_r) {
        receiveData(ival_right.m_comm, requests,
                    data_ptr_right + ival_right.m_local_start + recv_small_r + recv_count_large_r,
                    recv_count_large_r, recv_large_r, Constants::EXCHANGE_LARGE, ival_right.m_mpi_type);
      }
    }
    requests.waitAll();
  }

 private:
/*
 * Calculate how much small and large data need to be received
 */
  static void getRecvCount(QSInterval_SQS<T>& ival, int64_t& recv_small, int64_t& recv_large) {
    int small_end_pe = ival.getRankFromIndex(ival.m_missing_first_pe + ival.m_global_small_elements - 1);
    int large_start_pe = ival.getRankFromIndex(ival.m_missing_first_pe + ival.m_global_small_elements);
    int local_elements = ival.getLocalElements();

    if (large_start_pe > ival.m_rank) {
      recv_small = local_elements;
      recv_large = 0;
    } else if (small_end_pe < ival.m_rank) {
      recv_small = 0;
      recv_large = local_elements;
    } else {
      recv_small = ival.getOffsetFromIndex(ival.m_missing_first_pe + ival.m_global_small_elements)
                   - ival.m_local_start;
      recv_large = local_elements - recv_small;
    }
  }

/*
 * Calculate how much data need to be send then start non-blocking sends
 */
  static void sendData(QSInterval_SQS<T>& ival,
                       RequestVector& requests, int64_t& recv_small,
                       int64_t& recv_count_small, int64_t& recv_count_large) {
    int64_t small_start = ival.m_local_start;
    int64_t large_start = ival.m_local_start + ival.m_local_small_elements;
    int64_t large_end = ival.m_local_end;
    int64_t global_idx_small = ival.m_missing_first_pe + ival.m_presum_small;
    int64_t global_idx_large = ival.m_missing_first_pe + ival.m_global_small_elements
                               + ival.m_presum_large;
    T* buffer_small = ival.m_data->data() + ival.m_local_start;
    T* buffer_large = ival.m_data->data() + ival.m_local_start + recv_small;

    // send small elements
    recv_count_small = sendDataRecursive(ival, small_start,
                                         large_start, global_idx_small, buffer_small,
                                         requests, Constants::EXCHANGE_SMALL);

    // send large elements
    recv_count_large = sendDataRecursive(ival, large_start,
                                         large_end, global_idx_large, buffer_large,
                                         requests, Constants::EXCHANGE_LARGE);
  }

/*
 * Returns the number of elements that have been copied locally into the recv_buffer
 */
  static int sendDataRecursive(QSInterval_SQS<T>& ival, int64_t local_start_idx,
                               int64_t local_end_idx, int64_t global_start_idx,
                               T* recv_buffer, RequestVector& requests,
                               int tag) {
    // return if no elements need to be send
    if (local_start_idx >= local_end_idx)
      return 0;

    int target_rank = ival.getRankFromIndex(global_start_idx);
    int64_t send_max = ival.getIndexFromRank(target_rank + 1) - global_start_idx;
    int64_t local_elements = local_end_idx - local_start_idx;
    int64_t send_count = std::min(send_max, local_elements);

    int64_t copied_local = 0;
    if (target_rank == ival.m_rank) {
      copied_local += send_count;
      std::memcpy(recv_buffer, ival.m_buffer + local_start_idx, send_count * sizeof(T));
    } else {
      RBC::Request req;
      RBC::Isend(ival.m_buffer + local_start_idx, send_count, ival.m_mpi_type,
                 target_rank, tag, ival.m_comm, &req);
      requests.push_back(req);
    }

    if (local_elements > send_count) {
      // send remaining data
      copied_local += sendDataRecursive(ival, local_start_idx + send_count,
                                        local_end_idx, ival.getIndexFromRank(target_rank + 1),
                                        recv_buffer + copied_local, requests, tag);
    }
    return copied_local;
  }

/*
 * Starts a non-blocking receive if data can be received
 */
  static void receiveData(RBC::Comm const& comm, RequestVector& requests,
                          void* recvbuf, int64_t& recv_count, int64_t recv_total, int tag,
                          MPI_Datatype mpi_type) {
    if (recv_count < recv_total) {
      int ready;
      MPI_Status status;
      RBC::Iprobe(MPI_ANY_SOURCE, tag, comm, &ready, &status);
      if (ready) {
        int count;
        MPI_Get_count(&status, mpi_type, &count);
//                std::cout << W(recv_total) << W(recv_count) << W(count) << std::endl;
        assert(recv_count + count <= recv_total);
        int source = RBC::get_Rank_from_Status(comm, status);
        RBC::Request req;
        RBC::Irecv(recvbuf, count, mpi_type, source,
                   tag, comm, &req);
        requests.push_back(req);
        recv_count += count;
      }
    }
  }

/*
 * Copy data into the send buffer such that all small (and large) elements
 * are stored consecutively
 */
  static void copyDataToBuffer(QSInterval_SQS<T>& ival) {
    T* data_ptr = ival.m_data->data();
    int64_t copy = ival.m_bound1 - ival.m_local_start;
    std::memcpy(ival.m_buffer + ival.m_local_start, data_ptr + ival.m_local_start, copy * sizeof(T));

    int64_t small_right = ival.m_bound2 - ival.m_split;
    copy = ival.m_split - ival.m_bound1;
    std::memcpy(ival.m_buffer + ival.m_bound1 + small_right, data_ptr + ival.m_bound1, copy * sizeof(T));

    copy = ival.m_bound2 - ival.m_split;
    std::memcpy(ival.m_buffer + ival.m_bound1, data_ptr + ival.m_split, copy * sizeof(T));

    copy = ival.m_local_end - ival.m_bound2;
    std::memcpy(ival.m_buffer + ival.m_bound2, data_ptr + ival.m_bound2, copy * sizeof(T));
  }
};
}  // namespace JanusSort
