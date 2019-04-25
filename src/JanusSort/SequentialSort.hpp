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
#include <iostream>
#include <utility>
#include <vector>

#include "../../include/RBC/RBC.hpp"
#include "Constants.hpp"
#include "ips4o.hpp"
#include "QSInterval.hpp"

namespace JanusSort {
template <typename T>
class SequentialSort_SQS {
 public:
/*
 * Sort all local intervals
 */
  template <class Sorter>
  static void sortLocalIntervals(Sorter& sorter,
                                 std::vector<QSInterval_SQS<T> >& local_intervals) {
    for (size_t i = 0; i < local_intervals.size(); i++) {
      T* data_ptr = local_intervals[i].m_data->data();
      sorter(data_ptr + local_intervals[i].m_local_start,
             data_ptr + local_intervals[i].m_local_end);
    }
  }

/*
 * Sort the saved intervals with exactly two PEs
 */
  template <class Compare, class Sorter>
  static int64_t sortTwoPEIntervals(Compare&& comp, Sorter& sorter,
                                    std::vector<QSInterval_SQS<T> >& two_PE_intervals) {
    if (two_PE_intervals.size() == 2)
      return sortOnTwoPEsJanus(two_PE_intervals[0], two_PE_intervals[1],
                               std::forward<Compare>(comp), sorter);
    else if (two_PE_intervals.size() == 1)
      return sortOnTwoPEs(two_PE_intervals[0], std::forward<Compare>(comp), sorter);
    else
      assert(two_PE_intervals.size() == 0);
    return 0;
  }

 private:
/*
 * Sort an interval with only two PEs sequentially to terminate recursion
 */
  template <class Compare, class Sorter>
  static int64_t sortOnTwoPEs(QSInterval_SQS<T>& ival, Compare&& comp, Sorter& sorter) {
    RBC::Request requests[2];
    T* data_ptr = ival.m_data->data();
    // gather all elements on both PEs
    int partner = (ival.m_rank + 1) % 2;
    RBC::Isend(data_ptr + ival.m_local_start, ival.m_local_elements, ival.m_mpi_type,
               partner, Constants::TWO_PE, ival.m_comm, &requests[0]);

    int recv_elements = -1, flag = 0;
    MPI_Status status;
    while (recv_elements == -1) {
      RBC::Iprobe(partner, Constants::TWO_PE, ival.m_comm, &flag, &status);
      if (flag) {
        MPI_Get_count(&status, ival.m_mpi_type, &recv_elements);
      }
      int x;
      RBC::Test(&requests[0], &x, MPI_STATUS_IGNORE);
    }

    int64_t total_elements = ival.m_local_elements + recv_elements;
    T* tmp_buffer = new T[total_elements];
    RBC::Irecv(tmp_buffer, recv_elements, ival.m_mpi_type, partner, Constants::TWO_PE, ival.m_comm,
               &requests[1]);

    std::memcpy(tmp_buffer + recv_elements, data_ptr + ival.m_local_start,
                ival.m_local_elements * sizeof(T));
    RBC::Waitall(2, &requests[0], MPI_STATUSES_IGNORE);

    partitionAndSort(ival, std::forward<Compare>(comp), sorter, tmp_buffer, recv_elements);

    delete[] tmp_buffer;
    return total_elements;
  }

/*
 * Partition the buffer and sort one partition
 */
  template <class Compare, class Sorter>
  static void partitionAndSort(QSInterval_SQS<T>& ival, Compare&& comp, Sorter& sorter,
                               T* buffer, int recv_elements) {
    T* nth_element;
    if (ival.m_rank == 0)
      nth_element = buffer + ival.m_local_elements;
    else
      nth_element = buffer + recv_elements;

    int64_t total_elements = ival.m_local_elements + recv_elements;
    std::nth_element(buffer, nth_element, buffer + total_elements, std::forward<Compare>(comp));

    if (ival.m_rank == 0) {
      sorter(buffer, nth_element + 1);
    } else {
      sorter(nth_element, buffer + total_elements);
    }

    T* copy_ptr;
    if (ival.m_rank == 0)
      copy_ptr = buffer;
    else
      copy_ptr = buffer + recv_elements;
    T* data_ptr = ival.m_data->data();
    std::memcpy(data_ptr + ival.m_local_start, copy_ptr,
                ival.m_local_elements * sizeof(T));
  }

/*
 * Sort two intervals with two PEs simultaneously
 */
  template <class Compare, class Sorter>
  static int64_t sortOnTwoPEsJanus(QSInterval_SQS<T>& ival_1,
                                   QSInterval_SQS<T>& ival_2,
                                   Compare&& comp,
                                   Sorter& sorter) {
    RBC::Request requests[4];
    T* data_ptr_1 = ival_1.m_data->data();
    T* data_ptr_2 = ival_2.m_data->data();

    int partner_1 = (ival_1.m_rank + 1) % 2;
    int partner_2 = (ival_2.m_rank + 1) % 2;
    RBC::Isend(data_ptr_1 + ival_1.m_local_start, ival_1.m_local_elements, ival_1.m_mpi_type,
               partner_1, Constants::TWO_PE, ival_1.m_comm, &requests[0]);
    RBC::Isend(data_ptr_2 + ival_2.m_local_start, ival_2.m_local_elements, ival_2.m_mpi_type,
               partner_2, Constants::TWO_PE, ival_2.m_comm, &requests[2]);

    int recv_elements_1 = -1, flag_1 = 0, recv_elements_2 = -1, flag_2 = 0;
    MPI_Status status_1, status_2;
    while (recv_elements_1 == -1 || recv_elements_2 == -1) {
      RBC::Iprobe(partner_1, Constants::TWO_PE, ival_1.m_comm, &flag_1, &status_1);
      if (flag_1)
        MPI_Get_count(&status_1, ival_1.m_mpi_type, &recv_elements_1);
      RBC::Iprobe(partner_2, Constants::TWO_PE, ival_2.m_comm, &flag_2, &status_2);
      if (flag_2)
        MPI_Get_count(&status_2, ival_2.m_mpi_type, &recv_elements_2);
      int x1, x2;
      RBC::Test(&requests[0], &x1, MPI_STATUS_IGNORE);
      RBC::Test(&requests[2], &x2, MPI_STATUS_IGNORE);
    }

    int64_t total_elements_1 = ival_1.m_local_elements + recv_elements_1;
    int64_t total_elements_2 = ival_2.m_local_elements + recv_elements_2;
    T* buffer_1 = new T[total_elements_1];
    T* buffer_2 = new T[total_elements_2];

    RBC::Irecv(buffer_1, recv_elements_1, ival_1.m_mpi_type, partner_1, Constants::TWO_PE, ival_1.m_comm,
               &requests[1]);
    RBC::Irecv(buffer_2, recv_elements_2, ival_2.m_mpi_type, partner_2, Constants::TWO_PE, ival_2.m_comm,
               &requests[3]);

    std::memcpy(buffer_1 + recv_elements_1, data_ptr_1 + ival_1.m_local_start,
                ival_1.m_local_elements * sizeof(T));
    std::memcpy(buffer_2 + recv_elements_2, data_ptr_2 + ival_2.m_local_start,
                ival_2.m_local_elements * sizeof(T));

    RBC::Waitall(4, &requests[0], MPI_STATUSES_IGNORE);

    // partition data and sort one partition
    partitionAndSort(ival_1, std::forward<Compare>(comp), sorter, buffer_1, recv_elements_1);
    partitionAndSort(ival_2, std::forward<Compare>(comp), sorter, buffer_2, recv_elements_2);

    delete[] buffer_1;
    delete[] buffer_2;

    return total_elements_1 + total_elements_2;
  }
};
}  // namespace JanusSort
