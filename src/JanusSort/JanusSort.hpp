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
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "../../include/RBC/RBC.hpp"
#include "Constants.hpp"
#include "DataExchange.hpp"
#include "PivotSelection.hpp"
#include "QSInterval.hpp"
#include "SequentialSort.hpp"
#include "TbSplitter.hpp"

namespace JanusSort {
/*
 * This class represents the Quicksort algorithm
 */
template <typename T>
class Sorter {
 public:
/**
 * Constructor
 * @param seed Seed for RNG, has to be the same for all PEs
 * @param min_samples Minimal number of samples for the pivot selection
 * @param barriers Use barriers to measure the running time of the algorithm phases
 * @param split_MPI_comm If true, split communicators using MPI_Comm_create/MPI_Comm_split
 *      else use the RBC split operations
 * @param use_MPI_collectives If true, use the collective operations provided by MPI whenever possible,
 *      else always use the collective operations of RBC
 * @param add_pivot If true, use k1+k2+k3 as the number of samples,
 *      else use max(k1,k2,k3)
 */
  Sorter(MPI_Datatype mpi_type, int seed, int64_t min_samples = 64,
         bool barriers = false, bool split_MPI_comm = false,
         bool use_MPI_collectives = false, bool add_pivot = false)

    : m_mpi_type(mpi_type),
      m_seed(seed),
      m_barriers(barriers),
      m_split_mpi_comm(split_MPI_comm),
      m_use_mpi_collectives(use_MPI_collectives),
      m_add_pivots(add_pivot),
      m_min_samples(min_samples) {
    // SortingDatatype<T>::getMPIDatatype(&m_mpi_typen);
  }

// static bool validDatatype() {
//     MPI_Datatype mpi_type;
//     return SortingDatatype<T>::getMPIDatatype(&mpi_type) != MPI_ERR_TYPE;
// }

  ~Sorter() { }

/**
 * Sorts the input data
 * @param mpi_comm MPI commuicator (all ranks have to call the function)
 * @param data_vec Vector that contains the input data
 * @param global_elements The total number of elements on all PEs.
 *          Set the parameter to -1 if unknown or if the global input
 *          is not evenly distributed, i.e., x, ..., x, x-1, ..., x-1.
 *          If the parameter is not set to -1, the algorithm runs
 *          slightly faster for small inputs.
 */
  void sort(MPI_Comm mpi_comm, std::vector<T>& data_vec, int64_t global_elements = -1) {
    sort(mpi_comm, data_vec, global_elements, std::less<T>());
  }

/**
 * Sorts the input data with a custom compare operator
 * @param mpi_comm MPI commuicator (all ranks have to call the function)
 * @param data_vec Vector that contains the input data
 * @param global_elements The total number of elements on all PEs.
 *          Set the parameter to -1 if unknown or if the global input
 *          is not evenly distributed, i.e., x, ..., x, x-1, ..., x-1.
 *          If the parameter is not set to -1, the algorithm runs
 *          slightly faster for small inputs.
 * @param comp The compare operator
 */
  template <class Compare>
  void sort(MPI_Comm mpi_comm, std::vector<T>& data, Compare&& comp, int64_t global_elements) {
    RBC::Comm comm;
    RBC::Create_Comm_from_MPI(mpi_comm, &comm, m_use_mpi_collectives, m_split_mpi_comm);
    MPI_Barrier(mpi_comm);
    sort_range(comm, data, std::forward<Compare>(comp), global_elements);
  }

/**
 * Sort data on an RBC communicator
 * @param mpi_comm MPI commuicator (all ranks have to call the function)
 * @param data_vec Vector that contains the input data
 * @param global_elements The total number of elements on all PEs, set to -1 if unknown of if the global input is not evenly distributed, i.e., x, ..., x, x-1, ..., x-1.
 * @param comp The compare operator
 */
  template <class Compare>
  void sort_range(RBC::Comm comm, std::vector<T>& data,
                  Compare&& comp, int64_t global_elements) {
    assert(!comm.isEmpty());

    double total_start = getTime();
    // m_parent_comm = comm.GetMpiComm();
    this->m_data = &data;

    int size, rank;
    RBC::Comm_size(comm, &size);
    RBC::Comm_rank(comm, &rank);
    m_generator.seed(m_seed);
    m_sample_generator.seed(m_seed + rank);

    QSInterval_SQS<T> ival;
    assert(global_elements >= -1);
    if (global_elements == -1) {
      m_buffer = nullptr;
      // split_size and extra_elements will be assigned in calculateExchange
      // local_elements and global_end will be changed after dataExchange
      ival = QSInterval_SQS<T>(&data, m_buffer, -1, -1, 0,
                               data.size(), comm, 0, 0, m_mpi_type,
                               m_seed, m_min_samples, m_add_pivots,
                               true, false);
    } else {
      int64_t split_size = global_elements / size;
      int64_t extra_elements = global_elements % size;
      m_buffer = new T[data.size()];
      ival = QSInterval_SQS<T>(&data, m_buffer, split_size, extra_elements, 0,
                               data.size(), comm, 0, 0, m_mpi_type, m_seed,
                               m_min_samples, m_add_pivots, true);
    }

    auto sorter = ips4o::make_sorter<T*>(std::forward<Compare>(comp));

    /* Recursive */
    quickSort(ival, std::forward<Compare>(comp));

    delete[] m_buffer;

    /* Base Cases */
    double start, end;
    if (m_barriers)
      RBC::Barrier(comm);
    start = getTime();
    sortTwoPEIntervals(std::forward<Compare>(comp), sorter);
    end = getTime();
    t_sort_two = end - start;

    start = getTime();
    sortLocalIntervals(sorter);
    end = getTime();
    t_sort_local = end - start;

    double total_end = getTime();
    t_runtime = (total_end - total_start);
  }

/**
 * @return The maximal depth of recursion
 */
  int getDepth() {
    return m_depth;
  }

/**
 * Get timers and their names
 * @param timer_names Vector containing the names of the timers
 * @param max_timers Vector containing the maximal timer value across all PEs
 * @param comm RBC communicator
 */
  void getTimers(std::vector<std::string>& timer_names,
                 std::vector<double>& max_timers, RBC::Comm comm) {
    std::vector<double> timers;
    int size, rank;
    RBC::Comm_size(comm, &size);
    RBC::Comm_rank(comm, &rank);
    if (m_barriers) {
      timers.push_back(t_pivot);
      timer_names.push_back("pivot");
      timers.push_back(t_partition);
      timer_names.push_back("partition");
      timers.push_back(t_calculate);
      timer_names.push_back("calculate");
      timers.push_back(t_exchange);
      timer_names.push_back("exchange");
//            timers.push_back(t_sort_two);
//            timer_names.push_back("sort_two");
//            timers.push_back(t_sort_local);
//            timer_names.push_back("sort_local");
      timers.push_back(t_sort_local + t_sort_two);
      timer_names.push_back("base_cases");
      double sum = 0.0;
      for (size_t i = 0; i < timers.size(); i++)
        sum += timers[i];
      timers.push_back(sum);
      timer_names.push_back("sum");
    }
//        timers.push_back(bc1_elements);
//        timer_names.push_back("BaseCase1_elements");
//        timers.push_back(bc2_elements);
//        timer_names.push_back("BaseCase2_elements");
//        timers.push_back(t_runtime);
//        timer_names.push_back("runtime");
    timers.push_back(m_depth);
    timer_names.push_back("depth");
    timers.push_back(t_create_comms);
    timer_names.push_back("create_comms");

    for (size_t i = 0; i < timers.size(); i++) {
      double time = 0.0;
      RBC::Reduce(&timers[i], &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
      max_timers.push_back(time);
//            MPI_Reduce(&timers[i], &time, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
//            min_timers.push_back(time);
//            MPI_Reduce(&timers[i], &time, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
//            avg_timers.push_back(time / size);
    }

//        double runtimes[size];
//        MPI_Allgather(&t_runtime, 1, MPI_DOUBLE, runtimes, 1, MPI_DOUBLE, comm);
//        double t_max = 0.0;
//        int rank_max = 0;
//        for (int i = 0; i < size; i++) {
//            if (runtimes[i] > t_max) {
//                t_max = runtimes[i];
//                rank_max = i;
//            }
//        }
//        if (rank_max == 0) {
//            max_timers.push_back(bc1_elements);
//            timer_names.push_back("slow_BaseCase1_elements");
//            max_timers.push_back(bc2_elements);
//            timer_names.push_back("slow_BaseCase2_elements");
//            max_timers.push_back(depth);
//            timer_names.push_back("slow_depth");
//        } else {
//            if (rank == rank_max) {
//                double slowest_timers[3] = {bc1_elements,
//                    bc2_elements, static_cast<double>(depth)};
//                MPI_Send(slowest_timers, 5, MPI_DOUBLE, 0, 0, comm);
//            }
//            if (rank == 0) {
//                double slowest_timers[5];
//                MPI_Recv(slowest_timers, 5, MPI_DOUBLE, rank_max, 0, comm,
//                        MPI_STATUS_IGNORE);
//                max_timers.push_back(slowest_timers[0]);
//                timer_names.push_back("slow_BaseCase1_elements");
//                max_timers.push_back(slowest_timers[1]);
//                timer_names.push_back("slow_BaseCase2_elements");
//                max_timers.push_back(slowest_timers[2]);
//                timer_names.push_back("slow_depth");
//            }
//        }
  }

 private:
/*
 * Execute the Quicksort algorithm on the given QSInterval
 */
  template <class Compare>
  void quickSort(QSInterval_SQS<T>& ival, Compare&& comp) {
    m_depth++;

    // Check if recursion should be ended
    if (isBaseCase(ival))
      return;

    /* Pivot Selection */
    T pivot;
    int64_t split_idx;
    double t_start, t_end;
    t_start = startTime(ival.m_comm);
    bool zero_global_elements;
    getPivot(ival, pivot, split_idx, std::forward<Compare>(comp), zero_global_elements);
    t_end = getTime();
    t_pivot += (t_end - t_start);

    if (zero_global_elements)
      return;

    /* Partitioning */
    t_start = startTime(ival.m_comm);
    int64_t bound1, bound2;
    partitionData(ival, pivot, split_idx, &bound1, &bound2, std::forward<Compare>(comp));
    t_end = getTime();
    t_partition += (t_end - t_start);

    /* Calculate how data has to be exchanged */
    t_start = startTime(ival.m_comm);
    calculateExchangeData(ival, bound1, split_idx, bound2);
    t_end = getTime();
    t_calculate += (t_end - t_start);

    /* Exchange data */
    t_start = startTime(ival.m_comm);
    exchangeData(ival);
    t_end = getTime();
    t_exchange += (t_end - t_start);
    t_vec_exchange.push_back(t_end - t_start);

    /* Create QSIntervals for the next recursion level */
    int64_t mid, offset;
    int left_size;
    bool janus;
    calculateSplit(ival, left_size, offset, janus, mid);

    RBC::Comm comm_left, comm_right;
    if (m_use_mpi_collectives)
      t_start = startTime_barrier(ival.m_comm);
    else
      t_start = startTime(ival.m_comm);
    createNewCommunicators(ival, left_size, janus, &comm_left, &comm_right);
    t_end = getTime();
    t_create_comms += (t_end - t_start);

    QSInterval_SQS<T> ival_left, ival_right;
    createIntervals(ival, offset, left_size, janus,
                    mid, comm_left, comm_right, ival_left, ival_right);

    bool sort_left = false, sort_right = false;
    if (ival.m_rank <= ival_left.m_end_pe)
      sort_left = true;
    if (ival.m_rank >= ival_left.m_end_pe) {
      if (ival.m_rank > ival_left.m_end_pe || janus)
        sort_right = true;
    }

    /* Call recursively */
    if (sort_left && sort_right) {
      janusQuickSort(ival_left, ival_right, std::forward<Compare>(comp));
    } else if (sort_left) {
      quickSort(ival_left, std::forward<Compare>(comp));
    } else if (sort_right) {
      quickSort(ival_right, std::forward<Compare>(comp));
    }
  }

/*
 * Execute the Quicksort algorithm as janus PE
 */
  template <class Compare>
  void janusQuickSort(QSInterval_SQS<T>& ival_left, QSInterval_SQS<T>& ival_right,
                      Compare&& comp) {
    m_depth++;

    // Check if recursion should be ended
    if (isBaseCase(ival_left)) {
      quickSort(ival_right, std::forward<Compare>(comp));
      return;
    }
    if (isBaseCase(ival_right)) {
      quickSort(ival_left, std::forward<Compare>(comp));
      return;
    }

    /* Pivot Selection */
    T pivot_left, pivot_right;
    int64_t split_idx_left, split_idx_right;
    double t_start, t_end;
    t_start = startTimeJanus(ival_left.m_comm, ival_right.m_comm);
    getPivotJanus(ival_left, ival_right, pivot_left, pivot_right,
                  split_idx_left, split_idx_right, std::forward<Compare>(comp));
    t_end = getTime();
    t_pivot += (t_end - t_start);

    /* Partitioning */
    t_start = startTimeJanus(ival_left.m_comm, ival_right.m_comm);
    int64_t bound1_left, bound2_left, bound1_right, bound2_right;
    partitionData(ival_left, pivot_left, split_idx_left, &bound1_left,
                  &bound2_left, std::forward<Compare>(comp));
    partitionData(ival_right, pivot_right, split_idx_right, &bound1_right,
                  &bound2_right, std::forward<Compare>(comp));
    t_end = getTime();
    t_partition += (t_end - t_start);

    /* Calculate how data has to be exchanged */
    t_start = startTimeJanus(ival_left.m_comm, ival_right.m_comm);
    calculateExchangeDataJanus(ival_left, ival_right, bound1_left, split_idx_left,
                               bound2_left, bound1_right, split_idx_right, bound2_right);
    t_end = getTime();
    t_calculate += (t_end - t_start);

    /* Exchange Data */
    t_start = startTimeJanus(ival_left.m_comm, ival_right.m_comm);
    exchangeDataJanus(ival_left, ival_right);
    t_end = getTime();
    t_exchange += (t_end - t_start);
    t_vec_exchange.push_back(t_end - t_start);

    /* Create QSIntervals for the next recursion level */
    int64_t mid_left, mid_right, offset_left, offset_right;
    int left_size_left, left_size_right;
    bool janus_left, janus_right;
    calculateSplit(ival_left, left_size_left, offset_left, janus_left, mid_left);
    calculateSplit(ival_right, left_size_right, offset_right, janus_right, mid_right);
    RBC::Comm left1, right1, left2, right2;
    if (m_use_mpi_collectives)
      t_start = startTimeJanus_barrier(ival_left.m_comm, ival_right.m_comm);
    else
      t_start = startTimeJanus(ival_left.m_comm, ival_right.m_comm);
    createNewCommunicatorsJanus(ival_left, ival_right, left_size_left, janus_left,
                                left_size_right, janus_right, &left1, &right1, &left2, &right2);
    t_end = getTime();
    t_create_comms += (t_end - t_start);

    QSInterval_SQS<T> ival_left_left, ival_right_left,
      ival_left_right, ival_right_right;
    createIntervals(ival_left, offset_left, left_size_left,
                    janus_left, mid_left, left1, right1,
                    ival_left_left, ival_right_left);
    createIntervals(ival_right, offset_right, left_size_right,
                    janus_right, mid_right, left2, right2,
                    ival_left_right, ival_right_right);

    bool sort_left = false, sort_right = false;
    QSInterval_SQS<T>* left_i, * right_i;
    // Calculate new left interval and if it need to be sorted
    if (ival_right_left.m_number_of_pes == 1) {
      addLocalInterval(ival_right_left);
      left_i = &ival_left_left;
      if (ival_left_left.m_number_of_pes == ival_left.m_number_of_pes)
        sort_left = true;
    } else {
      left_i = &ival_right_left;
      sort_left = true;
    }
    // Calculate new right interval and if it need to be sorted
    if (ival_left_right.m_number_of_pes == 1) {
      addLocalInterval(ival_left_right);
      right_i = &ival_right_right;
      if (ival_right_right.m_number_of_pes == ival_right.m_number_of_pes)
        sort_right = true;
    } else {
      right_i = &ival_left_right;
      sort_right = true;
    }

    /* Call recursively */
    if (sort_left && sort_right) {
      janusQuickSort(*left_i, *right_i, std::forward<Compare>(comp));
    } else if (sort_left) {
      quickSort(*left_i, std::forward<Compare>(comp));
    } else if (sort_right) {
      quickSort(*right_i, std::forward<Compare>(comp));
    }
  }

/**
 * Check for base cases
 * @return true if base case, false if no base case
 */
  bool isBaseCase(QSInterval_SQS<T>& ival) {
    if (ival.m_rank == -1)
      return true;
    if (ival.m_number_of_pes == 2) {
      addTwoPEInterval(ival);
      return true;
    }
    if (ival.m_number_of_pes == 1) {
      addLocalInterval(ival);
      return true;
    }
    return false;
  }

/*
 * Returns the current time
 */
  double getTime() {
    return MPI_Wtime();
  }

  double startTime(RBC::Comm& comm) {
    if (!m_barriers)
      return getTime();
    RBC::Request req;
    RBC::Ibarrier(comm, &req);
    RBC::Wait(&req, MPI_STATUS_IGNORE);
    return getTime();
  }

  double startTime_barrier(RBC::Comm& comm) {
    RBC::Request req;
    RBC::Ibarrier(comm, &req);
    RBC::Wait(&req, MPI_STATUS_IGNORE);
    return getTime();
  }

  double startTimeJanus(RBC::Comm& left_comm, RBC::Comm& right_comm) {
    if (!m_barriers)
      return getTime();
    RBC::Request req[2];
    RBC::Ibarrier(left_comm, &req[0]);
    RBC::Ibarrier(right_comm, &req[1]);
    RBC::Waitall(2, req, MPI_STATUS_IGNORE);
    return getTime();
  }

  double startTimeJanus_barrier(RBC::Comm& left_comm, RBC::Comm& right_comm) {
    RBC::Request req[2];
    RBC::Ibarrier(left_comm, &req[0]);
    RBC::Ibarrier(right_comm, &req[1]);
    RBC::Waitall(2, req, MPI_STATUS_IGNORE);
    return getTime();
  }

/*
 * Select an element from the interval as the pivot
 */
  template <class Compare>
  void getPivot(QSInterval_SQS<T> const& ival, T& pivot, int64_t& split_idx,
                Compare&& comp, bool& zero_global_elements) {
    return PivotSelection_SQS<T>::getPivot(ival, pivot, split_idx, std::forward<Compare>(comp),
                                           m_generator, m_sample_generator, zero_global_elements);
  }

/*
 * Select an element as the pivot from both intervals
 */
  template <class Compare>
  void getPivotJanus(QSInterval_SQS<T> const& ival_left,
                     QSInterval_SQS<T> const& ival_right, T& pivot_left,
                     T& pivot_right, int64_t& split_idx_left, int64_t& split_idx_right,
                     Compare&& comp) {
    PivotSelection_SQS<T>::getPivotJanus(ival_left, ival_right, pivot_left, pivot_right,
                                         split_idx_left, split_idx_right, std::forward<Compare>(comp),
                                         m_generator, m_sample_generator);
  }

/*
 * Partitions the data separatly for the elements with index smaller less_idx
 * and the elements with index larger less_idx
 * Returns the indexes of the first element of the right partitions
 * @param index1 First element of the first partition with large elements
 * @param index2 First element of the second partition with large elements
 */
  template <class Compare>
  void partitionData(QSInterval_SQS<T> const& ival, T pivot, int64_t less_idx,
                     int64_t* index1, int64_t* index2, Compare&& comp) {
    int64_t start1 = ival.m_local_start, end1 = less_idx,
      start2 = less_idx, end2 = ival.m_local_end;
    *index1 = partitionSequence(m_data->data(), pivot, start1, end1, true,
                                std::forward<Compare>(comp));
    *index2 = partitionSequence(m_data->data(), pivot, start2, end2, false,
                                std::forward<Compare>(comp));
  }

/**
 * Partition the data with index [start, end)
 * @param less_equal If true, compare to the pivot with <=, else compare with >
 * @return Index of the first large element
 */
  template <class Compare>
  int64_t partitionSequence(T* data_ptr, T pivot, int64_t start, int64_t end,
                            bool less_equal, Compare&& comp) {
    T* bound;
    if (less_equal) {
      bound = std::partition(data_ptr + start, data_ptr + end,
                             [pivot, &comp](T x) {
          return !comp(pivot, x)  /*x <= pivot*/;
        });
    } else {
      bound = std::partition(data_ptr + start, data_ptr + end,
                             [pivot, &comp](T x) {
          return comp(x, pivot);
        });
    }
    return bound - data_ptr;
  }

/*
 * Prefix sum of small/large elements and broadcast of global small/large elements
 */
  void calculateExchangeData(QSInterval_SQS<T>& ival, int64_t bound1,
                             int64_t split, int64_t bound2) {
    elementsCalculation(ival, bound1, split, bound2);
    int64_t in[2] = { ival.m_local_small_elements, ival.m_local_large_elements };
    int64_t presum[2], global[2];
    RBC::Request request;
    RBC::IscanAndBcast(&in[0], &presum[0], &global[0], 2, MPI_LONG_LONG,
                       MPI_SUM, ival.m_comm, &request, Constants::CALC_EXCH);
    RBC::Wait(&request, MPI_STATUS_IGNORE);

    assignPresum(ival, presum, global);

    if (!ival.m_evenly_distributed) {
      ival.m_split_size = ival.m_global_elements / ival.m_number_of_pes;
      ival.m_extra_elements = ival.m_global_elements % ival.m_number_of_pes;
      int64_t buf_size = std::max(ival.m_local_elements, ival.getLocalElements());
      m_buffer = new T[buf_size];
      ival.m_buffer = m_buffer;
    }
  }

  void elementsCalculation(QSInterval_SQS<T>& ival, int64_t bound1,
                           int64_t split, int64_t bound2) {
    ival.m_bound1 = bound1;
    ival.m_bound2 = bound2;
    ival.m_split = split;
    ival.m_local_small_elements = (bound1 - ival.m_local_start) + (bound2 - split);
    ival.m_local_large_elements = ival.m_local_elements - ival.m_local_small_elements;
  }

  void assignPresum(QSInterval_SQS<T>& ival,
                    int64_t presum[2], int64_t global[2]) {
    ival.m_presum_small = presum[0] - ival.m_local_small_elements;
    ival.m_presum_large = presum[1] - ival.m_local_large_elements;
    ival.m_global_small_elements = global[0];
    ival.m_global_large_elements = global[1];
    ival.m_global_elements = ival.m_global_small_elements + ival.m_global_large_elements;
  }

  void calculateExchangeDataJanus(QSInterval_SQS<T>& ival_left,
                                  QSInterval_SQS<T>& ival_right,
                                  int64_t bound1_left, int64_t split_left,
                                  int64_t bound2_left, int64_t bound1_right,
                                  int64_t split_right, int64_t bound2_right) {
    elementsCalculation(ival_left, bound1_left, split_left, bound2_left);
    elementsCalculation(ival_right, bound1_right, split_right, bound2_right);

    int64_t in_left[2] = { ival_left.m_local_small_elements, ival_left.m_local_large_elements };
    int64_t in_right[2] = { ival_right.m_local_small_elements, ival_right.m_local_large_elements };
    int64_t presum_left[2], presum_right[2], global_left[2], global_right[2];
    RBC::Request requests[2];
    RBC::IscanAndBcast(&in_left[0], &presum_left[0], &global_left[0], 2, MPI_LONG_LONG,
                       MPI_SUM, ival_left.m_comm, &requests[1], Constants::CALC_EXCH);
    RBC::IscanAndBcast(&in_right[0], &presum_right[0], &global_right[0], 2, MPI_LONG_LONG,
                       MPI_SUM, ival_right.m_comm, &requests[0], Constants::CALC_EXCH);
    RBC::Waitall(2, requests, MPI_STATUSES_IGNORE);

    assignPresum(ival_left, presum_left, global_left);
    assignPresum(ival_right, presum_right, global_right);
  }

/*
 * Exchange the data with other PEs
 */
  void exchangeData(QSInterval_SQS<T>& ival) {
    DataExchange_SQS<T>::exchangeData(ival);

    if (!ival.m_evenly_distributed) {
      ival.m_local_elements = ival.getLocalElements();
      ival.m_local_end = ival.m_local_start + ival.m_local_elements;
    }
  }

/*
 * Exchange the data with other PEs on both intervals simultaneously
 */
  void exchangeDataJanus(QSInterval_SQS<T>& left, QSInterval_SQS<T>& right) {
    DataExchange_SQS<T>::exchangeDataJanus(left, right);
  }

/*
 * Calculate how the PEs should be split into two groups
 */
  void calculateSplit(QSInterval_SQS<T>& ival, int& left_size, int64_t& offset,
                      bool& janus, int64_t& mid) {
    assert(ival.m_global_small_elements != 0);
    int64_t last_small_element = ival.m_missing_first_pe + ival.m_global_small_elements - 1;

    left_size = ival.getRankFromIndex(last_small_element) + 1;
    offset = ival.getOffsetFromIndex(last_small_element);

    if (offset + 1 == ival.getSplitSize(left_size - 1))
      janus = false;
    else
      janus = true;

    if (ival.m_rank < left_size - 1) {
      mid = ival.m_local_end;
    } else if (ival.m_rank > left_size - 1) {
      mid = ival.m_local_start;
    } else {
      mid = offset + 1;
    }
  }

/*
 * Splits the communicator into two new, left and right
 */
  void createNewCommunicators(QSInterval_SQS<T>& ival, int64_t left_size,
                              bool janus, RBC::Comm* left, RBC::Comm* right) {
    int size;
    RBC::Comm_size(ival.m_comm, &size);
    int left_end = left_size - 1;
    int right_start = left_size;
    if (janus)
      right_start--;
    int right_end = std::min(static_cast<int64_t>(size - 1), ival.m_global_elements - 1);
    right_end = std::max(right_start, right_end);
    RBC::Split_Comm(ival.m_comm, 0, left_end, right_start, right_end,
                    left, right);
    RBC::Comm_free(ival.m_comm);
  }

  void createNewCommunicatorsJanus(QSInterval_SQS<T>& ival_left,
                                   QSInterval_SQS<T>& ival_right, int64_t left_size_left,
                                   bool janus_left, int64_t left_size_right, bool janus_right,
                                   RBC::Comm* left_1, RBC::Comm* right_1,
                                   RBC::Comm* left_2, RBC::Comm* right_2) {
    if (ival_left.m_blocking_priority) {
      createNewCommunicators(ival_left, left_size_left, janus_left, left_1, right_1);
      createNewCommunicators(ival_right, left_size_right, janus_right, left_2, right_2);
    } else {
      createNewCommunicators(ival_right, left_size_right, janus_right, left_2, right_2);
      createNewCommunicators(ival_left, left_size_left, janus_left, left_1, right_1);
    }
  }

/*
 * Create QSIntervals for the next recursion level
 */
  void createIntervals(QSInterval_SQS<T>& ival, int64_t offset, int left_size,
                       bool janus,
                       int64_t mid, RBC::Comm& comm_left, RBC::Comm& comm_right,
                       QSInterval_SQS<T>& ival_left,
                       QSInterval_SQS<T>& ival_right) {
    int64_t missing_last_left, missing_first_right;
    if (janus) {
      missing_last_left = ival.getSplitSize(left_size - 1) - (offset + 1);
      missing_first_right = offset + 1;
    } else {
      missing_last_left = 0;
      missing_first_right = 0;
    }

    int64_t start = ival.m_local_start;
    int64_t end = ival.m_local_end;
    int64_t extra_elements_left, extra_elements_right,
      split_size_left, split_size_right;
    if (left_size <= ival.m_extra_elements) {
      extra_elements_left = 0;
      split_size_left = ival.m_split_size + 1;
      extra_elements_right = ival.m_extra_elements - left_size;
      if (janus)
        extra_elements_right++;
    } else {
      extra_elements_left = ival.m_extra_elements;
      split_size_left = ival.m_split_size;
      extra_elements_right = 0;
    }
    split_size_right = ival.m_split_size;

    ival_left = QSInterval_SQS<T>(ival.m_data, ival.m_buffer, split_size_left, extra_elements_left,
                                  start, mid, comm_left, ival.m_missing_first_pe, missing_last_left,
                                  m_mpi_type, ival.m_seed, ival.m_min_samples, ival.m_add_pivot, true);
    ival_right = QSInterval_SQS<T>(ival.m_data, ival.m_buffer, split_size_right, extra_elements_right,
                                   mid, end, comm_right, missing_first_right, ival.m_missing_last_pe,
                                   m_mpi_type, ival.m_seed + 1, ival.m_min_samples, ival.m_add_pivot, false);
  }

/*
 * Add an interval with two PEs (base case)
 */
  void addTwoPEInterval(QSInterval_SQS<T> const& ival) {
    m_two_pe_intervals.push_back(ival);
  }

/*
 * Add an interval that can be sorted locally (base case)
 */
  void addLocalInterval(QSInterval_SQS<T>& ival) {
    m_local_intervals.push_back(ival);
  }

/*
 * Sort the saved intervals with exactly two PEs
 */
  template <class Compare, class Sorter>
  void sortTwoPEIntervals(Compare&& comp, Sorter& sorter) {
    bc2_elements = SequentialSort_SQS<T>::sortTwoPEIntervals(std::forward<Compare>(comp), sorter,
                                                             m_two_pe_intervals);
    for (size_t i = 0; i < m_two_pe_intervals.size(); i++)
      RBC::Comm_free(m_two_pe_intervals[i].m_comm);
  }
/*
 * Sort all local intervals
 */
  template <class Sorter>
  void sortLocalIntervals(Sorter& sorter) {
    SequentialSort_SQS<T>::sortLocalIntervals(sorter,
                                              m_local_intervals);
    bc1_elements = 0.0;
    for (size_t i = 0; i < m_local_intervals.size(); i++) {
      bc1_elements += m_local_intervals[i].m_local_elements;
      RBC::Comm_free(m_local_intervals[i].m_comm);
    }
  }

  MPI_Datatype m_mpi_type;
  int m_depth = 0, m_seed;
  double t_pivot = 0.0, t_calculate = 0.0, t_exchange = 0.0, t_partition = 0.0,
    t_sort_two = 0.0, t_sort_local = 0.0, t_create_comms = 0.0, t_runtime,
    bc1_elements, bc2_elements;
  std::vector<double> t_vec_exchange, exchange_times { 0.0, 0.0, 0.0, 0.0 };
  T* m_buffer;
  std::vector<T>* m_data;
// m_generator is synchronized between processes.
// m_sample_generator is an asynchronized random generator.
  std::mt19937_64 m_generator, m_sample_generator;
  std::vector<QSInterval_SQS<T> > m_local_intervals, m_two_pe_intervals;
  bool m_barriers, m_split_mpi_comm, m_use_mpi_collectives, m_add_pivots;
  int64_t m_min_samples;
};
}  // namespace JanusSort
