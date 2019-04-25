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
#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "../../include/RBC/RBC.hpp"
#include "Constants.hpp"
#include "QSInterval.hpp"
#include "TbSplitter.hpp"

#define W(X) #X << "=" << X << ", "

namespace JanusSort {
template <typename T>
class PivotSelection_SQS {
 public:
  template <class Compare>
  static void getPivot(QSInterval_SQS<T> const& ival, T& pivot,
                       int64_t& split_idx, Compare&& comp, std::mt19937_64& generator,
                       std::mt19937_64& sample_generator, bool& zero_global_elements) {
    int64_t global_samples, local_samples;

    if (ival.m_evenly_distributed)
      getLocalSamples_calculate(ival, global_samples, local_samples,
                                generator);
    else
      getLocalSamples_communicate(ival, global_samples, local_samples,
                                  generator);

    if (global_samples == -1) {
      zero_global_elements = true;
      return;
    } else {
      zero_global_elements = false;
    }

//        std::cout << W(ival.m_rank) << W(local_samples) << W(global_samples) << std::endl;

    std::vector<TbSplitter<T> > samples;
    pickLocalSamples(ival, local_samples, samples, std::forward<Compare>(comp), sample_generator);
    auto tb_splitter_comp = [comp](const TbSplitter<T>& first,
                                   const TbSplitter<T>& second) {
                              return first.compare(second, comp);
                            };

    // Merge function used in the gather
    auto merger = [&tb_splitter_comp](void* begin1, void* end1, void* begin2, void* end2, void* result) {
                    std::merge(static_cast<TbSplitter<T>*>(begin1), static_cast<TbSplitter<T>*>(end1),
                               static_cast<TbSplitter<T>*>(begin2), static_cast<TbSplitter<T>*>(end2),
                               static_cast<TbSplitter<T>*>(result), tb_splitter_comp);
                  };

    // Gather the samples to rank 0
    TbSplitter<T>* all_samples = nullptr;
    if (ival.m_rank == 0)
      all_samples = new TbSplitter<T>[global_samples];

    MPI_Datatype splitter_type = TbSplitter<T>::mpiType(ival.m_mpi_type);

    RBC::Request req_gather;
    RBC::Igatherm(&samples[0], samples.size(), splitter_type, all_samples, global_samples,
                  0, merger, ival.m_comm, &req_gather, Constants::PIVOT_GATHER);
    RBC::Wait(&req_gather, MPI_STATUS_IGNORE);

    TbSplitter<T> splitter;
    if (ival.m_rank == 0)
      splitter = all_samples[global_samples / 2];

    // Broadcast the pivot from rank 0
    RBC::Request req_bcast;
    RBC::Ibcast(&splitter, 1, splitter_type, 0, ival.m_comm, &req_bcast,
                Constants::PIVOT_BCAST);
    RBC::Wait(&req_bcast, MPI_STATUS_IGNORE);

    pivot = splitter.Splitter();
    selectSplitter(ival, splitter, split_idx);

    //        std::cout << W(ival.m_rank) << W(pivot) << std::endl;
    if (ival.m_rank == 0)
      delete[] all_samples;
  }

/*
 * Select a random element as the pivot from both intervals
 */
  template <class Compare>
  static void getPivotJanus(QSInterval_SQS<T> const& ival_left,
                            QSInterval_SQS<T> const& ival_right, T& pivot_left,
                            T& pivot_right, int64_t& split_idx_left, int64_t& split_idx_right,
                            Compare&& comp, std::mt19937_64& generator, std::mt19937_64& sample_generator) {
    int64_t global_samples_left, global_samples_right,
      local_samples_left, local_samples_right;

    // Randomly pick samples from local data
    getLocalSamples_calculate(ival_left, global_samples_left,
                              local_samples_left, generator);
    getLocalSamples_calculate(ival_right, global_samples_right,
                              local_samples_right, generator);

    std::vector<TbSplitter<T> > samples_left, samples_right;
    pickLocalSamples(ival_left, local_samples_left, samples_left, std::forward<Compare>(comp),
                     sample_generator);
    pickLocalSamples(ival_right, local_samples_right, samples_right,
                     std::forward<Compare>(comp), sample_generator);

    // Merge function used in the gather
    auto tb_splitter_comp = [comp](const TbSplitter<T>& first,
                                   const TbSplitter<T>& second) {
                              return first.compare(second, comp);
                            };
    auto merger = [tb_splitter_comp](void* begin1, void* end1, void* begin2, void* end2, void* result) {
                    std::merge(static_cast<TbSplitter<T>*>(begin1), static_cast<TbSplitter<T>*>(end1),
                               static_cast<TbSplitter<T>*>(begin2), static_cast<TbSplitter<T>*>(end2),
                               static_cast<TbSplitter<T>*>(result), tb_splitter_comp);
                  };

    TbSplitter<T> splitter_left, splitter_right;
    MPI_Datatype splitter_type = TbSplitter<T>::mpiType(ival_left.m_mpi_type);

    // Gather the samples
    TbSplitter<T>* all_samples = new TbSplitter<T>[global_samples_right];
    RBC::Request req_gather[2];
    RBC::Igatherm(&samples_left[0], samples_left.size(), splitter_type, nullptr,
                  global_samples_left, 0, merger, ival_left.m_comm, &req_gather[0],
                  Constants::PIVOT_GATHER);
    RBC::Igatherm(&samples_right[0], samples_right.size(), splitter_type, all_samples,
                  global_samples_right, 0, merger, ival_right.m_comm, &req_gather[1],
                  Constants::PIVOT_GATHER);
    RBC::Waitall(2, req_gather, MPI_STATUSES_IGNORE);

    // Determine pivot on right interval
    splitter_right = all_samples[global_samples_right / 2];

    // Broadcast the pivots
    RBC::Request req_bcast[2];
    RBC::Ibcast(&splitter_left, 1, splitter_type, 0, ival_left.m_comm, &req_bcast[0],
                Constants::PIVOT_BCAST);
    RBC::Ibcast(&splitter_right, 1, splitter_type, 0, ival_right.m_comm, &req_bcast[1],
                Constants::PIVOT_BCAST);
    RBC::Waitall(2, req_bcast, MPI_STATUSES_IGNORE);

    pivot_left = splitter_left.Splitter();
    pivot_right = splitter_right.Splitter();

    selectSplitter(ival_left, splitter_left, split_idx_left);
    selectSplitter(ival_right, splitter_right, split_idx_right);

    delete[] all_samples;
  }

 private:
/*
 * Determine how many samples need to be send and pick them randomly.
 * The input elements have to be evenly distributed, meaning each PE has
 * global_count / size elements and the remaining x = global_count % size elements
 * are distributed to the PEs [0, x-1].
 */
  static void getLocalSamples_calculate(QSInterval_SQS<T> const& ival,
                                        int64_t& total_samples,
                                        int64_t& local_samples,
                                        std::mt19937_64& generator) {
    total_samples = getSampleCount(ival.m_number_of_pes,
                                   ival.getIndexFromRank(ival.m_number_of_pes), ival.m_min_samples,
                                   ival.m_add_pivot);
    int max_height = std::ceil(std::log2(ival.m_number_of_pes));
    int own_height = 0;
    for (int i = 0; ((ival.m_rank >> i) % 2 == 0) && (i < max_height); i++)
      own_height++;

    int first_pe = 0;
    int last_pe = ival.m_number_of_pes - 1;
    local_samples = total_samples;
    generator.seed(ival.m_seed);

    for (int height = max_height; height > 0; height--) {
      if (first_pe + std::pow(2, height - 1) > last_pe) {
        // right subtree is empty
      } else {
        int left_size = std::pow(2, height - 1);
        assert(left_size > 0);
        int64_t left_elements = ival.getIndexFromRank(first_pe + left_size)
                                - ival.getIndexFromRank(first_pe);
        int64_t right_elements = ival.getIndexFromRank(last_pe + 1)
                                 - ival.getIndexFromRank(first_pe + left_size);
        if (first_pe == 0)
          left_elements -= ival.m_missing_first_pe;
        if (last_pe == ival.m_number_of_pes - 1)
          right_elements -= ival.m_missing_last_pe;

        assert(left_elements > 0);
        assert(right_elements >= 0);
        double percentage_left = static_cast<double>(left_elements)
                                 / static_cast<double>(left_elements + right_elements);
        assert(percentage_left > 0);

        std::binomial_distribution<int64_t> binom_distr(local_samples, percentage_left);
        int64_t samples_left = binom_distr(generator);
        int64_t samples_right = local_samples - samples_left;

        int mid_pe = first_pe + std::pow(2, height - 1);
        if (ival.m_rank < mid_pe) {
          // left side
          last_pe = mid_pe - 1;
          local_samples = samples_left;
        } else {
          // right side
          first_pe = mid_pe;
          local_samples = samples_right;
        }
      }
    }
  }

/*
 * Determine how much samples need to be send and pick them randomly
 */
  static void getLocalSamples_communicate(QSInterval_SQS<T> const& ival,
                                          int64_t& total_samples, int64_t& local_samples,
                                          std::mt19937_64& generator) {
    DistrToLocSampleCount(ival.m_local_elements, total_samples,
                          local_samples, generator, ival.m_comm, ival.m_min_samples, ival.m_add_pivot);
  }

/*
 * Returns the number of global and local samples
 */
  static void DistrToLocSampleCount(int64_t const local_elements,
                                    int64_t& global_samples, int64_t& local_samples,
                                    std::mt19937_64& async_gen, RBC::Comm const comm,
                                    int64_t min_samples, bool add_pivot) {
    int comm_size, rank;
    RBC::Comm_size(comm, &comm_size);
    RBC::Comm_rank(comm, &rank);

    // Calculate height in tree
    int logp = std::ceil(std::log2(comm_size));
    int height = 0;
    while ((rank >> height) % 2 == 0 && height < logp)
      height++;

    MPI_Status status;
    const int tag = Constants::DISTR_SAMPLE_COUNT;
    int64_t tree_elements = local_elements;
    std::vector<int64_t> load_l, load_r;

    // Gather element count
    for (int k = 0; k < height; k++) {
      int src_rank = rank + (1 << k);
      if (src_rank < comm_size) {
        int64_t right_subtree;
        RBC::Recv(&right_subtree, 1, MPI_LONG_LONG, src_rank,
                  tag, comm, &status);

        load_r.push_back(right_subtree);
        load_l.push_back(tree_elements);
        tree_elements += right_subtree;
      }
    }
    assert(tree_elements >= 0);
    if (rank > 0) {
      int target_id = rank - (1 << height);
      RBC::Send(&tree_elements, 1, MPI_LONG_LONG, target_id, tag, comm);
    }

    // Distribute samples
    int64_t tree_sample_cnt;
    if (rank == 0) {
      if (tree_elements == 0)
        tree_sample_cnt = -1;
      else
        tree_sample_cnt = getSampleCount(comm_size, tree_elements,
                                         min_samples, add_pivot);
      global_samples = tree_sample_cnt;
    } else {
      int src_id = rank - (1 << height);
      int64_t recvbuf[2];
      RBC::Recv(recvbuf, 2, MPI_LONG_LONG, src_id, tag, comm, &status);
      tree_sample_cnt = recvbuf[0];
      global_samples = recvbuf[1];
    }

    for (int kr = height; kr > 0; kr--) {
      int k = kr - 1;
      int target_rank = rank + (1 << k);
      if (target_rank < comm_size) {
        int64_t right_subtree_sample_cnt;

        if (tree_sample_cnt < 0) {
          // There are no global elements at all.
          right_subtree_sample_cnt = -1;
        } else if (tree_sample_cnt == 0) {
          right_subtree_sample_cnt = 0;
        } else if (load_r.back() == 0) {
          right_subtree_sample_cnt = 0;
        } else if (load_l.back() == 0) {
          right_subtree_sample_cnt = tree_sample_cnt;
          tree_sample_cnt -= right_subtree_sample_cnt;
        } else {
          double right_p = load_r.back() / (static_cast<double>(load_l.back())
                                            + static_cast<double>(load_r.back()));
          std::binomial_distribution<int64_t> distr(tree_sample_cnt, right_p);
          right_subtree_sample_cnt = distr(async_gen);
          tree_sample_cnt -= right_subtree_sample_cnt;
        }

        int64_t sendbuf[2] = { right_subtree_sample_cnt, global_samples };
        RBC::Send(sendbuf, 2, MPI_LONG_LONG, target_rank, tag, comm);

        load_l.pop_back();
        load_r.pop_back();
      }
    }
    local_samples = tree_sample_cnt;
  }

/*
 * Get the number of samples
 */
  static int64_t getSampleCount(int comm_size, int64_t global_elements,
                                int64_t min_samples, bool add_pivot) {
    if (global_elements == 0)
      return -1;

    int64_t k_1 = 16, k_2 = 50, k_3 = min_samples;  // tuning parameters
    int64_t count_1 = k_1 * std::log2(comm_size);
    int64_t count_2 = (global_elements / comm_size) / k_2;
    int64_t sample_count = std::max(count_1, std::max(count_2, k_3));
    if (add_pivot)
      sample_count = std::max(count_1 + count_2, k_3);
    if (sample_count % 2 == 0)
      sample_count++;

    return sample_count;
  }

/*
 * Pick samples randomly from local data
 */
  template <class Compare>
  static void pickLocalSamples(QSInterval_SQS<T> const& ival, int64_t sample_count,
                               std::vector<TbSplitter<T> >& sample_vec, Compare&& comp,
                               std::mt19937_64& generator) {
    T* data = ival.m_data->data();
    std::uniform_int_distribution<int64_t> distr(ival.m_local_start, ival.m_local_end - 1);
    for (int64_t i = 0; i < sample_count; i++) {
      int64_t index = distr(generator);
      int64_t global_index;
      if (ival.m_evenly_distributed)
        global_index = ival.getIndexFromRank(ival.m_rank) + index;
      else
        global_index = ival.m_rank;
      sample_vec.push_back(TbSplitter<T>(data[index], global_index));
    }

    auto tb_splitter_comp = [comp](const TbSplitter<T>& first,
                                   const TbSplitter<T>& second) {
                              return first.compare(second, comp);
                            };

    std::sort(sample_vec.begin(), sample_vec.end(), tb_splitter_comp);
  }

/*
 * Calculate the local splitter index
 */
  static void selectSplitter(QSInterval_SQS<T> const& ival,
                             TbSplitter<T>& splitter, int64_t& split_idx) {
    if (!ival.m_evenly_distributed) {
      if (ival.m_rank <= splitter.GID())
        split_idx = ival.m_local_end;
      else
        split_idx = ival.m_local_start;
      return;
    }

    int64_t splitter_pe;
    splitter_pe = ival.getRankFromIndex(splitter.GID());
    if (ival.m_rank < splitter_pe) {
      split_idx = ival.m_local_end;
    } else if (ival.m_rank > splitter_pe) {
      split_idx = ival.m_local_start;
    } else {
      split_idx = ival.getOffsetFromIndex(splitter.GID()) + 1;
    }
  }
};
}  // namespace JanusSort
