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
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <ips2pa.hpp>
#include <ips4o.hpp>
#include <RBC.hpp>
#include <tlx/algorithm.hpp>
#include <tlx/math.hpp>

#include "../Tools/Common.hpp"
#include "../Tools/CommonMpi.hpp"
#include "Alltoall.hpp"
#include "AmsData.hpp"
#include "DummyTracker.hpp"
#include "GroupMsgToPeAssignment.hpp"
#include "LocalSampleCount.hpp"
#include "LocalSampling.hpp"
#include "Overpartition/Overpartition.hpp"
#include "SplitterSelection.hpp"

namespace Ams {
namespace _internal {
bool testAllElementsAssigned(std::vector<size_t> loc_group_el_cnts,
                             size_t size) {
  const size_t assigned_el_cnt = std::accumulate(loc_group_el_cnts.begin(),
                                                 loc_group_el_cnts.end(),
                                                 0,
                                                 std::plus<size_t>());
  bool succ = assigned_el_cnt == size;

  if (!succ) {
    std::string warning =
      "Warning: We expected " + std::to_string(size) +
      " elements but we assigned " + std::to_string(assigned_el_cnt) + " elements: ";
    for (const auto& loc_group_el_cnt : loc_group_el_cnts) {
      warning += std::to_string(loc_group_el_cnt) + "\t";
    }
  }

  return succ;
}

bool verifySendDescription(size_t size,
                           const DistrRanges& distr_ranges) {
  std::vector<bool> bitset(size, false);

  bool succ = true;

  for (const auto& distr_range : distr_ranges) {
    for (size_t idx = distr_range.offset; idx != distr_range.offset + distr_range.size; ++idx) {
      if (bitset[idx] == true) {
        succ = false;
      } else {
        bitset[idx] = true;
      }
    }
  }

  for (size_t idx = 0; idx != size; ++idx) {
    if (bitset[idx] == false) {
      succ = false;
    }
  }

  return succ;
}

template <class T>
bool verifyNoDataLoss(std::vector<T> data, std::vector<T> tmp_data,
                      MPI_Datatype mpi_type, RBC::Comm comm) {
  size_t loc_data_size = data.size();
  size_t loc_tmp_data_size = tmp_data.size();
  size_t glob_data_size = 0, glob_tmp_data_size = 0;
  RBC::Allreduce(&loc_data_size,
                 &glob_data_size,
                 1,
                 Common::getMpiType(loc_data_size),
                 MPI_SUM,
                 comm);
  RBC::Allreduce(&loc_tmp_data_size,
                 &glob_tmp_data_size,
                 1,
                 Common::getMpiType(loc_tmp_data_size),
                 MPI_SUM,
                 comm);

  if (glob_data_size != glob_tmp_data_size) {
    RBC::Barrier(comm);
    return false;
  }

  std::vector<T> glob_data(glob_data_size);
  std::vector<T> glob_tmp_data(glob_tmp_data_size);

  std::sort(data.begin(), data.end());
  std::sort(tmp_data.begin(), tmp_data.end());

  auto merger = [](void* begin1, void* end1, void* begin2, void* end2, void* result) {
                  std::merge(static_cast<T*>(begin1), static_cast<T*>(end1),
                             static_cast<T*>(begin2),
                             static_cast<T*>(end2), static_cast<T*>(result));
                };

  RBC::Allgatherm(data.data(), data.size(), mpi_type, glob_data.data(), glob_data.size(),
                  merger, comm);

  RBC::Allgatherm(tmp_data.data(), tmp_data.size(), mpi_type, glob_tmp_data.data(),
                  glob_tmp_data.size(),
                  merger, comm);

  bool succ = true;

  for (size_t idx = 0; idx != glob_data.size(); ++idx) {
    if (glob_data[idx] != glob_tmp_data[idx]) {
      succ = false;
    }
  }

  RBC::Barrier(comm);

  return succ;
}

class RecDescrKway : public LevelDescrInterface {
 public:
  RecDescrKway() = delete;
  RecDescrKway(size_t k, const RBC::Comm& comm) :
    init_comm_(comm) {
    assert(k > 1 || comm.getSize() == 1);

    num_levels_ = 0;

    // Calculate communicators.
    auto* group_comm = &comm;
    while (group_comm->getSize() > 1) {
      ++num_levels_;
      auto group_info = LevelDescrInterface::calculateLevel(k, *group_comm);
      comms_.push_back(std::move(std::get<2>(group_info)));
      group_sizes_.push_back(std::move(std::get<0>(group_info)));
      my_group_indices_.push_back(std::get<1>(group_info));

      group_comm = &(comms_.back());
      my_group_ranks_.push_back(group_comm->getRank());
    }

    max_num_levels_ = 0;
    size_t size = comm.getSize();
    while (size > 1) {
      size = maxSubgroupSize(size, k);
      ++max_num_levels_;
    }
  }

  // Number of levels executed by this process.
  size_t myNumLevels() const final {
    return num_levels_;
  }

  // Maximum number of levels executed by any process.
  size_t maxNumLevels() const final {
    return max_num_levels_;
  }

  // Total number of processes.
  RBC::Comm & initComm() {
    return init_comm_;
  }

  /* @brief Communicator of the process sub-group on level 'level'.
   *
   * The group communicator on level 'i' is the communicator which
   * contains the processes of level 'i + 1'. The group communicator
   * on level 'GetNumlevels() - 1' contains just this process and is
   * the last group communicator.
   *
   * @param level 0 <= level < myNumLevels()
   */
  RBC::Comm & groupComm(size_t level) final {
    return comms_[level];
  }

  // Number of processes of each subgroup at level 'level'.
  const std::vector<size_t> & groupSizes(size_t level) const final {
    return group_sizes_[level];
  }

  // Index of the process group at level 'level' to which this process belongs.
  size_t myGroupIdx(size_t level) const final {
    return my_group_indices_[level];
  }

  // Rank of this process in its process group at level 'level'.
  size_t myGroupRank(size_t level) const final {
    return my_group_ranks_[level];
  }

 private:
  size_t num_levels_;
  size_t max_num_levels_;
  RBC::Comm init_comm_;
  std::vector<RBC::Comm> comms_;
  std::vector<std::vector<size_t> > group_sizes_;
  std::vector<size_t> my_group_indices_;
  std::vector<size_t> my_group_ranks_;
};

class RecDescrKways : public LevelDescrInterface {
 public:
  RecDescrKways() = delete;
  RecDescrKways(const RBC::Comm& comm, const std::vector<size_t>& ks) :
    init_comm_(comm) {
    assert(comm.getSize() <= std::accumulate(ks.begin(), ks.end(), 1, std::multiplies<size_t>()));

    num_levels_ = 0;

    auto* group_comm = &comm;
    while (group_comm->getSize() > 1) {
      auto group_info = LevelDescrInterface::calculateLevel(ks[num_levels_], *group_comm);
      comms_.push_back(std::move(std::get<2>(group_info)));
      group_sizes_.push_back(std::move(std::get<0>(group_info)));
      my_group_indices_.push_back(std::get<1>(group_info));

      group_comm = &(comms_.back());
      my_group_ranks_.push_back(group_comm->getRank());
      ++num_levels_;
    }

    max_num_levels_ = 0;
    size_t size = comm.getSize();
    while (size > 1) {
      size = maxSubgroupSize(size, ks[max_num_levels_]);
      ++max_num_levels_;
    }
  }

  // Number of levels executed by this process.
  size_t myNumLevels() const final {
    return num_levels_;
  }

  // Maximum number of levels executed by any process.
  size_t maxNumLevels() const final {
    return max_num_levels_;
  }

  // Total number of processes.
  RBC::Comm & initComm() {
    return init_comm_;
  }

  /* @brief Communicator of the process sub-group on level 'level'.
   *
   * The group communicator on level 'i' is the communicator which
   * contains the processes of level 'i + 1'. The group communicator
   * on level 'GetNumlevels() - 1' contains just this process and is
   * the last group communicator.
   *
   * @param level 0 <= level < numLevels()
   */
  RBC::Comm & groupComm(size_t level) final {
    return comms_[level];
  }

  // Sizes of the process groups at level 'level'
  const std::vector<size_t> & groupSizes(size_t level) const final {
    return group_sizes_[level];
  }

  // Index of the process group at level 'level' to which this process belongs.
  size_t myGroupIdx(size_t level) const final {
    return my_group_indices_[level];
  }

  // Rank of this process in its process group at level 'level'.
  size_t myGroupRank(size_t level) const final {
    return my_group_ranks_[level];
  }

 private:
  size_t num_levels_;
  size_t max_num_levels_;
  RBC::Comm init_comm_;
  std::vector<RBC::Comm> comms_;
  std::vector<std::vector<size_t> > group_sizes_;
  std::vector<size_t> my_group_indices_;
  std::vector<size_t> my_group_ranks_;
};

size_t totalNumElements(size_t loc_el_cnt, const RBC::Comm& comm) {
  size_t glob_el_cnt = 0;
  RBC::Allreduce(&loc_el_cnt, &glob_el_cnt, 1, Common::getMpiType(loc_el_cnt), MPI_SUM, comm);
  return glob_el_cnt;
}

/* @brief Removes duplicates from a sorted vector.
 *
 * Removes duplicates from a sorted vector using the function comp
 * and resizes the vector afterwards.
 *
 * @param comp Must implement a less function. Less or equal is not allowed.
 */
template <class Container, class Comp>
void removeDuplicates(Container& sorted_container, const Comp comp) {
  assert(std::is_sorted(sorted_container.begin(), sorted_container.end(), comp));

  const auto is_equal = [&comp](const auto& left, const auto& right) {
                          return !comp(left, right);
                        };

  const auto it = std::unique(sorted_container.begin(), sorted_container.end(), is_equal);
  sorted_container.resize(std::distance(sorted_container.begin(), it));
}

// Returns number of elements in each partition.
template <class AmsData>
std::vector<size_t> partitionInplaceWithEqualBuckets(AmsData& ams_data,
                                                     size_t glob_sample_cnt,
                                                     size_t glob_splitter_cnt,
                                                     bool* use_equal_buckets) {
  using Tags = typename AmsData::Tags;
  
  ams_data.config.tracker.sampling_t.start(ams_data.comm());

  std::vector<typename AmsData::T> samples;
  if (glob_sample_cnt > ams_data.n_act) {
    samples = ams_data.data;
    glob_sample_cnt = ams_data.n_act;
    glob_splitter_cnt = std::min(ams_data.n_act, glob_splitter_cnt);
  } else if (ams_data.level() == 0) {
    // Standard random sampling.

    const size_t loc_sample_cnt = locSampleCountWithoutSameInputSize<Tags>(
      ams_data.data.size(), glob_sample_cnt, ams_data.async_gen, ams_data.comm());

    samples = sampleUniform(ams_data.data, loc_sample_cnt, ams_data.async_gen);
  } else {
    // We assume that we have 'ams_data.residual'
    // elements. 'ams_data.residual' is the maximum number of
    // elements which a process got assigned in the last message
    // assignment. Note that we sample from the range [0,
    // ams_data.residual - 1]. When we sample an element in
    // ams_data.data which does not exist (out of bounds), we do
    // not pick a sample. This means that we might have less
    // samples. However, we still select our splitters with the
    // initial oversampling ration. This results in correct
    // sampling as we threat skipped samples as elements with key
    // 'infinity'. We have to do it this way to avoid that a
    // process with many elements does not sample too many
    // elements (e.g., if the last process group gets just few
    // elements but is executed with a large residual).

    const size_t loc_sample_cnt = locSampleCountWithSameInputSize(glob_sample_cnt,
                                                                  ams_data.sync_gen,
                                                                  ams_data.comm());

    samples = sampleUniform(ams_data.data, ams_data.residual,
                            loc_sample_cnt, ams_data.async_gen);
  }


  auto splitters = Ams::SplitterSelection::SplitterSelection(ams_data,
                                                             ams_data.mpi_type,
                                                             samples,
                                                             ams_data.comp,
                                                             glob_sample_cnt,
                                                             glob_splitter_cnt,
                                                             ams_data.config.use_two_tree);

  ams_data.config.tracker.sampling_t.stop();

  if (splitters.empty()) {
    *use_equal_buckets = false;
    return { ams_data.data.size() };
  }

  ams_data.config.tracker.partition_t.start(ams_data.comm());

  // Partitioning.
  removeDuplicates(splitters, ams_data.comp);

  *use_equal_buckets = true;

  const size_t bucket_cnt = 2 * splitters.size() + 1;
  std::vector<size_t> bucket_sizes(bucket_cnt);
  ips2pa::partition(ams_data.data.begin(), ams_data.data.end(), splitters.begin(),
                    splitters.end(), bucket_sizes.data(),
                    *use_equal_buckets, ams_data.comp, ams_data.async_gen());

  ams_data.config.tracker.partition_t.stop();

  return bucket_sizes;
}

// Returns number of elements in each partition.
template <class AmsData>
std::vector<size_t> partitionInplaceWithoutEqualBuckets(AmsData& ams_data,
                                                        size_t glob_sample_cnt, size_t
                                                        glob_splitter_cnt,
                                                        bool* use_equal_buckets) {
  using Tags = typename AmsData::Tags;
  
  ams_data.config.tracker.sampling_t.start(ams_data.comm());

  std::vector<typename AmsData::T> samples;

  if (glob_sample_cnt > ams_data.n_act) {
    samples = ams_data.data;
    glob_sample_cnt = ams_data.n_act;
    glob_splitter_cnt = std::min(ams_data.n_act, glob_splitter_cnt);
  } else if (ams_data.level() == 0) {
    // Standard random sampling.

    const size_t loc_sample_cnt = locSampleCountWithoutSameInputSize<Tags>(
      ams_data.data.size(), glob_sample_cnt, ams_data.async_gen, ams_data.comm());

    samples = sampleUniform(ams_data.data, loc_sample_cnt, ams_data.async_gen);
  } else {
    // We assume that we have 'ams_data.residual'
    // elements. 'ams_data.residual' is the maximum number of
    // elements which a process got assigned in the last message
    // assignment. Note that we sample from the range [0,
    // ams_data.residual - 1]. When we sample an element in
    // ams_data.data which does not exist (out of bounds), we do
    // not pick a sample. This means that we might have less
    // samples. However, we still select our splitters with the
    // initial oversampling ration. This results in correct
    // sampling as we threat skipped samples as elements with key
    // 'infinity'. We have to do it this way to avoid that a
    // process with many elements does not sample too many
    // elements (e.g., if the last process group gets just few
    // elements but is executed with a large residual).

    const size_t loc_sample_cnt = locSampleCountWithSameInputSize(glob_sample_cnt,
                                                                  ams_data.sync_gen,
                                                                  ams_data.comm());

    samples = sampleUniform(ams_data.data, ams_data.residual,
                            loc_sample_cnt, ams_data.async_gen);
  }

  auto splitters = Ams::SplitterSelection::SplitterSelection(ams_data,
                                                             ams_data.mpi_type,
                                                             samples,
                                                             ams_data.comp,
                                                             glob_sample_cnt,
                                                             glob_splitter_cnt,
                                                             ams_data.config.use_two_tree);

  ams_data.config.tracker.sampling_t.stop();

  if (splitters.empty()) {
    *use_equal_buckets = false;
    return { ams_data.data.size() };
  }

  ams_data.config.tracker.partition_t.start(ams_data.comm());

  // Partitioning.
  removeDuplicates(splitters, ams_data.comp);

  // Partitioning.

  *use_equal_buckets = false;

  const size_t bucket_cnt = splitters.size() + 1;
  std::vector<size_t> bucket_sizes(bucket_cnt);
  ips2pa::partition(ams_data.data.begin(), ams_data.data.end(), splitters.begin(),
                    splitters.end(), bucket_sizes.data(),
                    *use_equal_buckets, ams_data.comp, ams_data.async_gen());

  ams_data.config.tracker.partition_t.stop();

  return bucket_sizes;
}


template <class AmsData>
std::vector<size_t> partition(AmsData& ams_data,
                              const size_t glob_sample_cnt, const size_t glob_splitter_cnt,
                              bool* use_equal_buckets) {
  if (ams_data.config.part_strategy ==
      PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING) {
    return partitionInplaceWithEqualBuckets(ams_data,
                                            glob_sample_cnt, glob_splitter_cnt,
                                            use_equal_buckets);
  } else {
    return partitionInplaceWithoutEqualBuckets(ams_data,
                                               glob_sample_cnt, glob_splitter_cnt,
                                               use_equal_buckets);
  }
}

template <class PairType>
void mpiMaxSum(PairType* in,
               PairType* inout,
               int* len, MPI_Datatype*  /*type*/) {
  for (int i = 0; i != *len; ++i) {
    inout[i].first += in[i].first;
    inout[i].second = std::max(inout[i].second, in[i].second);
  }
}


double calcLevelEpsilon(size_t level, size_t total_level,
                        size_t np_init, size_t np_act, double init_eps) {
  // Average epsilon per level to guarantee final imbalance of
  // 1 + init_eps.
  double level_eps = init_eps + 1.;
  level_eps = pow(level_eps, 1. / static_cast<double>(total_level));
  level_eps = level_eps - 1.;

  // Maximum epsilon after this level.
  const double max_level_eps = pow(level_eps + 1., level + 1) - 1.;

  // Current epsilon
  const double act_eps = static_cast<double>(np_act) / np_init - 1.;

  const double eps = std::max(level_eps, max_level_eps - act_eps);

  return eps;
}

std::pair<size_t, size_t> calcNumSplittersSamples(size_t init_nprocs, size_t k, double eps) {
  assert(init_nprocs > 0);

  // Overpartitioning ratio (variable b in paper)

  assert(eps > 0);
  const size_t op = static_cast<size_t>(ceil(2. / eps));
  assert(op >= 1);


  // Oversampling ratio (variable a in paper)

  const double log_init_nprocs = tlx::integer_log2_ceil(init_nprocs);
  const size_t os = tlx::div_ceil(13 * log_init_nprocs, op);
  assert(os >= 1);

  // Calculate number of splitters and samples.

  const size_t num_splitters = op * k - 1;
  const size_t num_samples = os * k * op;

  return { num_splitters, num_samples };
}

template <class AmsData>
void exchangeKway(AmsData& ams_data, const DistrRanges& distr_ranges, size_t max_num_recv_msgs,
                  size_t max_recv_els) {
  using Tags = typename AmsData::Tags;
  
  // Sorted by target processes.
  assert(std::is_sorted(distr_ranges.begin(), distr_ranges.end(),
                        [](const auto& left, const auto& right) {
        return left.pe < right.pe;
      }));

  // No pieces of size zero.
  assert(std::find_if(distr_ranges.begin(), distr_ranges.end(),
                      [](const auto& piece) {
        return piece.size == 0;
      }) == distr_ranges.end());

  // At most one message is send to a target process.
  assert(std::adjacent_find(distr_ranges.begin(), distr_ranges.end(),
                            [](const auto& left, const auto& right) {
        return left.pe == right.pe;
      }) == distr_ranges.end());

  if (ams_data.comm().useMPICollectives()) {
    std::vector<int> send_counts(ams_data.comm().getSize(), 0);
    for (const auto& distr_range : distr_ranges) {
      send_counts[distr_range.pe] = distr_range.size;
    }

    Alltoallv::MPIAlltoallv(ams_data.config.tracker,
                            ams_data.data,
                            send_counts,
                            ams_data.tmp_data,
                            ams_data.mpi_type,
                            ams_data.comm());
  } else if (ams_data.config.distr_strategy == DistributionStrategy::EXCHANGE_WITHOUT_RECV_SIZES) {
    Alltoallv::exchangeWithoutRecvSizes<Tags>(ams_data.config.tracker, ams_data.data, distr_ranges,
                                        ams_data.tmp_data, max_num_recv_msgs, max_recv_els,
                                        ams_data.mpi_type, ams_data.comm());
  } else if (ams_data.config.distr_strategy == DistributionStrategy::EXCHANGE_WITH_RECV_SIZES) {
    Alltoallv::exchangeWithRecvSizes<Tags>(ams_data.config.tracker, ams_data.data,
                                     distr_ranges, ams_data.tmp_data,
                                     ams_data.mpi_type, ams_data.comm());
  } else {
    Alltoallv::exchangeWithRecvSizesAndPorts<Tags>(ams_data.config.tracker, ams_data.data,
                                             distr_ranges, ams_data.tmp_data,
                                             ams_data.mpi_type, ams_data.comm());
  }
}

template <class AmsData>
void exchangePway(AmsData& ams_data, const std::vector<size_t>& loc_group_el_cnts,
                  const std::vector<size_t>& glob_group_el_cnts) {
  using Tags = typename AmsData::Tags;
  if (ams_data.comm().useMPICollectives()) {
    // todo convert loc group_el_cnts to int?
    std::vector<int> send_counts(loc_group_el_cnts.begin(), loc_group_el_cnts.end());
    Alltoallv::MPIAlltoallv(ams_data.config.tracker,
                            ams_data.data,
                            send_counts,
                            ams_data.tmp_data,
                            ams_data.mpi_type,
                            ams_data.comm());
  } else {
    DistrRanges distr_ranges;
    distr_ranges.reserve(ams_data.nprocs);
    size_t offset = 0;

    for (size_t pe = 0; pe != ams_data.nprocs; ++pe) {
      const auto num_send_els = loc_group_el_cnts[pe];
      if (num_send_els) {
        distr_ranges.emplace_back(pe, offset, num_send_els);
        offset += num_send_els;
      }
    }

    assert(testAllElementsAssigned(loc_group_el_cnts, ams_data.data.size())
           & verifySendDescription(ams_data.data.size(), distr_ranges));

    if (ams_data.config.distr_strategy == DistributionStrategy::EXCHANGE_WITHOUT_RECV_SIZES) {
      Alltoallv::exchangeWithoutRecvSizes<Tags>(ams_data.config.tracker, ams_data.data, distr_ranges,
                                          ams_data.tmp_data,
                                          std::min(glob_group_el_cnts[ams_data.myrank],
                                                   ams_data.nprocs),
                                          glob_group_el_cnts[ams_data.myrank],
                                          ams_data.mpi_type, ams_data.comm());
    } else if (ams_data.config.distr_strategy == DistributionStrategy::EXCHANGE_WITH_RECV_SIZES) {
      Alltoallv::exchangeWithRecvSizes<Tags>(ams_data.config.tracker, ams_data.data,
                                       distr_ranges, ams_data.tmp_data,
                                       ams_data.mpi_type, ams_data.comm());
    } else {
      Alltoallv::exchangeWithRecvSizesAndPorts<Tags>(ams_data.config.tracker, ams_data.data,
                                               distr_ranges, ams_data.tmp_data,
                                               ams_data.mpi_type, ams_data.comm());
    }

    assert(ams_data.tmp_data.size() == glob_group_el_cnts[ams_data.myrank]);
  }

  assert(verifyNoDataLoss(ams_data.data, ams_data.tmp_data, ams_data.mpi_type,
                          ams_data.comm()));
}

template <class AmsData>
void recSort(AmsData& ams_data) {
  {    // Guarantees that temp objects have been destroyed.
    if (ams_data.n_act == 0) {
      return;
    }

    assert(ams_data.n_act == totalNumElements(ams_data.data.size(), ams_data.comm()));

    ams_data.config.tracker.various_t.start(ams_data.comm());

    const std::vector<size_t>& group_sizes = ams_data.groupSizes();
    const std::vector<size_t> group_sizes_exscan = [&]() {
                                                     std::vector<size_t> group_sizes_exscan(
                                                       group_sizes.size() + 1);
                                                     tlx::exclusive_scan(group_sizes.begin(),
                                                                         group_sizes.end(),
                                                                         group_sizes_exscan.begin(),
                                                                         0);
                                                     return group_sizes_exscan;
                                                   } ();
    const RBC::Comm& group_comm = ams_data.groupComm();
    const size_t my_group_idx = ams_data.myGroupIdx();
    const size_t my_group_rank = ams_data.myGroupRank();
    const size_t num_groups = group_sizes.size();

    const double level_eps = calcLevelEpsilon(ams_data.level(),
                                              ams_data.level_descrs->maxNumLevels(),
                                              ams_data.np_ceiled_init,
                                              ams_data.np_ceiled_act, ams_data.config.max_epsilon);

    const auto[num_splitters, num_samples] = calcNumSplittersSamples(ams_data.init_nprocs,
                                                                     num_groups, level_eps);

    ams_data.config.tracker.various_t.stop();

    ams_data.config.tracker.splitters_c_.add(num_splitters);
    ams_data.config.tracker.samples_c_.add(num_samples);

    bool use_equal_buckets = true;

    const std::vector<size_t> loc_bucket_sizes = partition(ams_data, num_samples,
                                                           num_splitters, &use_equal_buckets);
    assert(std::accumulate(loc_bucket_sizes.begin(), loc_bucket_sizes.end(), 0ul) ==
           ams_data.data.size());

    ams_data.config.tracker.splitter_allgather_scan_t.start(ams_data.comm());

    std::vector<size_t> glob_bucket_sizes(loc_bucket_sizes.size());
    std::vector<size_t> loc_bucket_sizes_exscan(loc_bucket_sizes.size());
    if (ams_data.config.use_two_tree) {
      // Create a temporary communicator which ensures that the
      // twotree implementation is used even when RBC collectives are
      // disabled in the RBC communicator 'comm'.
      RBC::Comm rbc_comm, rbc_subcomm;
      // Splitting disabled, RBC collectives enabled
      RBC::Create_Comm_from_MPI(ams_data.comm().get(), &rbc_comm, true, false, false);
      RBC::Comm_create_group(rbc_comm, &rbc_subcomm, ams_data.comm().getMpiFirst(),
                             ams_data.comm().getMpiLast(), ams_data.comm().getStride());

      RBC::_internal::optimized::ScanAndBcastTwotree(loc_bucket_sizes.data(),
                                                     loc_bucket_sizes_exscan.data(),
                                                     glob_bucket_sizes.data(),
                                                     loc_bucket_sizes.size(),
                                                     Common::getMpiType(loc_bucket_sizes),
                                                     MPI_SUM,
                                                     rbc_subcomm);
    } else {
      RBC::ScanAndBcast(loc_bucket_sizes.data(),
                        loc_bucket_sizes_exscan.data(), glob_bucket_sizes.data(),
                        loc_bucket_sizes.size(), Common::getMpiType(loc_bucket_sizes),
                        MPI_SUM, ams_data.comm());
    }
    for (size_t idx = 0; idx != loc_bucket_sizes_exscan.size(); ++idx) {
      loc_bucket_sizes_exscan[idx] -= loc_bucket_sizes[idx];
    }

    ams_data.config.tracker.splitter_allgather_scan_t.stop();

    // Calculate number of splitter and samples

    ams_data.config.tracker.overpartition_t.start(ams_data.comm());

    size_t op_it_cnt = 0;
    std::vector<size_t> loc_group_el_cnts;
    std::vector<size_t> loc_group_el_cnts_scan;
    std::vector<size_t> glob_group_el_cnts;

    tie(loc_group_el_cnts, loc_group_el_cnts_scan, glob_group_el_cnts) =
      Overpartition::assignElementsToGroups(group_sizes,
                                            loc_bucket_sizes, loc_bucket_sizes_exscan,
                                            glob_bucket_sizes,
                                            level_eps, use_equal_buckets, &op_it_cnt,
                                            ams_data.comm());
    ams_data.config.tracker.overpartition_repeats_c_.add(op_it_cnt);

    ams_data.config.tracker.overpartition_t.stop();

    /* Data exchange */

    // p-way exchange: last level of recursion
    if (ams_data.nprocs == group_sizes.size()) {
      assert(loc_group_el_cnts.size() == ams_data.nprocs);

      ams_data.config.tracker.exchange_t.start(ams_data.comm());

      exchangePway(ams_data, loc_group_el_cnts, glob_group_el_cnts);

      ams_data.data.swap(ams_data.tmp_data);

      ams_data.config.tracker.exchange_t.stop();
      return;
    }

    ams_data.config.tracker.msg_assignment_t.start(ams_data.comm());

    // Calculate residual capacity.
    for (size_t i = 0; i != num_groups; ++i) {
      const auto res = tlx::div_ceil(glob_group_el_cnts[i], group_sizes[i]);
      ams_data.residual = std::max(ams_data.residual, res);
    }
    ams_data.np_ceiled_act = ams_data.residual;

    DistrRanges distr_ranges;
    size_t max_num_recv_msgs;
    size_t max_recv_els;

    if (ams_data.config.use_dma) {
      std::tie(distr_ranges, max_num_recv_msgs, max_recv_els) =
        GroupMsgToPeAssignment::detAssignment<typename AmsData::Tags>(loc_group_el_cnts,
                                              glob_group_el_cnts,
                                              group_sizes,
                                              group_sizes_exscan,
                                              my_group_idx,
                                              my_group_rank,
                                              ams_data.residual,
                                              ams_data.config.distr_strategy,
                                              ams_data.config.tracker,
                                              ams_data.comm(),
                                              group_comm);
    } else {
      std::tie(distr_ranges, max_num_recv_msgs, max_recv_els) =
        GroupMsgToPeAssignment::simpleAssignment(group_sizes,
                                                 group_sizes_exscan,
                                                 loc_group_el_cnts,
                                                 loc_group_el_cnts_scan,
                                                 glob_group_el_cnts,
                                                 my_group_idx,
                                                 ams_data.comm());
    }

    ams_data.config.tracker.msg_assignment_t.stop();

    ams_data.config.tracker.exchange_t.start(ams_data.comm());

    exchangeKway(ams_data, distr_ranges, max_num_recv_msgs, max_recv_els);

    ams_data.data.swap(ams_data.tmp_data);

    // Update data load
    ams_data.n_act = glob_group_el_cnts[my_group_idx];

    ams_data.config.tracker.exchange_t.stop();
  }    // Guarantees that temp data is destroyed.

  ams_data.increaseLevel();

  // there might be further levels for other process groups!
  if (ams_data.nprocs == 1) {
    return;
  }
  assert(ams_data.nprocs > 1);
  assert(ams_data.level() < ams_data.myNumLevels());
  recSort(ams_data);
}

// Change ptr of level_descrs to unique_ptr
template <class AmsTags, class T, class Config, class Comp>
void sort(LevelDescrInterface* level_descrs, const Config& config,
          Comp comp, MPI_Datatype mpi_type,
          std::mt19937_64& async_gen, std::vector<T>& data) {
  const RBC::Comm& comm = level_descrs->initComm();

  if (level_descrs->myNumLevels() > 0) {
    config.tracker.various_t.start(comm);

    // Calculate total number of elements and maximum n/p.
    size_t sync_gen_seed = comm.getRank() == 0 ? async_gen() : 0;
    Tools::Tuple<size_t, size_t> ar(data.size(), sync_gen_seed);
    Tools::Tuple<size_t, size_t> aggr;

    static MPI_Op sum_max_op = MPI_OP_NULL;
    if (sum_max_op == MPI_OP_NULL) {
      MPI_Op_create(reinterpret_cast<MPI_User_function*>(&mpiMaxSum<Tools::Tuple<size_t, size_t> >),
                    true,
                    &sum_max_op);
    }

    MPI_Datatype tuple_type = Tools::Tuple<size_t, size_t>::MpiType(
      Common::getMpiType<size_t>(), Common::getMpiType<size_t>());
    RBC::Allreduce(&ar, &aggr, 1, tuple_type, sum_max_op, comm);
    assert(comm.getRank() != 0 || sync_gen_seed == aggr.second);

    const size_t n = aggr.first;
    sync_gen_seed = aggr.second;

    const size_t max_exp_np = tlx::div_ceil(n, comm.getSize()) * (1. + config.max_epsilon);
    std::vector<T> tmp_data;
    tmp_data.reserve(max_exp_np);

    AmsData<AmsTags, T, Config, Comp> ams_data{ config, comp, mpi_type, level_descrs, async_gen,
                                       data, tmp_data, n, sync_gen_seed, comm };

    config.tracker.various_t.stop();

    // sort recursively
    recSort(ams_data);
  }

  config.tracker.local_sort_t.start(comm);

  if (config.use_ips4o) {
    // Slow on juqueen
    ips4o::sort(data.begin(), data.end(), comp);
  } else {
    // Fast on juqueen
    std::sort(data.begin(), data.end(), comp);
  }

  config.tracker.local_sort_t.stop();
}
}  // end namespace _internal

// RBC functions

template <class T, class Tracker, class Comp, class AmsTags>
void sortTracker(MPI_Datatype mpi_type,
                 std::vector<T>& data, int k,
                 std::mt19937_64& async_gen,
                 Tracker& tracker,
                 const RBC::Comm& comm,
                 Comp comp,
                 double imbalance,
                 bool use_dma,
                 PartitioningStrategy part_strategy,
                 DistributionStrategy distr_strategy,
                 bool use_ips4o,
                 bool use_two_tree) {
  _internal::Config<Comp, Tracker> config(tracker,
                                          use_dma,
                                          use_ips4o,
                                          use_two_tree,
                                          imbalance - 1.0,
                                          part_strategy,
                                          distr_strategy);
  config.tracker.split_comm_t.start(comm);
  _internal::RecDescrKway level_descrs(k, comm);
  config.tracker.split_comm_t.stop();
  _internal::sort<AmsTags>(&level_descrs, config, comp, mpi_type, async_gen, data);
}

template <class T, class Comp, class AmsTags>
void sort(MPI_Datatype mpi_type, std::vector<T>& data, int k,
          std::mt19937_64& async_gen,
          const RBC::Comm& comm,
          Comp comp,
          double imbalance,
          bool use_dma,
          PartitioningStrategy part_strategy,
          DistributionStrategy distr_strategy,
          bool use_ips4o,
          bool use_two_tree) {
  _internal::RecDescrKway level_descrs(k, comm);
  _internal::DummyTracker tracker;
  _internal::Config<Comp, _internal::DummyTracker> config(tracker,
                                                          use_dma,
                                                          use_ips4o,
                                                          use_two_tree,
                                                          imbalance - 1.0,
                                                          part_strategy,
                                                          distr_strategy);
  _internal::sort<AmsTags>(&level_descrs, config, comp, mpi_type, async_gen, data);
}

template <class T, class Tracker, class Comp, class AmsTags>
void sortTracker(MPI_Datatype mpi_type, std::vector<T>& data,
                 std::vector<size_t>& ks,
                 std::mt19937_64& async_gen,
                 Tracker& tracker,
                 const RBC::Comm& comm,
                 Comp comp,
                 double imbalance,
                 bool use_dma,
                 PartitioningStrategy part_strategy,
                 DistributionStrategy distr_strategy,
                 bool use_ips4o,
                 bool use_two_tree) {
  _internal::Config<Comp, Tracker> config(tracker,
                                          use_dma,
                                          use_ips4o,
                                          use_two_tree,
                                          imbalance - 1.0,
                                          part_strategy,
                                          distr_strategy);
  config.tracker.split_comm_t.start(comm);
  _internal::RecDescrKways level_descrs(comm, ks);
  config.tracker.split_comm_.stop();
  _internal::sort<AmsTags>(&level_descrs, config, comp, mpi_type, async_gen, data);
}

template <class T, class Comp, class AmsTags>
void sort(MPI_Datatype mpi_type, std::vector<T>& data,
          std::vector<size_t>& ks,
          std::mt19937_64& async_gen,
          const RBC::Comm& comm,
          Comp comp,
          double imbalance,
          bool use_dma,
          PartitioningStrategy part_strategy,
          DistributionStrategy distr_strategy,
          bool use_ips4o,
          bool use_two_tree) {
  _internal::RecDescrKways level_descrs(comm, ks);
  _internal::DummyTracker tracker;
  _internal::Config<Comp, _internal::DummyTracker> config(mpi_type,
                                                          tracker, comp,
                                                          use_dma,
                                                          use_ips4o,
                                                          use_two_tree,
                                                          imbalance - 1.0,
                                                          part_strategy,
                                                          distr_strategy);
  _internal::sort<AmsTags>(&level_descrs, config, comp, mpi_type, async_gen, data);
}

template <class T, class Tracker, class Comp, class AmsTags>
void sortTrackerLevel(MPI_Datatype mpi_type, std::vector<T>& data, int l,
                      std::mt19937_64& async_gen,
                      Tracker& tracker,
                      const RBC::Comm& comm,
                      Comp comp,
                      double imbalance,
                      bool use_dma,
                      PartitioningStrategy part_strategy,
                      DistributionStrategy distr_strategy,
                      bool use_ips4o,
                      bool use_two_tree) {
  const auto k = 1 << tlx::div_ceil(tlx::integer_log2_ceil(comm.getSize()), l);

  _internal::Config<Comp, Tracker> config(tracker,
                                          use_dma,
                                          use_ips4o,
                                          use_two_tree,
                                          imbalance - 1.0,
                                          part_strategy,
                                          distr_strategy);
  config.tracker.split_comm_t.start(comm);
  _internal::RecDescrKway level_descrs(k, comm);
  config.tracker.split_comm_t.stop();
  _internal::sort<AmsTags>(&level_descrs, config, comp, mpi_type, async_gen, data);
}

template <class T, class Comp, class AmsTags>
void sortLevel(MPI_Datatype mpi_type, std::vector<T>& data, int l,
               std::mt19937_64& async_gen,
               const RBC::Comm& comm,
               Comp comp,
               double imbalance,
               bool use_dma,
               PartitioningStrategy part_strategy,
               DistributionStrategy distr_strategy,
               bool use_ips4o,
               bool use_two_tree) {
  const auto k = 1 << tlx::div_ceil(tlx::integer_log2_ceil(comm.getSize()), l);

  _internal::RecDescrKway level_descrs(k, comm);
  _internal::DummyTracker tracker;
  _internal::Config<Comp, _internal::DummyTracker> config(tracker,
                                                          use_dma,
                                                          use_ips4o,
                                                          use_two_tree,
                                                          imbalance - 1.0,
                                                          part_strategy,
                                                          distr_strategy);
  _internal::sort<AmsTags>(&level_descrs, config, comp, mpi_type, async_gen, data);
}

// MPI functions

template <class T, class Tracker, class Comp, class AmsTags>
void sortTracker(MPI_Datatype mpi_type,
                 std::vector<T>& data, int k,
                 std::mt19937_64& async_gen,
                 Tracker& tracker,
                 MPI_Comm comm,
                 Comp comp,
                 double imbalance,
                 bool use_dma,
                 PartitioningStrategy part_strategy,
                 DistributionStrategy distr_strategy,
                 bool use_ips4o,
                 bool use_two_tree) {
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  sortTracker<T, Tracker, Comp, AmsTags>(mpi_type,
              data,
              k,
              async_gen,
              tracker,
              rcomm,
              comp,
              imbalance,
              use_dma,
              part_strategy,
              distr_strategy,
              use_ips4o,
              use_two_tree);
}

template <class T, class Comp, class AmsTags>
void sort(MPI_Datatype mpi_type, std::vector<T>& data, int k,
          std::mt19937_64& async_gen,
          MPI_Comm comm,
          Comp comp,
          double imbalance,
          bool use_dma,
          PartitioningStrategy part_strategy,
          DistributionStrategy distr_strategy,
          bool use_ips4o,
          bool use_two_tree) {
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  sort<T, Comp, AmsTags>(mpi_type,
       data,
       k,
       async_gen,
       rcomm,
       comp,
       imbalance,
       use_dma,
       part_strategy,
       distr_strategy,
       use_ips4o,
       use_two_tree);
}

template <class T, class Tracker, class Comp, class AmsTags>
void sortTracker(MPI_Datatype mpi_type, std::vector<T>& data,
                 std::vector<size_t>& ks,
                 std::mt19937_64& async_gen,
                 Tracker& tracker,
                 MPI_Comm comm,
                 Comp comp,
                 double imbalance,
                 bool use_dma,
                 PartitioningStrategy part_strategy,
                 DistributionStrategy distr_strategy,
                 bool use_ips4o,
                 bool use_two_tree) {
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  sortTracker<T, Tracker, Comp, AmsTags>(mpi_type,
              data,
              ks,
              async_gen,
              tracker,
              rcomm,
              comp,
              imbalance,
              use_dma,
              part_strategy,
              distr_strategy,
              use_ips4o,
              use_two_tree);
}

template <class T, class Comp, class AmsTags>
void sort(MPI_Datatype mpi_type, std::vector<T>& data,
          std::vector<size_t>& ks,
          std::mt19937_64& async_gen,
          MPI_Comm comm,
          Comp comp,
          double imbalance,
          bool use_dma,
          PartitioningStrategy part_strategy,
          DistributionStrategy distr_strategy,
          bool use_ips4o,
          bool use_two_tree) {
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  sort<T, Comp, AmsTags>(mpi_type,
       data,
       ks,
       async_gen,
       rcomm,
       comp,
       imbalance,
       use_dma,
       part_strategy,
       distr_strategy,
       use_ips4o,
       use_two_tree);
}

template <class T, class Tracker, class Comp, class AmsTags>
void sortTrackerLevel(MPI_Datatype mpi_type, std::vector<T>& data, int l,
                      std::mt19937_64& async_gen,
                      Tracker& tracker,
                      MPI_Comm comm,
                      Comp comp,
                      double imbalance,
                      bool use_dma,
                      PartitioningStrategy part_strategy,
                      DistributionStrategy distr_strategy,
                      bool use_ips4o,
                      bool use_two_tree) {
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  sortTrackerLevel<T, Tracker, Comp, AmsTags>(mpi_type,
                   data,
                   l,
                   async_gen,
                   tracker,
                   rcomm,
                   comp,
                   imbalance,
                   use_dma,
                   part_strategy,
                   distr_strategy,
                   use_ips4o,
                   use_two_tree);
}

template <class T, class Comp, class AmsTags>
void sortLevel(MPI_Datatype mpi_type, std::vector<T>& data, int l,
               std::mt19937_64& async_gen,
               MPI_Comm comm,
               Comp comp,
               double imbalance,
               bool use_dma,
               PartitioningStrategy part_strategy,
               DistributionStrategy distr_strategy,
               bool use_ips4o,
               bool use_two_tree) {
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  sortLevel<T, Comp, AmsTags>(mpi_type,
            data,
            l,
            async_gen,
            rcomm,
            comp,
            imbalance,
            use_dma,
            part_strategy,
            distr_strategy,
            use_ips4o,
            use_two_tree);
}
}  // end namespace Ams
