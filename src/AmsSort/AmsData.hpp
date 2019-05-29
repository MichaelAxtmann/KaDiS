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
#include <cstddef>
#include <memory>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <RBC.hpp>
#include <tlx/math.hpp>

#include "../../include/AmsSort/Configuration.hpp"

namespace Ams {
namespace _internal {
class LevelDescrInterface {
 public:
  /* @brief Number of levels
   *
   * Returns the number of levels of AMS-sort. If the number of levels
   * differs between PEs, getnumlevels returns the maximum number of levels.
   */
  virtual size_t myNumLevels() const = 0;
  virtual size_t maxNumLevels() const = 0;
  virtual RBC::Comm & initComm() = 0;
  /* @brief Returns the group communicator on level <level>.
   *
   * Returns the group communicator on level <level>. The last group
   * communicator contains a single PE.
   */
  virtual RBC::Comm & groupComm(size_t level) = 0;
  virtual const std::vector<size_t> & groupSizes(size_t level) const = 0;
  virtual size_t myGroupIdx(size_t level) const = 0;
  virtual size_t myGroupRank(size_t level) const = 0;

 protected:
  size_t maxSubgroupSize(size_t p, size_t k) const {
    return tlx::div_ceil(p, k);
  }

  /* @brief Calculates the process groups the communicator.
   *
   * This method creates for each process of 'comm' the communicator
   * of its process subgroup. In total, min(|comm|, k) groups are
   * created. If |comm|<k, groups of size one are
   * created. Otherwise, groups of size |comm|/'k' (first |comm| % k
   * processes) or of size 'k-1' (remaining processes) are
   * constructed.
   */
  std::tuple<std::vector<size_t>, size_t, RBC::Comm> calculateLevel(size_t k,
                                                                    const RBC::Comm& comm) const {
    using return_type = std::tuple<std::vector<size_t>, size_t, RBC::Comm>;

    const size_t num_groups = std::min<size_t>(k, comm.getSize());
    const size_t num_large_groups = comm.getSize() % num_groups;
    const size_t group_size = comm.getSize() / num_groups;
    const size_t large_group_size = group_size + 1;

    std::vector<size_t> group_sizes(num_large_groups, large_group_size);
    group_sizes.insert(group_sizes.end(), num_groups - num_large_groups, group_size);

    if (static_cast<size_t>(comm.getRank()) < num_large_groups * large_group_size) {
      const size_t my_group_idx = comm.getRank() / large_group_size;
      const size_t first = my_group_idx * large_group_size;
      const size_t last = first + large_group_size - 1;
      RBC::Comm group_comm;
      RBC::Comm_create_group(comm, &group_comm, first, last);
      return return_type{ group_sizes, my_group_idx, std::move(group_comm) };
    } else {
      const size_t num_large_groups_pes = num_large_groups * large_group_size;
      const size_t num_left_regular_groups = (comm.getRank() - num_large_groups_pes)
                                             / group_size;
      const size_t my_group_idx = num_large_groups + num_left_regular_groups;
      const size_t first = num_large_groups_pes + num_left_regular_groups * group_size;
      const size_t last = first + group_size - 1;
      RBC::Comm group_comm;
      RBC::Comm_create_group(comm, &group_comm, first, last);
      return return_type{ group_sizes, my_group_idx, std::move(group_comm) };
    }
  }
};

template <typename Comp, class Tracker>
struct Config {
  Config(Tracker& tracker,
         bool use_dma,
         bool use_ips4o,
         bool use_two_tree,
         double max_epsilon,
         PartitioningStrategy part_strategy,
         DistributionStrategy distr_strategy) :
    tracker(tracker),
    use_dma(use_dma),
    use_ips4o(use_ips4o),
    use_two_tree(use_two_tree),
    max_epsilon(max_epsilon),
    part_strategy(part_strategy),
    distr_strategy(distr_strategy)
  { }

  Tracker& tracker;
  const bool use_dma;
  const bool use_ips4o;
  const bool use_two_tree;
  const double max_epsilon;
  const PartitioningStrategy part_strategy;
  const DistributionStrategy distr_strategy;
};

template <class AmsTags, class value_type, class Config, class Comp>
class AmsData {
 public:
  using Tags = AmsTags;
  using T = value_type;

  // Make class non-copyable.
  AmsData() = delete;
  AmsData(const AmsData&) = delete;
  void operator= (const AmsData&) = delete;

  AmsData(const Config& config,
          Comp comp,
          MPI_Datatype mpi_type,
          LevelDescrInterface* level_descrs,
          std::mt19937_64& async_gen,
          std::vector<T>& data,
          std::vector<T>& tmp_data,
          size_t n,
          size_t sync_gen_seed,
          const RBC::Comm& comm) :
    config(config),
    comp(comp),
    mpi_type(mpi_type),
    level_descrs(level_descrs),
    async_gen(async_gen),
    sync_gen(sync_gen_seed),
    data(data),
    tmp_data(tmp_data),
    residual(0),
    n_init(n),
    np_ceiled_init(tlx::div_ceil(n, comm.getSize())),
    n_act(n),
    np_ceiled_act(tlx::div_ceil(n, comm.getSize())),
    nprocs(comm.getSize()),
    init_nprocs(comm.getSize()),
    myrank(comm.getRank()),
    comm_(comm),
    curr_level_(0)
  { }

  const Config& config;
  const Comp comp;
  const MPI_Datatype mpi_type;
  LevelDescrInterface* level_descrs;
  std::mt19937_64& async_gen;
  std::mt19937_64 sync_gen;
  std::vector<T>& data;
  std::vector<T>& tmp_data;

  // Initialized with local input size.
  size_t residual;
  const size_t n_init;
  const size_t np_ceiled_init;
  size_t n_act;
  size_t np_ceiled_act;

  size_t nprocs;
  const size_t init_nprocs;
  size_t myrank;

  RBC::Comm & comm() {
    return comm_;
  }

  size_t level() {
    return curr_level_;
  }

  void increaseLevel() {
    comm_ = level_descrs->groupComm(curr_level_);
    ++curr_level_;
    myrank = comm_.getRank();
    nprocs = comm_.getSize();
  }

  const std::vector<size_t> & groupSizes() {
    return level_descrs->groupSizes(curr_level_);
  }

  RBC::Comm & groupComm() {
    return level_descrs->groupComm(curr_level_);
  }

  size_t myGroupIdx() const {
    return level_descrs->myGroupIdx(curr_level_);
  }

  size_t myGroupRank() const {
    return level_descrs->myGroupRank(curr_level_);
  }

  size_t myNumLevels() const {
    return level_descrs->myNumLevels();
  }

 private:
  RBC::Comm comm_;
  size_t curr_level_;
};
}  // namespace _internal
}  // namespace Ams
