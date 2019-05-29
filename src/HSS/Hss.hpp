/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiS).
 *
 * Copyright (c) 2020, Michael Axtmann <michael.axtmann@kit.edu>
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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "../AmsSort/AmsSort.hpp"
#include "../AmsSort/DistrRange.hpp"
#include "../AmsSort/DummyTracker.hpp"
#include "../Tools/Common.hpp"
#include "../Tools/CommonMpi.hpp"

#include "ips2pa.hpp"
#include "ips4o.hpp"

#include <tlx/algorithm.hpp>
#include <tlx/container.hpp>
#include <tlx/math.hpp>

#include <RBC.hpp>

namespace Hss {
namespace _internal {
template <class T>
struct TbSample {
  TbSample(const T& splitter, const int64_t& GID) noexcept :
    splitter(splitter),
    GID(GID) {
    assert(GID != std::numeric_limits<int64_t>::max());
  }

  TbSample() noexcept :
    splitter(T()),
    GID(std::numeric_limits<size_t>::max())
  { }

  T splitter;
  size_t GID;

  static MPI_Datatype MpiType(const MPI_Datatype& mpi_type) {
    static MPI_Datatype splitter_type = MPI_DATATYPE_NULL;

    if (splitter_type == MPI_DATATYPE_NULL) {
      const int nitems = 2;
      int blocklengths[2] = { 1, 1 };
      MPI_Datatype types[2];
      types[0] = mpi_type;
      types[1] = Common::getMpiType<decltype(GID)>();
      MPI_Aint offsets[2] = { offsetof(TbSample<T>, splitter),
                              offsetof(TbSample<T>, GID) };

      MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                             &splitter_type);
      MPI_Type_commit(&splitter_type);
    }

    return splitter_type;
  }
};

template <class AmsTags, typename Comp, class Tracker>
struct Config {
  using Tags = AmsTags;
  
  Config(MPI_Datatype mpi_type,
         Tracker& tracker, Comp comp,
         Ams::DistributionStrategy distr_strategy,
         bool use_deterministic_group_assignment,
         bool keep_local_data,
         bool use_ips4o,
         bool use_two_tree,
         double max_epsilon) :
    mpi_type(mpi_type),
    tracker(tracker),
    comp(comp),
    distr_strategy(distr_strategy),
    use_deterministic_group_assignment(use_deterministic_group_assignment),
    keep_local_data(keep_local_data),
    use_ips4o(use_ips4o),
    use_two_tree(use_two_tree),
    max_epsilon(max_epsilon)
  { }

  const MPI_Datatype mpi_type;
  Tracker& tracker;
  const Comp comp;
  const Ams::DistributionStrategy distr_strategy;
  const bool use_deterministic_group_assignment;
  const bool keep_local_data;
  const bool use_ips4o;
  const bool use_two_tree;
  const double max_epsilon;
};

class Range {
 public:
  Range() = delete;
  Range(size_t target_idx,
        size_t lower_accepted_rank,
        size_t upper_accepted_rank,
        size_t left_idx,
        size_t right_idx,
        size_t left_glob_idx,
        size_t right_glob_idx) :
    target_idx_(target_idx),
    lower_accepted_rank_(lower_accepted_rank),
    upper_accepted_rank_(upper_accepted_rank),
    left_idx_(left_idx),
    right_idx_(right_idx),
    left_glob_idx_(left_glob_idx),
    right_glob_idx_(right_glob_idx) { }

  size_t GetTargetIdx() const {
    return target_idx_;
  }

  size_t GetLeftIdx() const {
    return left_idx_;
  }

  size_t GetRightIdx() const {
    return right_idx_;
  }

  size_t GetLeftGlobIdx() const {
    return left_glob_idx_;
  }

  size_t GetRightGlobIdx() const {
    return right_glob_idx_;
  }

  void SetLeftIdx(size_t idx) {
    left_idx_ = idx;
  }

  void SetRightIdx(size_t idx) {
    right_idx_ = idx;
  }

  void SetLeftGlobIdx(size_t idx) {
    left_glob_idx_ = idx;
  }

  void SetRightGlobIdx(size_t idx) {
    right_glob_idx_ = idx;
  }

  std::pair<size_t, size_t> GetBestSplits() const {
    if (target_idx_ - left_glob_idx_ < right_glob_idx_ - target_idx_) {
      return std::pair<size_t, size_t>(left_idx_, left_glob_idx_);
    } else {
      return std::pair<size_t, size_t>(right_idx_, right_glob_idx_);
    }
  }

  bool IsValidRange() const {
    return lower_accepted_rank_ <= left_glob_idx_ ||
           right_glob_idx_ <= upper_accepted_rank_;
  }

  void Print(std::ostream& os) const {
    os << "Range("
       << target_idx_ << ", "
       << lower_accepted_rank_ << ", "
       << upper_accepted_rank_ << ", "
       << left_idx_ << ", "
       << right_idx_ << ", "
       << left_glob_idx_ << ", "
       << right_glob_idx_ << ")";
  }

 private:
  size_t target_idx_;
  size_t lower_accepted_rank_, upper_accepted_rank_;
  size_t left_idx_, right_idx_;
  size_t left_glob_idx_, right_glob_idx_;
};


std::ostream& operator<< (std::ostream& os,
                          const Range& r) {
  r.Print(os);
  return os;
}


using Ranges = std::vector<Range>;

template <class T>
std::vector<TbSample<T> > Sample(const std::vector<T>& data,
                                 const Ranges& ranges,
                                 size_t first_el_glob_rank,
                                 double prob,
                                 std::mt19937_64& async_gen) {
  assert(prob > 0);

  std::vector<TbSample<T> > samples;

  // Select all samples. Skip equal ranges.
  if (prob >= 1) {
    for (auto range = ranges.begin(); range != ranges.end(); ++range) {
      if (range != ranges.begin()) {
        const auto prev_range = range - 1;
        if (prev_range->GetLeftIdx() == range->GetLeftIdx() &&
            prev_range->GetRightIdx() == range->GetRightIdx()) {
          continue;
        } else {
          assert(prev_range->GetRightIdx() <= range->GetLeftIdx());
        }
      }
      for (auto idx = range->GetLeftIdx(); idx != range->GetRightIdx(); ++idx) {
        samples.emplace_back(data[idx], first_el_glob_rank + idx);
      }
    }
  } else {
    std::geometric_distribution<size_t> distr(prob);

    if (ranges.size() == 0) {
      return samples;
    }

    auto range_ptr = ranges.begin();
    size_t pos = range_ptr->GetLeftIdx();

    // Sampling by iterating from range to range.
    while (range_ptr != ranges.end()) {
      assert(ranges.begin() <= range_ptr && range_ptr < ranges.end());
      // The next element is offset steps away.
      size_t offset = distr(async_gen);

      // The current range does not contain elements from pos to the
      // right end.  So jump to the subsequent ranges. Subsequent
      // ranges are either exactly the same Ranges or do not overlap
      // at all.
      while (range_ptr->GetRightIdx() <= pos + offset) {
        // No range left.
        if (range_ptr + 1 == ranges.end()) {
          return samples;
        }

        assert((range_ptr->GetLeftIdx() == (range_ptr + 1)->GetLeftIdx() &&
                range_ptr->GetRightIdx() == (range_ptr + 1)->GetRightIdx()) ||
               range_ptr->GetRightIdx() <= (range_ptr + 1)->GetLeftIdx());

        // The remaining elements of the current offset are removed.
        offset -= range_ptr->GetRightIdx() - pos;

        // Jump to the next range. Attention: skip equal ranges. Equal
        // ranges may exist as the ranges initially cover all
        // elements.  Ranges are either equal or do not overlap at
        // all.
        auto prev_range_ptr = range_ptr;
        do {
          prev_range_ptr = range_ptr;
          ++range_ptr;
          if (range_ptr == ranges.end()) {
            return samples;
          }
        } while (prev_range_ptr->GetLeftIdx() == range_ptr->GetLeftIdx() &&
                 prev_range_ptr->GetRightIdx() == range_ptr->GetRightIdx());

        assert(prev_range_ptr->GetRightIdx() <= range_ptr->GetLeftIdx());
        assert(ranges.begin() <= range_ptr && range_ptr < ranges.end());

        pos = range_ptr->GetLeftIdx();
      }

      pos = pos + offset;
      assert(ranges.begin() <= range_ptr && range_ptr < ranges.end());
      assert(pos < range_ptr->GetRightIdx());
      assert(pos >= 0 && pos < data.size());
      samples.emplace_back(data[pos], first_el_glob_rank + pos);
      pos++;
    }
  }

  return samples;
}

// Returns number of elements in each partition.
template <class T, class Config>
std::tuple<std::vector<size_t>, std::vector<size_t> >
Partition(const std::vector<T>& data, double imb,
          const std::vector<size_t>& group_sizes, const std::vector<size_t>& group_sizes_exscan,
          size_t n, std::mt19937_64& async_gen, const Config& config,
          const RBC::Comm& comm) {
  const size_t nprocs = comm.getSize();

  const auto& type_comp = config.comp;
  const auto sample_comp = [&type_comp](const TbSample<T>& left, const TbSample<T>& right) {
                             // We do not break ties as each element is selected at
                             // most once.
                             return type_comp(left.splitter, right.splitter);
                           };
  MPI_Datatype mpi_type = TbSample<T>::MpiType(config.mpi_type);

  // Create a temporary communicator which ensures that the twotree
  // implementation is used even when RBC collectives are disabled in
  // the RBC communicator 'comm'.
  RBC::Comm rbc_comm, rbc_subcomm;
  // Splitting disabled, RBC collectives enabled
  RBC::Create_Comm_from_MPI(comm.get(), &rbc_comm, true, false, false);
  RBC::Comm_create_group(rbc_comm, &rbc_subcomm, comm.getMpiFirst(),
                         comm.getMpiLast(), comm.getStride());

  const size_t group_cnt = group_sizes.size();
  const size_t loc_el_cnt = data.size();

  const size_t first_el_glob_rank = [&]() {
                                      size_t r = 0;
                                      RBC::Exscan(&loc_el_cnt,
                                                  &r,
                                                  1,
                                                  Common::getMpiType(loc_el_cnt),
                                                  MPI_SUM,
                                                  comm);
                                      return r;
                                    } ();

  Ranges ranges;
  for (size_t i = 0; i != group_cnt - 1; ++i) {
    const size_t target_rank = static_cast<double>(n) / nprocs * group_sizes_exscan[i + 1];
    const size_t lower_accepted_rank = target_rank - std::ceil(
      0.5 * n / nprocs * group_sizes[i] * imb);
    const size_t upper_accepted_rank = target_rank + std::ceil(
      0.5 * n / nprocs * group_sizes[i + 1] * imb);
    ranges.emplace_back(target_rank,
                        lower_accepted_rank,
                        upper_accepted_rank,
                        0, loc_el_cnt,
                        0, n);
  }

  size_t remaining_el_cnt = n;
  size_t tot_sample_cnt = 0;
  size_t hs_rounds = 0;

  const auto range_is_valid = [](const Range& r) {
                                return r.IsValidRange();
                              };

  // Sampling by iterating from range to range.
  do {
    // Collect local samples
    const double prob = std::min(5. * group_cnt / remaining_el_cnt, 1.);
    const std::vector<TbSample<T> > lsamples = Sample(data,
                                                      ranges,
                                                      first_el_glob_rank,
                                                      prob,
                                                      async_gen);

    // Calculate sorted array of global splitters.
    const auto gsamples = Ams::SplitterSelection::_internal::Allgathermerge(
      lsamples, mpi_type, sample_comp, config.use_two_tree, comm);

    if (gsamples.empty()) {
      continue;
    }

    if (gsamples.empty()) {
      continue;
    }
    tot_sample_cnt += gsamples.size();

    // Calculate local histogram
    std::vector<size_t> loc_histogram(gsamples.size());
    for (size_t i = 0; i != gsamples.size(); ++i) {
      if (first_el_glob_rank <= gsamples[i].GID &&
          // We picked that sample.
          gsamples[i].GID < first_el_glob_rank + loc_el_cnt) {
        loc_histogram[i] = gsamples[i].GID - first_el_glob_rank;
      } else if (gsamples[i].GID < first_el_glob_rank) {
        // Sample has been picked by a PE to our left.
        const auto it =
          std::lower_bound(data.begin(), data.end(), gsamples[i].splitter, config.comp);
        loc_histogram[i] = it - data.begin();
      } else {
        assert(first_el_glob_rank + loc_el_cnt <= gsamples[i].GID);
        const auto it =
          std::upper_bound(data.begin(), data.end(), gsamples[i].splitter, config.comp);
        loc_histogram[i] = it - data.begin();
      }
    }

    // Calculate global histogram
    // Always use the twotree implementation as this implementation is
    // super fast.
    std::vector<size_t> glob_histogram(gsamples.size());
    RBC::_internal::optimized::AllreduceTwotree(
      loc_histogram.data(), glob_histogram.data(),
      gsamples.size(), Common::getMpiType(loc_histogram), MPI_SUM, rbc_subcomm);

    // Refinement of sample ranges
    auto range_it = ranges.begin();
    assert(std::is_sorted(data.begin(), data.end(), config.comp));
    for (int64_t sample_idx = 0; sample_idx != static_cast<int64_t>(gsamples.size());
         ++sample_idx) {
      if (range_it == ranges.end()) {
        break;
      }

      if (glob_histogram[sample_idx] < range_it->GetTargetIdx()) {
        // We found a sample with rank smaller than our target rank ->
        // try to update
        if (glob_histogram[sample_idx] >= range_it->GetLeftGlobIdx()) {
          // Update
          range_it->SetLeftIdx(loc_histogram[sample_idx]);
          range_it->SetLeftGlobIdx(glob_histogram[sample_idx]);
        }

        // Go to the next range if no splitters are left. Remain at
        // the current sample as it might be used to update the left
        // part of the next range
        if (sample_idx + 1 == static_cast<int64_t>(gsamples.size())) {
          ++range_it;
          --sample_idx;
        }
      } else {
        // We found a sample with rank smaller than our target rank ->
        // try to update and go to next range
        assert(range_it->GetTargetIdx() <= glob_histogram[sample_idx]);

        if (glob_histogram[sample_idx] <= range_it->GetRightGlobIdx()) {
          // Update
          range_it->SetRightIdx(loc_histogram[sample_idx]);
          range_it->SetRightGlobIdx(glob_histogram[sample_idx]);
        }

        // Go to next range and go to previous sample as it might be
        // used to update the left part of the next range
        ++range_it;
        sample_idx = std::max<int64_t>(sample_idx - 2, -1);
      }
    }

    // Sum up interval sizes
    remaining_el_cnt = ranges[0].GetRightGlobIdx() - ranges[0].GetLeftGlobIdx();
    if (ranges.size() > 1) {
      auto prev_range_it = ranges.begin();
      for (auto range_it = prev_range_it + 1;
           range_it != ranges.end();
           prev_range_it++, range_it++) {
        if (prev_range_it->GetLeftGlobIdx() != range_it->GetLeftGlobIdx() &&
            prev_range_it->GetRightGlobIdx() != range_it->GetRightGlobIdx()) {
          assert(prev_range_it->GetRightIdx() <= range_it->GetLeftIdx());
          assert(prev_range_it->GetRightGlobIdx() <= range_it->GetLeftGlobIdx());
          remaining_el_cnt += range_it->GetRightGlobIdx() - range_it->GetLeftGlobIdx();
        }
      }
    }

    ++hs_rounds;
  } while (!std::all_of(ranges.begin(), ranges.end(), range_is_valid));

  config.tracker.overpartition_repeats_c_.add(hs_rounds);

  // Calculate splitter positions
  std::vector<size_t> loc_group_el_cnts(group_cnt);
  std::vector<size_t> glob_group_el_cnts(group_cnt);

  size_t prev_loc_split = 0;
  size_t prev_glob_split = 0;

  for (size_t group = 0; group != group_cnt - 1; ++group) {
    size_t act_loc_split = 0;
    size_t act_glob_split = 0;

    std::tie(act_loc_split, act_glob_split) = ranges[group].GetBestSplits();

    assert(prev_loc_split <= act_loc_split);
    assert(prev_glob_split <= act_glob_split);

    loc_group_el_cnts[group] = act_loc_split - prev_loc_split;
    glob_group_el_cnts[group] = act_glob_split - prev_glob_split;

    prev_loc_split = act_loc_split;
    prev_glob_split = act_glob_split;
  }

  loc_group_el_cnts[group_cnt - 1] = loc_el_cnt - prev_loc_split;
  glob_group_el_cnts[group_cnt - 1] = n - prev_glob_split;

  config.tracker.samples_c_.add(tot_sample_cnt);

  return std::tuple<std::vector<size_t>, std::vector<size_t> >{ loc_group_el_cnts,
                                                                glob_group_el_cnts };
}

template <class T, class Config>
std::vector<std::pair<T*, T*> > exchangeKway(const Config& config,
                                             std::vector<T>& data,
                                             std::vector<T>& tmp_data,
                                             const DistrRanges& distr_ranges,
                                             size_t max_num_recv_msgs,
                                             size_t max_recv_els,
                                             // const std::vector<size_t>& loc_group_el_cnts,
                                             // const std::vector<size_t>& glob_group_el_cnts,
                                             const RBC::Comm& comm) {
  using Tags = typename Config::Tags;
  
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

  if (comm.useMPICollectives()) {
    std::vector<int> send_counts(comm.getSize(), 0);
    for (const auto& distr_range : distr_ranges) {
      send_counts[distr_range.pe] = distr_range.size;
    }

    return Alltoallv::MPIAlltoallvRanges(config.tracker, data, send_counts, tmp_data,
                                         config.mpi_type, comm);
  } else if (config.distr_strategy == Ams::DistributionStrategy::EXCHANGE_WITHOUT_RECV_SIZES) {
    return Alltoallv::exchangeWithoutRecvSizesReturnRanges<Tags>(config.tracker,
                                                           data,
                                                           distr_ranges,
                                                           tmp_data,
                                                           max_num_recv_msgs,
                                                           max_recv_els,
                                                           config.mpi_type,
                                                           comm);
  } else if (config.distr_strategy == Ams::DistributionStrategy::EXCHANGE_WITH_RECV_SIZES) {
    return Alltoallv::exchangeWithRecvSizesReturnRanges<Tags>(config.tracker,
                                                        data,
                                                        distr_ranges,
                                                        tmp_data,
                                                        config.mpi_type,
                                                        comm);
  } else {
    return Alltoallv::exchangeWithRecvSizesAndPortsReturnRanges<Tags>(config.tracker, data, distr_ranges,
                                                                tmp_data, config.mpi_type, comm);
  }
}

template <class T, class Config>
std::vector<std::pair<T*, T*> > exchangePway(const Config& config, std::vector<T>& data,
                                             std::vector<T>& tmp_data,
                                             const std::vector<size_t>& loc_group_el_cnts,
                                             const std::vector<size_t>& glob_group_el_cnts,
                                             const RBC::Comm& comm) {
  using Tags = typename Config::Tags;
  
  const size_t nprocs = comm.getSize();
  const size_t myrank = comm.getRank();

  if (comm.useMPICollectives()) {
    std::vector<int> send_counts(loc_group_el_cnts.begin(), loc_group_el_cnts.end());

    return Alltoallv::MPIAlltoallvRanges(config.tracker, data, send_counts, tmp_data,
                                         config.mpi_type, comm);
  } else {
    DistrRanges distr_ranges;
    distr_ranges.reserve(nprocs);
    size_t offset = 0;
    for (size_t pe_idx = 0; pe_idx != nprocs; ++pe_idx) {
      const size_t num_send_els = loc_group_el_cnts[pe_idx];
      if (num_send_els) {
        distr_ranges.emplace_back(pe_idx, offset, num_send_els);
        offset += num_send_els;
      }
    }

    assert(Ams::_internal::testAllElementsAssigned(loc_group_el_cnts, data.size())
           & Ams::_internal::verifySendDescription(data.size(), distr_ranges));

    tmp_data.resize(glob_group_el_cnts[myrank]);

    if (config.distr_strategy == Ams::DistributionStrategy::EXCHANGE_WITHOUT_RECV_SIZES) {
      const size_t max_msg_cnt = std::min<size_t>(nprocs, glob_group_el_cnts[myrank]);
      return Alltoallv::exchangeWithoutRecvSizesReturnRanges<Tags>(config.tracker, data, distr_ranges,
                                                             tmp_data, max_msg_cnt,
                                                             glob_group_el_cnts[myrank],
                                                             config.mpi_type, comm);
    } else if (config.distr_strategy == Ams::DistributionStrategy::EXCHANGE_WITH_RECV_SIZES) {
      return Alltoallv::exchangeWithRecvSizesReturnRanges<Tags>(config.tracker, data, distr_ranges,
                                                          tmp_data, config.mpi_type, comm);
    } else {
      return Alltoallv::exchangeWithRecvSizesAndPortsReturnRanges<Tags>(config.tracker,
                                                                  data,
                                                                  distr_ranges,
                                                                  tmp_data,
                                                                  config.mpi_type,
                                                                  comm);
    }
  }
}

template <class T, class Config>
void KWaySort(const size_t init_nprocs,
              Ams::_internal::LevelDescrInterface* level_descrs,
              const Config& config,
              size_t level_idx,
              std::mt19937_64& async_gen,
              std::vector<T>& data, std::vector<T>& tmp_data,
              size_t init_max_np,
              size_t max_np,
              size_t glob_el_cnt,
              double remaining_eps,
              RBC::Comm comm) {
  {  // Guarantees that temp data has been destroyed.
    if (glob_el_cnt == 0) {
      return;
    }

    const size_t nprocs = comm.getSize();
    config.tracker.various_t.start(comm);
    assert(glob_el_cnt == Ams::_internal::totalNumElements(data.size(), comm));

    const std::vector<size_t>& group_sizes = level_descrs->groupSizes(level_idx);
    const std::vector<size_t> group_sizes_exscan = [&group_sizes, &comm]() {
                                                     std::vector<size_t> group_sizes_exscan(
                                                       group_sizes.size() + 1);
                                                     tlx::exclusive_scan(group_sizes.begin(),
                                                                         group_sizes.end(),
                                                                         group_sizes_exscan.begin(),
                                                                         0);
                                                     return group_sizes_exscan;
                                                   } ();
    RBC::Comm group_comm = level_descrs->groupComm(level_idx);
    const size_t my_group_idx = level_descrs->myGroupIdx(level_idx);
    const size_t my_group_rank = level_descrs->myGroupRank(level_idx);
    const size_t num_groups = group_sizes.size();

    double level_eps = Ams::_internal::calcLevelEpsilon(level_idx,
                                                        level_descrs->maxNumLevels(),
                                                        init_max_np,
                                                        max_np,
                                                        config.max_epsilon);

    config.tracker.various_t.stop();

    config.tracker.sampling_t.start(comm);

    std::vector<size_t> loc_group_el_cnts;
    std::vector<size_t> loc_group_el_cnts_scan;
    std::vector<size_t> glob_group_el_cnts;

    tie(loc_group_el_cnts, glob_group_el_cnts) = Partition(
      data, level_eps, group_sizes, group_sizes_exscan,
      glob_el_cnt, async_gen, config, comm);

    config.tracker.sampling_t.stop();

    /* Data exchange */

    // p-way exchange: last level of recursion
    if (nprocs == group_sizes.size()) {
      assert(loc_group_el_cnts.size() == nprocs);

      config.tracker.exchange_t.start(comm);

      auto ranges = exchangePway(config, data, tmp_data, loc_group_el_cnts, glob_group_el_cnts,
                                 comm);

      config.tracker.exchange_t.stop();

      config.tracker.partition_t.start(comm);
      data.resize(tmp_data.size());

      Common::multiwayMerge(ranges, data.data(), config.comp);
      config.tracker.partition_t.stop();

      return;
    }

    config.tracker.msg_assignment_t.start(comm);

    // Calculate residual capacity.
    for (size_t i = 0; i != num_groups; ++i) {
      const size_t res = tlx::div_ceil(glob_group_el_cnts[i], group_sizes[i]);
      max_np = std::max(max_np, res);
    }

    // k-way exchange
    std::vector<std::pair<T*, T*> > ranges;
    DistrRanges distr_ranges;
    size_t max_num_recv_msgs;
    size_t max_recv_els;

    if (config.use_deterministic_group_assignment) {
      std::tie(distr_ranges, max_num_recv_msgs, max_recv_els) =
        Ams::_internal::GroupMsgToPeAssignment::detAssignment<typename Config::Tags>(
          loc_group_el_cnts, glob_group_el_cnts,
          group_sizes, group_sizes_exscan, my_group_idx, my_group_rank,
          max_np, config.distr_strategy, config.tracker, comm, group_comm);
    } else {
      std::tie(distr_ranges, max_num_recv_msgs, max_recv_els) =
        Ams::_internal::GroupMsgToPeAssignment::simpleAssignment(
          group_sizes, group_sizes_exscan,
          loc_group_el_cnts, glob_group_el_cnts,
          my_group_idx, config.use_two_tree, comm);
    }

    config.tracker.msg_assignment_t.stop();

    config.tracker.exchange_t.start(comm);

    ranges = exchangeKway(config, data, tmp_data, distr_ranges, max_num_recv_msgs, max_recv_els,
                          comm);

    assert(tmp_data.size() <= max_np);

    config.tracker.exchange_t.stop();

    assert(Ams::_internal::verifyNoDataLoss(data, tmp_data, config.mpi_type, comm));

    config.tracker.partition_t.start(comm);
    data.resize(tmp_data.size());
    Common::multiwayMerge(ranges, data.data(), config.comp);
    config.tracker.partition_t.stop();


    // Update data load
    glob_el_cnt = glob_group_el_cnts[my_group_idx];
  }  // Guarantees that temp data is destroyed.

  auto group_comm = level_descrs->groupComm(level_idx);
  const size_t nprocs = group_comm.getSize();
  if (nprocs > 1) {
    assert(level_idx + 1 < level_descrs->myNumLevels());
    Hss::_internal::KWaySort(init_nprocs,
                             level_descrs, config,
                             level_idx + 1,
                             async_gen, data,
                             tmp_data,
                             init_max_np,
                             max_np,
                             glob_el_cnt,
                             remaining_eps,
                             group_comm);
  }
}

template <class T, class Config>
void sort(Ams::_internal::LevelDescrInterface* level_descrs, const Config& config,
          std::mt19937_64& async_gen, std::vector<T>& data) {
  auto comm = level_descrs->initComm();
  const size_t nprocs = comm.getSize();

  config.tracker.local_sort_t.start(comm);

  if (config.use_ips4o) {
    // Slow on juqueen
    ips4o::sort(data.begin(), data.end(), config.comp);
  } else {
    // Fast on juqueen
    std::sort(data.begin(), data.end(), config.comp);
  }

  config.tracker.local_sort_t.stop();

  if (level_descrs->myNumLevels() > 0) {
    config.tracker.various_t.start(comm);

    const size_t np = data.size();
    size_t n;
    RBC::Allreduce(&np, &n, 1, Common::getMpiType(np), MPI_SUM, comm);

    const size_t max_np = tlx::div_ceil(n, nprocs);

    const size_t max_exp_np = max_np * (1. + config.max_epsilon);
    std::vector<T> tmp_data;
    tmp_data.reserve(max_exp_np);

    config.tracker.various_t.stop();

    // sort recursively
    const size_t level_idx = 0;
    // todo nprocs needed?
    Hss::_internal::KWaySort(nprocs, level_descrs,
                             config,
                             level_idx,
                             async_gen, data, tmp_data,
                             max_np,
                             max_np,
                             n,
                             config.max_epsilon, comm);
  }
}
}  // end namespace _internal

template <class T, class Tracker, class Comp, class AmsTags>
void sortTracker(MPI_Datatype mpi_type, std::vector<T>& data, int k,
                 std::mt19937_64& async_gen,
                 Tracker& tracker,
                 const RBC::Comm& comm,
                 Comp comp,
                 double imbalance,
                 Ams::DistributionStrategy distr,
                 bool use_deterministic_group_assignment,
                 bool keep_local_data,
                 bool use_ips4o,
                 bool use_two_tree) {
  Ams::_internal::RecDescrKway level_descrs(k, comm);
  _internal::Config<AmsTags, Comp, Tracker> config(mpi_type,
                                          tracker, comp, distr,
                                          use_deterministic_group_assignment,
                                          keep_local_data,
                                          use_ips4o,
                                          use_two_tree,
                                          imbalance - 1.0);
  _internal::sort(&level_descrs, config, async_gen, data);
}

template <class T, class Comp, class AmsTags>
void sort(MPI_Datatype mpi_type, std::vector<T>& data, int k,
          std::mt19937_64& async_gen,
          const RBC::Comm& comm,
          Comp comp,
          double imbalance,
          Ams::DistributionStrategy distr,
          bool use_deterministic_group_assignment,
          bool keep_local_data,
          bool use_ips4o,
          bool use_two_tree) {
  Ams::_internal::RecDescrKway level_descrs(k, comm);
  Ams::_internal::DummyTracker tracker;
  _internal::Config<AmsTags, Comp, Ams::_internal::DummyTracker> config(mpi_type,
                                                               tracker, comp, distr,
                                                               use_deterministic_group_assignment,
                                                               keep_local_data,
                                                               use_ips4o,
                                                               use_two_tree,
                                                               imbalance - 1.0);
  _internal::sort(&level_descrs, config, async_gen, data);
}

template <class T, class Tracker, class Comp, class AmsTags>
void sortTracker(MPI_Datatype mpi_type, std::vector<T>& data,
                 std::vector<size_t>& ks,
                 std::mt19937_64& async_gen,
                 Tracker& tracker,
                 const RBC::Comm& comm,
                 Comp comp,
                 double imbalance,
                 Ams::DistributionStrategy distr,
                 bool use_deterministic_group_assignment,
                 bool keep_local_data,
                 bool use_ips4o,
                 bool use_two_tree) {
  _internal::Config<AmsTags, Comp, Tracker> config(mpi_type,
                                          tracker, comp, distr,
                                          use_deterministic_group_assignment,
                                          keep_local_data,
                                          use_ips4o,
                                          use_two_tree,
                                          imbalance - 1.0);
  config.tracker.split_comm_t.start(comm);
  Ams::_internal::RecDescrKways level_descrs(comm, ks);
  config.tracker.split_comm_t.stop();
  _internal::sort(&level_descrs, config, async_gen, data);
}

template <class T, class Comp, class AmsTags>
void sort(MPI_Datatype mpi_type, std::vector<T>& data,
          std::vector<size_t>& ks,
          std::mt19937_64& async_gen,
          const RBC::Comm& comm,
          Comp comp,
          double imbalance,
          Ams::DistributionStrategy distr,
          bool use_deterministic_group_assignment,
          bool keep_local_data,
          bool use_ips4o,
          bool use_two_tree) {
  Ams::_internal::RecDescrKways level_descrs(comm, ks);
  Ams::_internal::DummyTracker tracker;
  _internal::Config<AmsTags, Comp, Ams::_internal::DummyTracker> config(mpi_type,
                                                               tracker, comp, distr,
                                                               use_deterministic_group_assignment,
                                                               keep_local_data,
                                                               use_ips4o,
                                                               use_two_tree,
                                                               imbalance - 1.0);
  _internal::sort(&level_descrs, config, async_gen, data);
}

template <class T, class Tracker, class Comp, class AmsTags>
void sortLevelTracker(MPI_Datatype mpi_type, std::vector<T>& data, int l,
                      std::mt19937_64& async_gen,
                      Tracker& tracker,
                      const RBC::Comm& comm,
                      Comp comp,
                      double imbalance,
                      Ams::DistributionStrategy distr,
                      bool use_deterministic_group_assignment,
                      bool keep_local_data,
                      bool use_ips4o,
                      bool use_two_tree) {
  const auto k = 1 << tlx::div_ceil(tlx::integer_log2_ceil(comm.getSize()), l);

  _internal::Config<AmsTags, Comp, Tracker> config(mpi_type,
                                          tracker, comp, distr,
                                          use_deterministic_group_assignment,
                                          keep_local_data,
                                          use_ips4o,
                                          use_two_tree,
                                          imbalance - 1.0);
  config.tracker.split_comm_t.start(comm);
  Ams::_internal::RecDescrKway level_descrs(comm, k);
  config.tracker.split_comm_t.stop();
  _internal::sort(&level_descrs, config, async_gen, data);
}

template <class T, class Comp, class AmsTags>
void sortLevel(MPI_Datatype mpi_type, std::vector<T>& data, int l,
               std::mt19937_64& async_gen,
               const RBC::Comm& comm,
               Comp comp,
               double imbalance,
               Ams::DistributionStrategy distr,
               bool use_deterministic_group_assignment,
               bool keep_local_data,
               bool use_ips4o,
               bool use_two_tree) {
  const auto k = 1 << tlx::div_ceil(tlx::integer_log2_ceil(comm.getSize()), l);

  Ams::_internal::RecDescrKway level_descrs(k, comm);
  Ams::_internal::DummyTracker tracker;
  _internal::Config<AmsTags, Comp, Ams::_internal::DummyTracker> config(mpi_type,
                                                               tracker, comp, distr,
                                                               use_deterministic_group_assignment,
                                                               keep_local_data,
                                                               use_ips4o,
                                                               use_two_tree,
                                                               imbalance - 1.0);
  _internal::sort(&level_descrs, config, async_gen, data);
}

template <class T, class Tracker, class Comp, class AmsTags>
void sortTracker(MPI_Datatype mpi_type, std::vector<T>& data, int k,
                 std::mt19937_64& async_gen,
                 Tracker& tracker,
                 MPI_Comm comm,
                 Comp comp,
                 double imbalance,
                 Ams::DistributionStrategy distr,
                 bool use_deterministic_group_assignment,
                 bool keep_local_data,
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
              distr,
              use_deterministic_group_assignment,
              keep_local_data,
              use_ips4o,
              use_two_tree);
}

template <class T, class Comp, class AmsTags>
void sort(MPI_Datatype mpi_type, std::vector<T>& data, int k,
          std::mt19937_64& async_gen,
          MPI_Comm comm,
          Comp comp,
          double imbalance,
          Ams::DistributionStrategy distr,
          bool use_deterministic_group_assignment,
          bool keep_local_data,
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
       distr,
       use_deterministic_group_assignment,
       keep_local_data,
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
                 Ams::DistributionStrategy distr,
                 bool use_deterministic_group_assignment,
                 bool keep_local_data,
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
              distr,
              use_deterministic_group_assignment,
              keep_local_data,
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
          Ams::DistributionStrategy distr,
          bool use_deterministic_group_assignment,
          bool keep_local_data,
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
       distr,
       use_deterministic_group_assignment,
       keep_local_data,
       use_ips4o,
       use_two_tree);
}

template <class T, class Tracker, class Comp, class AmsTags>
void sortLevelTracker(MPI_Datatype mpi_type, std::vector<T>& data, int l,
                      std::mt19937_64& async_gen,
                      Tracker& tracker,
                      MPI_Comm comm,
                      Comp comp,
                      double imbalance,
                      Ams::DistributionStrategy distr,
                      bool use_deterministic_group_assignment,
                      bool keep_local_data,
                      bool use_ips4o,
                      bool use_two_tree) {
  RBC::Comm rcomm;
  RBC::Create_Comm_from_MPI(comm, &rcomm);

  sortLevelTracker<T, Tracker, Comp, AmsTags>(mpi_type,
                   data,
                   l,
                   async_gen,
                   tracker,
                   rcomm,
                   comp,
                   imbalance,
                   distr,
                   use_deterministic_group_assignment,
                   keep_local_data,
                   use_ips4o,
                   use_two_tree);
}

template <class T, class Comp, class AmsTags>
void sortLevel(MPI_Datatype mpi_type, std::vector<T>& data, int l,
               std::mt19937_64& async_gen,
               MPI_Comm comm,
               Comp comp,
               double imbalance,
               Ams::DistributionStrategy distr,
               bool use_deterministic_group_assignment,
               bool keep_local_data,
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
            distr,
            use_deterministic_group_assignment,
            keep_local_data,
            use_ips4o,
            use_two_tree);
}
}  // end namespace Hss
