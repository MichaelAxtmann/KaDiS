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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include <tlx/math.hpp>

#include "EqualBucketBound.hpp"

namespace Overpartition {
namespace _internal {
std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t> >
EqualBucketBound::assignElementsToGroup(const std::vector<size_t>& loc_bucket_sizes,
                                        std::vector<size_t> loc_bucket_sizes_exscan,
                                        std::vector<size_t> glob_bucket_sizes,
                                        const std::vector<size_t>& group_sizes,
                                        const size_t pe_capacity) {
  const size_t num_buckets = loc_bucket_sizes.size();
  const size_t num_groups = group_sizes.size();

  /* Calculates ranges of local data in the distributed data array.
   * The distributed data array already contains the partitioned data.
   *
   * Example:
   * 4 Buckets and 3 PEs
   * Global partitiones: 0 - 10, 10 - 40, 40 - 46, 46 - 51
   * PE 1: (0, 3), (10, 22), (40, 42), (46, 49)
   * PE 2: (3, 7), (22, 34), (42, 45), (49, 50)
   * PE 3: (7, 10), (34, 40), (45, 46), (50, 51)
   */
  auto data_ranges =
    [&loc_bucket_sizes, &loc_bucket_sizes_exscan, &glob_bucket_sizes, &num_buckets]() {
      std::vector<std::pair<size_t, size_t> > glob_bucket_sizes_exscan(
        num_buckets);
      size_t glob_offset = 0;
      for (size_t i = 0; i != num_buckets; ++i) {
        glob_bucket_sizes_exscan[i] = std::pair<size_t, size_t>{
          glob_offset + loc_bucket_sizes_exscan[i],
          glob_offset + loc_bucket_sizes_exscan[i] +
          loc_bucket_sizes[i]
        };
        glob_offset += glob_bucket_sizes[i];
      }

      return glob_bucket_sizes_exscan;
    } ();
  auto data_range = data_ranges.begin();

  // Number of elements which this PE sends to a group.
  std::vector<size_t> loc_group_el_cnts(num_groups, 0);
  // Number of elements which this PE and PEs with smaller index send to a group.
  std::vector<size_t> loc_group_el_cnts_scan(num_groups, 0);
  // Number of elements which all PEs send to a group.
  std::vector<size_t> glob_group_el_cnts(num_groups, 0);

  // Number of global elements moved to the current group.
  size_t num_moved_elements = 0;

  // Assigns el_cnt global elements to group group_idx.
  auto move_to_group =
    [&num_moved_elements,
     &loc_group_el_cnts,           // Groups which get local elements assigned
     &glob_group_el_cnts,          // Groups which get global elements assigned
     &data_range,          // Current data range which we want to assign to a group.
     &data_ranges](size_t el_cnt,
                   size_t group_idx) -> size_t {
      size_t locally_moved_el_cnt = 0;
      while (data_range != data_ranges.end() &&
             data_range->first < num_moved_elements + el_cnt) {
        if (data_range->first == data_range->second) {
          ++data_range;
        } else {
          auto num_els = std::min(num_moved_elements + el_cnt,
                                  data_range->second) -
                         data_range->first;
          loc_group_el_cnts[group_idx] += num_els;
          data_range->first += num_els;
          locally_moved_el_cnt += num_els;
        }
      }

      // Update global group size
      glob_group_el_cnts[group_idx] += el_cnt;

      num_moved_elements += el_cnt;
      return locally_moved_el_cnt;
    };

  /* Loop assigns buckets or parts of buckets (if equal buckets) to groups.
   * Assigned buckets
   * are passed to the method move_to_group which extracts the local elements.
   */

  // Number of unassigned global elements of current bucket.
  auto glob_bucket_size = glob_bucket_sizes.begin();

  // Number of unassigned elements of current bucket on PEs with smaller index.
  auto loc_bucket_size_exscan = loc_bucket_sizes_exscan.begin();

  bool is_equal_bucket = false;
  for (size_t group_idx = 0; group_idx < num_groups; group_idx++) {
    // 'load' tracks total number of assigned elements to the current group.
    size_t load = 0;
    const size_t group_capacity = pe_capacity * group_sizes[group_idx];
    while (glob_bucket_size != glob_bucket_sizes.end()) {
      if (load + *glob_bucket_size <= group_capacity) {
        // Add all remaining elements of the current bucket
        // to the current group.
        load += *glob_bucket_size;
        size_t locally_moved_el_cnt = move_to_group(*glob_bucket_size, group_idx);
        glob_bucket_size++;

        // Add all elements of PEs with smaller index to prefix sum.
        loc_group_el_cnts_scan[group_idx] += *loc_bucket_size_exscan + locally_moved_el_cnt;
        loc_bucket_size_exscan++;

        // Every second bucket is an equal bucket if we use equal
        // buckets.
        is_equal_bucket = !is_equal_bucket;
      } else if (is_equal_bucket) {
        // Calculate number of elements which are added to the current
        // group and removed from the current bucket.
        const size_t new_load = group_capacity - load;

        // Add load
        load += new_load;
        // Remove load from bucket
        *glob_bucket_size -= new_load;
        assert(*glob_bucket_size > 0);

        // move new_load elements of bucket (glob_bucket_size -
        // glob_bucket_sizes.begin()) to group group_idx
        size_t locally_moved_el_cnt = move_to_group(new_load, group_idx);

        // Number of elements added by PEs with smaller index.
        size_t small_pe_el_cnt = std::min<size_t>(new_load, *loc_bucket_size_exscan);
        *loc_bucket_size_exscan -= small_pe_el_cnt;
        // Add all elements of PEs with smaller index and own elements to prefix sum.
        loc_group_el_cnts_scan[group_idx] += small_pe_el_cnt + locally_moved_el_cnt;

        break;
      } else {
        // No equal bucket and current bucket does not fit into current
        // group
        break;
      }
    }
  }

  using ReturnType = std::tuple<std::vector<size_t>,
                                std::vector<size_t>,
                                std::vector<size_t> >;
  return ReturnType{ loc_group_el_cnts, loc_group_el_cnts_scan, glob_group_el_cnts };
}

EqualBucketBound EqualBucketBound::getBound(const std::vector<size_t>& group_sizes,
                                            std::vector<size_t> bucket_sizes,
                                            const size_t pe_capacity) {
  // maximum amount of load which has been assigned to a bucket
  size_t upper_bound_candidate = 0;
  // minimum amount of load which is required to move the first bucket
  // of a group to the previous group
  size_t lower_bound_candidate = std::numeric_limits<size_t>::max();

  auto bucket_size = bucket_sizes.begin();

  // Equal bucket handling
  bool is_equal_bucket = false;
  size_t num_pes_covered_by_equal_bucket = 0;

  for (size_t group_idx = 0; group_idx < group_sizes.size(); group_idx++) {
    bool group_cloeses_at_least_one_bucket = false;
    bool group_ends_on_open_equal_bucket = false;
    size_t group_load_after_last_finished_bucket = 0;
    size_t group_capacity = pe_capacity * group_sizes[group_idx];
    size_t load = 0;

    while (bucket_size != bucket_sizes.end()) {
      if (load + *bucket_size <= group_capacity) {
        load += *bucket_size;
        group_load_after_last_finished_bucket = load;
        group_cloeses_at_least_one_bucket = true;
        group_ends_on_open_equal_bucket = false;
        bucket_size++;
        // Every second bucket is an equal bucket if we use equal
        // buckets.
        is_equal_bucket = !is_equal_bucket;
      } else if (is_equal_bucket) {
        const size_t new_load = group_capacity - load;
        load += new_load;
        *bucket_size -= new_load;
        assert(*bucket_size > 0);
        // Bucket still contains elements which belong into the next
        // group.
        group_ends_on_open_equal_bucket = true;
        break;
      } else {
        // No equal bucket and current bucket does not fit into current
        // group
        break;
      }
    }

    // as there is still a bucket left, one could add the bucket
    // to the current group, if pe_capacity would be higher.
    if (bucket_size != bucket_sizes.end()) {
      lower_bound_candidate = std::min(
        lower_bound_candidate,
        pe_capacity + tlx::div_ceil(
          (*bucket_size - (group_capacity - load)),
          (num_pes_covered_by_equal_bucket + group_sizes[group_idx])));
    }

    if (group_cloeses_at_least_one_bucket) {
      const size_t curr_upper_bound_candidate =
        pe_capacity -
        (group_capacity - group_load_after_last_finished_bucket) /
        (num_pes_covered_by_equal_bucket + group_sizes[group_idx]);
      upper_bound_candidate =
        std::max(curr_upper_bound_candidate, upper_bound_candidate);
    }

    if (group_ends_on_open_equal_bucket) {
      num_pes_covered_by_equal_bucket += group_sizes[group_idx];
    } else {
      num_pes_covered_by_equal_bucket = 0;
    }
  }

  // the algorithm was able to assign all bucket_sizes.size() to PEs
  if (bucket_size == bucket_sizes.end()) {
    return EqualBucketBound(bound_type::upper_bound, upper_bound_candidate);
  } else {
    return EqualBucketBound(bound_type::lower_bound, lower_bound_candidate);
  }
}
}  // end namespace _internal
}  // end namespace Overpartition
