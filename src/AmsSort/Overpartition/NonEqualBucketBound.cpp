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
#include <vector>

#include <tlx/math.hpp>

#include "NonEqualBucketBound.hpp"

namespace Overpartition {
namespace _internal {
std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t> >
NonEqualBucketBound::assignElementsToGroup(const std::vector<size_t>& loc_bsizes,
                                           const std::vector<size_t>& loc_bsizes_exscan,
                                           const std::vector<size_t>& glob_bsizes,
                                           const std::vector<size_t>& group_sizes,
                                           const size_t pe_capacity) {
  const size_t num_groups = group_sizes.size();
  const size_t num_buckets = glob_bsizes.size();

  std::vector<size_t> loc_gsizes(num_groups, 0);
  std::vector<size_t> glob_gsizes(num_groups, 0);
  std::vector<size_t> loc_gsizes_scan(num_groups, 0);

  // Iterate over groups and assign as many buckets as possible.
  size_t b = 0;
  for (size_t g = 0; g < num_groups; g++) {
    while (b < num_buckets &&
           glob_gsizes[g] + glob_bsizes[b] <= group_sizes[g] * pe_capacity) {
      loc_gsizes[g] += loc_bsizes[b];
      loc_gsizes_scan[g] += loc_bsizes_exscan[b] + loc_bsizes[b];
      glob_gsizes[g] += glob_bsizes[b];
      b++;
    }
  }

  // All buckets must be assigned.
  assert(b == num_buckets);
  using return_type = std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t> >;
  return return_type{ loc_gsizes, loc_gsizes_scan, glob_gsizes };
}

NonEqualBucketBound NonEqualBucketBound::getBound(const std::vector<size_t>& group_sizes,
                                                  const std::vector<size_t>& glob_bucket_sizes,
                                                  const size_t pe_capacity) {
  // maximum amount of load which has been assigned to a partition
  size_t max_load = 0;
  size_t act_group_load = 0;
  size_t next_load = std::numeric_limits<size_t>::max();

  const size_t group_cnt = group_sizes.size();

  const size_t bucket_cnt = glob_bucket_sizes.size();
  size_t bucket_idx = 0;

  for (size_t group_idx = 0; group_idx < group_cnt; group_idx++) {
    while (bucket_idx < bucket_cnt &&
           act_group_load + glob_bucket_sizes[bucket_idx] <= group_sizes[group_idx] * pe_capacity) {
      act_group_load += glob_bucket_sizes[bucket_idx];
      bucket_idx++;
    }

    max_load = std::max(max_load, tlx::div_ceil(act_group_load, group_sizes[group_idx]));

    // as there is still a partition left, one could add the partition
    // to this pe, if maxload would be higher.
    if (bucket_idx < bucket_cnt) {
      next_load =
        std::min(next_load, tlx::div_ceil(act_group_load + glob_bucket_sizes[bucket_idx],
                                          group_sizes[group_idx]));
    }

    act_group_load = 0;
  }

  // the algorithm was able to assign all glob_bucket_sizes.size() to PEs
  if (bucket_idx == bucket_cnt) {
    return NonEqualBucketBound(bound_type::upper_bound, max_load);
  } else {
    return NonEqualBucketBound(bound_type::lower_bound, next_load);
  }
}
}  // end namespace _internal
}  // end namespace Overpartition
