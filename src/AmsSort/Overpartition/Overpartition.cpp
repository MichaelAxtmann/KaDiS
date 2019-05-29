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
#include <cstdint>
#include <numeric>
#include <tuple>
#include <vector>

#include "../../Tools/CommonMpi.hpp"
#include "./EqualBucketBound.hpp"
#include "./NonEqualBucketBound.hpp"
#include "Overpartition.hpp"

#include <RBC.hpp>

namespace Overpartition {
namespace _internal {
size_t getMyTestBound(size_t lower_pe_bound, size_t upper_pe_bound, size_t myrank,
                      size_t nprocs) {
  const double range_width = (upper_pe_bound - lower_pe_bound) / static_cast<double>(nprocs);
  const size_t endpoint = lower_pe_bound + range_width * (myrank + 1);
  return endpoint;
}

template <class Bound>
std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t> >
assignElementsToGroups(const std::vector<size_t> group_sizes, const
                       std::vector<size_t>& loc_bucket_sizes, const
                       std::vector<size_t>& loc_bucket_sizes_exscan, const
                       std::vector<size_t>& glob_bucket_sizes, double expected_imbalance,
                       size_t* it_cnt, const RBC::Comm& comm) {
  /* Calculate initial lower and upper bounds for the binary search */

  // calculate initial lower bound
  const size_t total_el_cnt = std::accumulate(
    glob_bucket_sizes.begin(), glob_bucket_sizes.end(), static_cast<size_t>(0));
  // size_t lower_pe_bound = tlx::div_ceil(total_el_cnt, num_groups);
  size_t lower_pe_bound = tlx::div_ceil(total_el_cnt, comm.getSize());

  // Calculate initial upper bound depending on expected imbalance
  // size_t upper_bound = lower_pe_bound * (1. + expected_imbalance);
  size_t upper_pe_bound = lower_pe_bound * (1. + expected_imbalance);
  expected_imbalance += expected_imbalance;
  // Compute a bound for the passed load.

  bool valid_init_bound = false;
  do {
    size_t local_end_point =
      _internal::getMyTestBound(lower_pe_bound, upper_pe_bound, comm.getRank(), comm.getSize());
    Bound b = Bound::getBound(group_sizes, glob_bucket_sizes, local_end_point);

    // if (b.getBoundType() == Bound::bound_type::upper_bound) {
    //     assert(ValidNonEqualBucketBound(
    //                comm, group_sizes,
    //                loc_bucket_sizes,
    //                loc_bucket_sizes_exscan,
    //                glob_bucket_sizes,
    //                upper_pe_bound));
    // }

    // diffs[0] is defined by the shift of lower_pe_bound which is allowed
    // based on
    // the local_end_point.
    // diffs[1] is defined by the shift of upper_pe_bound which is allowed
    // based on
    // the local_end_point.
    int64_t diffs[2] = { -1, -1 };
    int64_t aggr_diffs[2] = { -1, -1 };
    if (b.getBoundType() == Bound::bound_type::lower_bound) {
      diffs[0] = b.getBoundValue() - lower_pe_bound;
    } else {
      diffs[1] = upper_pe_bound - b.getBoundValue();
    }
    // Aggregate shifts to get the maximal shift of lower_pe_bound and
    // upper_pe_bound.
    RBC::Allreduce(diffs, aggr_diffs, 2, Common::getMpiType(diffs), MPI_MAX, comm);

    if (aggr_diffs[1] == -1) {
      // No PE found a upper bound.

      lower_pe_bound = upper_pe_bound + 1;
      upper_pe_bound = upper_pe_bound + std::max<size_t>(upper_pe_bound * expected_imbalance, 1l);
    } else {
      // We found an initial upper bound.

      // Apply shifts.
      lower_pe_bound = lower_pe_bound + aggr_diffs[0];
      upper_pe_bound = upper_pe_bound - aggr_diffs[1];
      valid_init_bound = true;
    }

    *it_cnt += 1;
  } while (!valid_init_bound);

  /* Parallel binary search of optimal load */

  // If upper_pe_bound == lower_pe_bound, we have found the optimal load.
  // By invariant upper_pe_bound is a feasible load.
  // By invariant there exists no feasible load which is lower than
  // lower_pe_bound.
  // As soon as lower_pe_bound == upper_pe_bound, we found the minimum feasible
  // load.
  while (upper_pe_bound > lower_pe_bound) {
    assert(upper_pe_bound
           >= Bound::getBound(group_sizes, glob_bucket_sizes, upper_pe_bound).getBoundValue());
    size_t local_end_point =
      _internal::getMyTestBound(lower_pe_bound, upper_pe_bound, comm.getRank(), comm.getSize());
    const Bound b = Bound::getBound(group_sizes, glob_bucket_sizes, local_end_point);
    *it_cnt += 1;
    // diffs[0] is defined by the shift of lower_pe_bound which is
    // allowed based on the local_end_point.
    // diffs[1] is defined by the shift of upper_pe_bound which is
    // allowed based on the local_end_point.
    int64_t diffs[2] = { 0, 0 };
    int64_t aggr_diffs[2] = { 0, 0 };
    if (b.getBoundType() == Bound::bound_type::lower_bound) {
      // todo simplify: do not exchange differences
      diffs[0] = b.getBoundValue() - lower_pe_bound;
    } else {
      diffs[1] = upper_pe_bound - b.getBoundValue();
    }
    assert(diffs[0] <= static_cast<int64_t>(upper_pe_bound) - static_cast<int64_t>(lower_pe_bound));
    assert(diffs[1] <= static_cast<int64_t>(upper_pe_bound) - static_cast<int64_t>(lower_pe_bound));
    // Aggregate shifts to get the maximal shift of lower_pe_bound and
    // upper_pe_bound.
    RBC::Allreduce(diffs, aggr_diffs, 2, Common::getMpiType(diffs), MPI_MAX, comm);
    // Apply shifts.
    lower_pe_bound = lower_pe_bound + aggr_diffs[0];
    upper_pe_bound = upper_pe_bound - aggr_diffs[1];
  }

  /* Calculate assignments of buckets to groups */
  return Bound::assignElementsToGroup(
    loc_bucket_sizes, loc_bucket_sizes_exscan, glob_bucket_sizes,
    group_sizes, lower_pe_bound);
}
}  // end namespace _internal

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t> >
assignElementsToGroups(const std::vector<size_t> group_sizes,
                       const std::vector<size_t>& loc_bucket_sizes,
                       const std::vector<size_t>& loc_bucket_sizes_exscan,
                       const std::vector<size_t>& glob_bucket_sizes,
                       double expected_imbalance,
                       bool use_equal_buckets,
                       size_t* it_cnt, const RBC::Comm& comm) {
  if (use_equal_buckets) {
    return _internal::assignElementsToGroups<_internal::EqualBucketBound>(
      group_sizes, loc_bucket_sizes, loc_bucket_sizes_exscan, glob_bucket_sizes,
      expected_imbalance, it_cnt, comm);
  } else {
    return _internal::assignElementsToGroups<_internal::NonEqualBucketBound>(
      group_sizes, loc_bucket_sizes, loc_bucket_sizes_exscan, glob_bucket_sizes,
      expected_imbalance, it_cnt, comm);
  }
}
}  // end namespace Overpartition
