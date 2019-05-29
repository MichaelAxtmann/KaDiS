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

#include <cstddef>
#include <tuple>
#include <vector>

namespace Overpartition {
namespace _internal {
/** Left or right bound.
 *
 * EqualBucketBound determines a left or right bound calculated based on a specific
 * maximum accepted load.
 */
class EqualBucketBound {
 public:
  enum bound_type { lower_bound, upper_bound };

  EqualBucketBound(bound_type type, size_t value) :
    type(type),
    value_(value) { }

  bound_type getBoundType() const { return type; }
  size_t getBoundValue() const { return value_; }

  static std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t> >
  assignElementsToGroup(const std::vector<size_t>& loc_bucket_sizes,
                        std::vector<size_t> loc_bucket_sizes_exscan,
                        std::vector<size_t> glob_bucket_sizes,
                        const std::vector<size_t>& group_sizes,
                        const size_t pe_capacity);

/* @brief Assigns elements of buckets to groups and calculates an upper or lower
 *bound.
 *
 * Assigns elements of buckets to groups. If all elements has been assigned, the
 *method
 * returns a valid upper bound which is smaller or equal to the value of
 *capacity.
 * If the assignment is not able to assign all elements, the method returns a
 *lower bound.
 * The target is to calculate a upper bound which is as small as possible or a
 *lower bound
 * which is as large as possible.
 *
 * Lower bound calculation:
 * The method calculates multiple lower bound candidates.
 * The smallest candidate will be the final lower bound.
 * Algorithm:
 * We calculate a new candidate for each group. If the group is full,
 * we add the next, unassigned bucket (could also be a partial bucket of an
 *equal bucket)
 * to the group and increase the bound as few as possible.
 * To do so, we count the number of previous groups which contain an equal
 *bucket
 * wrapping into the group before.
 * We have to increase the bound by the size of the bucket which we add the the
 *current group,
 * scaled by the number of previous buckets containing an equal bucket wrap.
 *
 * Upper bound:
 * The method calculates multiple upper bound candidates.
 * The largest candidate will be the final upper bound.
 * Algorithm:
 * We calculate a new candidate for each group that contains the last element of
 *at least one bucket.
 * If a group does not contain such elements, there is no bucket assigned to
 *this group
 * which could hit the bound.
 * To compute a new upper bound, we decrease the capacity until the uppermost
 *bucket hits the capacity.
 * We also take care of previous groups if those groups contain equal buckets
 * wrapping into the group before.
 *
 * @param group_sizes Number of PEs in each group.
 */
  static EqualBucketBound getBound(const std::vector<size_t>& group_sizes,
                                   std::vector<size_t> bucket_sizes,
                                   const size_t pe_capacity);

  // an upper bound is feasible anyway. If this Bound is a lower bound,
  // we are not able to make a statement
  bool isFeasableBound() const { return type == bound_type::upper_bound; }

 private:
  bound_type type;
  size_t value_;
};
}  // end namespace _internal
}  // end namespace Overpartition
