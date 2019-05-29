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
#include <cassert>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "tlx/algorithm.hpp"
#include "tlx/math.hpp"

#include "../Tools/Common.hpp"
#include "TieBreaker.hpp"

namespace Ams {
namespace _internal {
template <typename T>
class DecisionTree {
 public:
  static void build(T* splitter_tree, const T* samples,
                    size_t num_splitters) {
    recurse(splitter_tree, num_splitters, samples, samples + num_splitters, 1);
  }

 private:
  static T recurse(T* tree, const size_t num_splitters,
                   const T* lo, const T* hi, unsigned int treeidx) {
    // Pick middle element as splitter.
    const T* mid = lo + (hi - lo) / 2;
    T key = tree[treeidx] = *mid;
    // New boundaries.
    const T* midlo = mid;
    const T* midhi = mid + 1;
    if (2 * treeidx < num_splitters) {
      recurse(tree, num_splitters, lo, midlo, 2 * treeidx + 0);
      return recurse(tree, num_splitters, midhi, hi, 2 * treeidx + 1);
    } else {
      return key;
    }
  }
};

template <class T, class Comp, size_t MAX_LOG_BUCKET = 8, size_t UNROLL_CLASSIFIER = 7>
class KWayPartitioner {
  using oracle_type = uint32_t;

 public:
  explicit KWayPartitioner(Comp comp) :
    comp_(comp)
  { }

  std::vector<size_t> partition(std::vector<T>& a, std::vector<T>& a_tmp,
                                T* input_splitter, const size_t num_buckets) const {
    // We may need to create a larger splitter array later.
    T* splitter = input_splitter;
    std::unique_ptr<T[]> splitter_uptr(nullptr);

    const size_t log_num_buckets = tlx::integer_log2_ceil(num_buckets);
    const size_t num_buckets_ceiled = 1 << log_num_buckets;
    size_t num_splitters_ceiled = num_buckets_ceiled - 1;

    // As the algorithm only handles bucket sizes which are a pow of two
    // we have do adapt the input.
    const bool num_buckets_is_power_of_two = tlx::is_power_of_two(num_buckets);
    if (!num_buckets_is_power_of_two) {
#ifndef NDEBUG
      assert(tlx::is_power_of_two(num_buckets_ceiled));
      size_t logk_algo_test = tlx::integer_log2_floor(num_buckets_ceiled);
      assert(num_buckets_ceiled > num_buckets);
      // assert(logk_algo_test == logk + 1);
      assert(log_num_buckets == logk_algo_test);
#endif

      // Increase splitter array.
      splitter_uptr = std::make_unique<T[]>(num_splitters_ceiled);
      splitter = splitter_uptr.get();
      // Copy splitters to new array.
      std::copy(input_splitter, input_splitter + num_buckets - 1,
                splitter);
      // Fill remaining splitters with dummy splitters.
      for (auto s_algo_it = splitter + num_buckets - 1;
           s_algo_it != splitter + num_splitters_ceiled; s_algo_it++) {
        *s_algo_it = splitter[num_buckets - 2];
      }
    }

    // Partition a and store results in bucket_sizes.

    a_tmp.resize(a.size());

    std::unique_ptr<T[]> tree_splitter(new T[num_buckets_ceiled]);
    std::unique_ptr<T[]> tree_splitter_sorted(new T[num_buckets_ceiled]);
    std::unique_ptr<oracle_type[]> oracle(new oracle_type[a.size()]);

    std::vector<size_t> bucket_sizes(num_buckets_ceiled);
    std::vector<size_t> tmp_bucket_sizes(num_buckets_ceiled);

    bucket_sizes[0] = a.size();

    recurse(a, a_tmp, splitter,
            bucket_sizes, tmp_bucket_sizes, num_buckets_ceiled, num_buckets_ceiled,
            log_num_buckets, tree_splitter.get(), tree_splitter_sorted.get(), oracle);

    if (!num_buckets_is_power_of_two) {
      // As the elements which belong into the last used
      // bucket are stored in the last dummy bucket, we
      // have to correct the bucket sizes.
      using bs_type = decltype(bucket_sizes)::value_type;
      using bs_it_type = decltype(bucket_sizes)::const_iterator;
      bucket_sizes[num_buckets - 1] =
        std::accumulate<bs_it_type, bs_type>(
          bucket_sizes.begin() + num_buckets,
          bucket_sizes.begin() + num_buckets_ceiled,
          0);

      // Remove dummy buckets.
      bucket_sizes.resize(num_buckets);

      // sum of remaining buckets should be the amount of passed elements.
      assert(std::accumulate(bucket_sizes.begin(), bucket_sizes.end(), size_t{ 0 }) ==
             a.size());
      assert(input_splitter != splitter);
    }

    return bucket_sizes;
  }

 private:
  void distribute(const T* a_begin,
                  const T* const a_end,
                  T* const a_tmp,
                  const oracle_type* oracle,
                  size_t* const bucket_sizes,
                  const oracle_type num_buckets) const {
    constexpr const int kUnroll = UNROLL_CLASSIFIER;

    std::unique_ptr<size_t[]> psum_bucket_sizes(new size_t[num_buckets + 1]);
    tlx::exclusive_scan(bucket_sizes, bucket_sizes + num_buckets,
                        psum_bucket_sizes.get(), oracle_type{ 0 }, std::plus<>{ });

    assert(psum_bucket_sizes[num_buckets] == static_cast<size_t>(a_end - a_begin));

    ASSUME_NOT(a_end < a_begin);

    if (kUnroll <= a_end - a_begin) {
      ASSUME_NOT(a_begin + kUnroll > a_end);
      for (auto cutoff = a_end - kUnroll;
           a_begin <= cutoff;
           a_begin += kUnroll, oracle += kUnroll) {
        for (int i = 0; i < kUnroll; ++i) {
          a_tmp[psum_bucket_sizes[oracle[i]]++] = a_begin[i];
        }
      }
    }

    for ( ; a_begin != a_end; ++a_begin, ++oracle) {
      a_tmp[psum_bucket_sizes[*oracle]++] = *a_begin;
    }
  }

  /**
   * Classifies all elements using a callback.
   */
  void classify(T* begin, T* end, T* splitter, oracle_type* oracle,
                size_t* bucket_sizes, size_t log_num_buckets) const {
    static_assert(MAX_LOG_BUCKET <= 8, "Too many buckets.");
    switch (log_num_buckets) {
      case 1: classifyUnrolled<1>(begin, end, splitter,
                                  oracle, bucket_sizes);
        break;
      case 2: classifyUnrolled<2>(begin, end, splitter,
                                  oracle, bucket_sizes);
        break;
      case 3: classifyUnrolled<3>(begin, end, splitter,
                                  oracle, bucket_sizes);
        break;
      case 4: classifyUnrolled<4>(begin, end, splitter,
                                  oracle, bucket_sizes);
        break;
      case 5: classifyUnrolled<5>(begin, end, splitter,
                                  oracle, bucket_sizes);
        break;
      case 6: classifyUnrolled<6>(begin, end, splitter,
                                  oracle, bucket_sizes);
        break;
      case 7: classifyUnrolled<7>(begin, end, splitter,
                                  oracle, bucket_sizes);
        break;
      case 8: classifyUnrolled<8>(begin, end, splitter,
                                  oracle, bucket_sizes);
        break;
    }
  }

  /**
   * Classifies all elements using a callback.
   */
  template <int kLogBuckets>
  void classifyUnrolled(T* begin, const T* end, T* splitter,
                        oracle_type* oracle, size_t* bucket_sizes) const {
    using bucket_type = size_t;
    constexpr const bucket_type kNumBuckets = 1l << kLogBuckets;
    constexpr const int kUnroll = UNROLL_CLASSIFIER;

    bucket_type b[kUnroll];
    ASSUME_NOT(begin > end);

    if (begin + kUnroll <= end) {
      ASSUME_NOT(begin + kUnroll > end);

      for (auto cutoff = end - kUnroll; begin <= cutoff; begin += kUnroll, oracle += kUnroll) {
        for (int i = 0; i < kUnroll; ++i)
          b[i] = 1;

        for (int l = 0; l < kLogBuckets; ++l)
          for (int i = 0; i < kUnroll; ++i)
            b[i] = 2 * b[i] + comp_(splitter[b[i]], begin[i]);

        for (int i = 0; i < kUnroll; ++i) {
          const auto bucket_id = b[i] - kNumBuckets;
          *(oracle + i) = bucket_id;
          ++bucket_sizes[bucket_id];
        }
      }
    }

    ASSUME_NOT(begin > end);
    for ( ; begin != end; ++begin, ++oracle) {
      bucket_type b = 1;
      for (int l = 0; l < kLogBuckets; ++l)
        b = 2 * b + comp_(splitter[b], *begin);
      const auto bucket_id = b - kNumBuckets;
      *oracle = bucket_id;
      ++bucket_sizes[bucket_id];
    }
  }

  void recurse(std::vector<T>& a, std::vector<T>& a_tmp, T* splitters,
               std::vector<size_t>& bucket_sizes, std::vector<size_t>& tmp_bucket_sizes,
               const size_t num_init_buckets, const size_t num_buckets, const size_t
               log_num_buckets,
               T* tree_splitter, T* tree_splitter_sorted,
               std::unique_ptr<oracle_type[]>& oracle) const {
    assert(num_init_buckets % num_buckets == 0);

    // Reset values.
    std::fill(tmp_bucket_sizes.begin(), tmp_bucket_sizes.begin() + num_init_buckets, 0);

    size_t bucket_offset = 0;
    assert(num_init_buckets % num_buckets == 0);

    // Calculate actual number of buckets at this level.
    // Create buckets of (almost) the same size.
    const size_t num_levels = tlx::div_ceil(log_num_buckets, MAX_LOG_BUCKET);
    assert(num_levels > 0);
    const size_t log_num_local_buckets =
      tlx::div_ceil(log_num_buckets, num_levels);
    assert(log_num_local_buckets <= MAX_LOG_BUCKET);
    assert(log_num_local_buckets > 0);
    const size_t num_local_buckets = 1 << log_num_local_buckets;

    // Partition each stripe.
    size_t num_stripes = num_init_buckets / num_buckets;
    for (size_t stripe_idx = 0; stripe_idx != num_stripes; ++stripe_idx) {
      assert(a.size() >= bucket_offset + bucket_sizes[stripe_idx]);
      auto begin_stripe = a.data() + bucket_offset;
      auto end_stripe = begin_stripe + bucket_sizes[stripe_idx];
      auto begin_tmp_stripe = a_tmp.data() + bucket_offset;
      auto stripe_splitters = splitters + stripe_idx * num_buckets;
      auto stripe_bucket_sizes =
        tmp_bucket_sizes.data() + stripe_idx * num_local_buckets;

      /* Create subtree and sorted splitter array */

      const size_t num_stripe_buckets = num_buckets / num_local_buckets;
      if (num_stripe_buckets == 1) {
        DecisionTree<T>::build(tree_splitter, stripe_splitters,
                               num_local_buckets - 1);
      } else {
        for (size_t idx = 1; idx != num_local_buckets; ++idx) {
          tree_splitter_sorted[idx - 1] =
            stripe_splitters[idx * num_stripe_buckets - 1];
        }
        DecisionTree<T>::build(tree_splitter, tree_splitter_sorted,
                               num_local_buckets - 1);
      }

      classify(begin_stripe, end_stripe, tree_splitter, oracle.get(),
               stripe_bucket_sizes, log_num_local_buckets);
      distribute(begin_stripe, end_stripe, begin_tmp_stripe, oracle.get(),
                 stripe_bucket_sizes, num_local_buckets);

      bucket_offset += bucket_sizes[stripe_idx];
    }

    bucket_sizes.swap(tmp_bucket_sizes);

    a.swap(a_tmp);

    if (num_local_buckets != num_buckets) {
      assert(num_buckets % num_local_buckets == 0);
      recurse(a, a_tmp, splitters,
              bucket_sizes, tmp_bucket_sizes, num_init_buckets,
              num_buckets / num_local_buckets,
              log_num_buckets - log_num_local_buckets,
              tree_splitter, tree_splitter_sorted, oracle);
    }
  }

  Comp comp_;
};

template <class T, class Comp, size_t MAX_LOG_BUCKET = 8, size_t UNROLL_CLASSIFIER = 7>
class KWayPartitionerImplTieBreaker {
  using oracle_type = uint32_t;

 public:
  KWayPartitionerImplTieBreaker(Comp comp, int64_t local_index_begin, int64_t local_index_end) :
    comp_(comp),
    local_index_begin_(local_index_begin),
    local_index_end_(local_index_end)
  { }

  std::vector<size_t> partition(std::vector<T>& a, std::vector<T>& a_tmp,
                                TieBreaker<T>* input_splitter, const size_t num_buckets) const {
    // Remove duplicate splitters here, we will calculate the offsets
    // later.
    std::vector<T> unique_splitters = getUniqueValues(input_splitter, num_buckets - 1);
    const size_t num_unique_buckets = unique_splitters.size() + 1;

    const size_t log_num_buckets = tlx::integer_log2_ceil(num_unique_buckets);
    const size_t num_unique_buckets_ceiled = 1 << log_num_buckets;
    const size_t num_unique_splitters_ceiled = num_unique_buckets_ceiled - 1;

    const bool num_buckets_is_power_of_two = tlx::is_power_of_two(num_unique_buckets);
    // Fill additional splitter slots with dummies.
    if (!num_buckets_is_power_of_two) {
      while (unique_splitters.size() < num_unique_splitters_ceiled) {
        unique_splitters.push_back(unique_splitters.back());
      }
    }

    // Twice the number of buckets for equal buckets.
    std::vector<size_t> bucket_sizes(2 * num_unique_buckets_ceiled, 0);
    std::unique_ptr<oracle_type[]> oracle(new oracle_type[a.size()]);

    partition(a, a_tmp, bucket_sizes, input_splitter, unique_splitters, num_unique_buckets_ceiled,
              log_num_buckets, num_buckets, oracle);

    if (!num_buckets_is_power_of_two) {
      assert(std::accumulate(bucket_sizes.begin(), bucket_sizes.end(), size_t{ 0 })
             == a.size());
      // Move the number of the largest bucket to the last real bucket.
      size_t larger = 0, equal = 0;
      for (size_t i = num_unique_buckets; i < num_unique_buckets_ceiled; ++i) {
        larger += bucket_sizes[2 * i];
        equal += bucket_sizes[2 * i + 1];
      }

      bucket_sizes[2 * num_unique_buckets - 3] += equal;
      bucket_sizes[2 * num_unique_buckets - 2] += larger;
      bucket_sizes.resize(2 * num_unique_buckets);
      // sum of remaining buckets should be the amount of passed elements.
      assert(std::accumulate(bucket_sizes.begin(), bucket_sizes.end(), size_t{ 0 })
             == a.size());
    }

    // Now we need to undo the compaction of the splitter
    // array where we have filtered out duplicates. All
    // local splitters have their local index set to the
    // size of the equal bucket at the time they were
    // added into the equal bucket
    std::vector<size_t> real_bucket_sizes(num_buckets, 0);

    // For each unique splitter bucket, we set the bucket
    // size in the 'real_bucket_size' array to the number
    // of elements belonging to that unique_splitter
    // bucket.
    size_t j = 0;
    for (size_t i = 0; i < num_unique_buckets; ++i) {
      real_bucket_sizes[j] = bucket_sizes[2 * i];
      // skip same elements
      while (j < num_buckets - 1 && (input_splitter[j].splitter == unique_splitters[i])) ++j;
    }

    // Equal bucket handling
    // Step 1: within the same value skip all global splitters with id < 0
    // Step 2: for all local splitters: bucket_size += id[idx] - id[idx-1].
    // Step 3: excess at the end equal_bucket_size - id[last] is added to the next bucket.
    j = 0;
    const size_t n = a.size();

    for (size_t i = 0; i < num_unique_buckets; ++i) {
      while (j < num_buckets - 1 && i < num_unique_buckets - 1 &&
             (input_splitter[j].splitter < unique_splitters[i] ||
              (input_splitter[j].splitter == unique_splitters[i] &&
               input_splitter[j].GID < 0))) ++j;

      size_t prev_size = 0;
      while (j < num_buckets - 1 && i < num_unique_buckets - 1 &&
             (input_splitter[j].splitter == unique_splitters[i] &&
              static_cast<size_t>(input_splitter[j].GID) < n)) {
        real_bucket_sizes[j] += input_splitter[j].GID - prev_size;
        prev_size = input_splitter[j].GID;
        ++j;
      }

      real_bucket_sizes[j] += bucket_sizes[2 * i + 1] - prev_size;
    }

    return real_bucket_sizes;
  }

 private:
  void partition(std::vector<T>& begin_a, std::vector<T>& end_a, std::vector<size_t>& bucket_sizes,
                 TieBreaker<T>* s, std::vector<T>& unique_splitters,
                 const size_t num_buckets, const size_t log_num_buckets, const size_t k_real,
                 std::unique_ptr<oracle_type[]>& oracle) const {
    end_a.resize(begin_a.size());
    std::vector<size_t> bucket_sizes_tmp(2 * num_buckets);

    bucket_sizes[0] = begin_a.size();

    std::unique_ptr<T[]> tree_splitter(new T[num_buckets]);
    std::unique_ptr<T[]> tree_splitter_sorted(new T[num_buckets]);
    std::unique_ptr<TieBreakerRef<T>[]> local_splitters(new TieBreakerRef<T>[k_real]);

    // Substract global array id from each spliter, to make their indices local
    size_t num_local_splitters = 0;
    for (size_t i = 0; i != k_real - 1; ++i) {
      s[i].GID -= local_index_begin_;
      if (s[i].GID >= 0 && s[i].GID < static_cast<int64_t>(begin_a.size())) {
        local_splitters[num_local_splitters++] = TieBreakerRef<T>(s[i]);
      }
    }

    const size_t ref_cnt = num_local_splitters + 2;
    std::unique_ptr<TieBreakerRef<T>[]> local_splitter_ref(new TieBreakerRef<T>[ref_cnt]);

    recurse(
      begin_a, end_a, unique_splitters, local_splitters.get(),
      num_local_splitters, local_splitter_ref.get(), bucket_sizes, bucket_sizes_tmp,
      num_buckets, num_buckets, log_num_buckets,
      tree_splitter.get(), tree_splitter_sorted.get(), oracle);
  }


  void recurse(std::vector<T>& a, std::vector<T>& a_tmp, std::vector<T>& splitters,
               TieBreakerRef<T>* local_splitters, size_t num_local_splitters,
               TieBreakerRef<T>* local_splitter_ref,
               std::vector<size_t>& bucket_sizes, std::vector<size_t>& bucket_sizes_tmp,
               const size_t num_init_buckets, const size_t num_buckets,
               const size_t log_num_buckets, T* tree_splitter, T* tree_splitter_sorted,
               std::unique_ptr<oracle_type[]>& oracle) const {
    assert(num_init_buckets % num_buckets == 0);

    std::fill(bucket_sizes_tmp.begin(), bucket_sizes_tmp.begin() + 2 * num_init_buckets, 0);

    size_t bucket_offset = 0;
    assert(num_init_buckets % num_buckets == 0);

    // Calculate actual number of buckets at this level.
#if 1
    // Create buckets of (almost) the same size.
    const size_t num_levels = tlx::div_ceil(log_num_buckets, MAX_LOG_BUCKET);
    assert(num_levels > 0);
    const size_t log_num_local_buckets =
      tlx::div_ceil(log_num_buckets, num_levels);
    assert(log_num_local_buckets <= MAX_LOG_BUCKET);
    assert(log_num_local_buckets > 0);
    const size_t num_local_buckets = 1 << log_num_local_buckets;

#elif 0
    // Create large buckets first.
    const size_t log_num_local_buckets =
      std::min(log_num_sub_buckets, MAX_LOG_BUCKET);
    const size_t num_local_buckets = 1 << log_num_local_buckets;
#else
    // Create small buckets first.
    const size_t log_num_levels_floored = log_num_sub_buckets / MAX_LOG_BUCKET;
    const size_t log_num_floored_sub_buckets =
      log_num_levels_floored * MAX_LOG_BUCKET;
    const size_t residual_log = log_num_sub_buckets - log_num_floored_sub_buckets;
    size_t num_local_buckets = 1 << MAX_LOG_BUCKET;
    size_t log_num_local_buckets = MAX_LOG_BUCKET;
    if (residual_log > 0) {
      num_local_buckets = 1 << residual_log;
      log_num_local_buckets = residual_log;
    }
#endif

    // Running index over local splitters
    size_t splitter_idx = 0;

    // Splitter index of one past the last splitter of the current global bucket
    size_t end_pos = num_buckets - 1;

    // Partition each stripe.
    size_t num_stripes = num_init_buckets / num_buckets;
    for (size_t stripe_idx = 0; stripe_idx != num_stripes; ++stripe_idx) {
      assert(a.size() >= bucket_offset + bucket_sizes[2 * stripe_idx]);

      auto begin_stripe = a.data() + bucket_offset;
      auto end_stripe = begin_stripe + bucket_sizes[2 * stripe_idx];
      auto begin_tmp_stripe = a_tmp.data() + bucket_offset;

      // First dummy splitter
      local_splitter_ref[0] = TieBreakerRef<T>(-1);
      size_t num_local_stripe_splitters = 1;

      {
        auto stripe_splitters = splitters.data() + stripe_idx * num_buckets;
        const size_t num_stripe_buckets = num_buckets / num_local_buckets;
        for (size_t idx = 1; idx != num_local_buckets; ++idx) {
          tree_splitter_sorted[idx - 1] = stripe_splitters[idx * num_stripe_buckets - 1];
        }
      }
      DecisionTree<T>::build(tree_splitter, tree_splitter_sorted, num_local_buckets - 1);

      // Collect local splitters for this stripe.
      while (splitter_idx < num_local_splitters) {
        T& candidate = local_splitters[splitter_idx].ref->splitter;
        // We reached local splitters that cross a splitter boundary
        // That means we are not local anymore for this global bucket
        if (end_pos < num_init_buckets - 1 && candidate >= splitters[end_pos]) break;
        local_splitter_ref[num_local_stripe_splitters++] = local_splitters[splitter_idx++];
      }

      while (splitter_idx < num_local_splitters && end_pos < num_init_buckets - 1) {
        T& val = local_splitters[splitter_idx].ref->splitter;
        if (val != splitters[end_pos]) break;
        ++splitter_idx;
      }

      // Sort references of local splitters of this stripe by their GIDs
      std::sort(local_splitter_ref, local_splitter_ref + num_local_stripe_splitters);

      // Second dummy splitter
      local_splitter_ref[num_local_stripe_splitters] = TieBreakerRef<T>((end_stripe -
                                                                         begin_stripe) - 1);
      ++num_local_stripe_splitters;

      size_t* stripe_bucket_sizes = bucket_sizes_tmp.data() + 2 * stripe_idx * num_local_buckets;

      classify(begin_stripe, tree_splitter, tree_splitter_sorted, oracle.get(),
               local_splitter_ref, local_splitter_ref + num_local_stripe_splitters,
               stripe_bucket_sizes, log_num_local_buckets);
      distribute(begin_stripe, end_stripe, begin_tmp_stripe, oracle.get(),
                 stripe_bucket_sizes, 2 * num_local_buckets);

      // Copy the equal buckets of splitters which we already used to partition the data.
      std::copy(begin_stripe + bucket_sizes[2 * stripe_idx],
                begin_stripe + bucket_sizes[2 * stripe_idx] + bucket_sizes[2 * stripe_idx + 1],
                begin_tmp_stripe + bucket_sizes[2 * stripe_idx]);
      bucket_sizes_tmp[2 * (stripe_idx + 1) * num_local_buckets - 1] = bucket_sizes[2 * stripe_idx +
                                                                                    1];

      bucket_offset += bucket_sizes[2 * stripe_idx] + bucket_sizes[2 * stripe_idx + 1];
      end_pos += num_buckets;
    }

    bucket_sizes.swap(bucket_sizes_tmp);
    a.swap(a_tmp);

    if (num_local_buckets != num_buckets) {
      assert(num_buckets % num_local_buckets == 0);

      recurse(a, a_tmp, splitters, local_splitters, num_local_splitters, local_splitter_ref,
              bucket_sizes, bucket_sizes_tmp, num_init_buckets,
              num_buckets / num_local_buckets, log_num_buckets - log_num_local_buckets,
              tree_splitter, tree_splitter_sorted, oracle);
    }
  }

  // Extracts the elements of the TieBreakers and removes duplicates.
  std::vector<T> getUniqueValues(TieBreaker<T>* splitter,
                                 size_t count) const {
    std::vector<T> filter;

    for (size_t i = 0; i < count; ++i) {
      size_t j = i;
      while (j < count && splitter[i].splitter == splitter[j].splitter) ++j;
      filter.push_back(splitter[i].splitter);
      i = j - 1;
    }

    return filter;
  }

  void distribute(const T* a_begin,
                  const T* const a_end,
                  T* const a_tmp,
                  const oracle_type* oracle,
                  size_t* const bucket_sizes,
                  const oracle_type num_buckets) const {
    constexpr const int kUnroll = UNROLL_CLASSIFIER;

    std::unique_ptr<size_t[]> sbs(new size_t[num_buckets + 1]);
    tlx::exclusive_scan(bucket_sizes, bucket_sizes + num_buckets,
                        sbs.get(), oracle_type{ 0 }, std::plus<>{ });

    assert(sbs[num_buckets] == static_cast<size_t>(a_end - a_begin));

    ASSUME_NOT(a_end < a_begin);

    if (kUnroll <= a_end - a_begin) {
      ASSUME_NOT(a_begin + kUnroll > a_end);
      for (auto cutoff = a_end - kUnroll;
           a_begin <= cutoff;
           a_begin += kUnroll, oracle += kUnroll) {
        for (int i = 0; i < kUnroll; ++i) {
          a_tmp[sbs[oracle[i]]++] = a_begin[i];
        }
      }
    }

    for ( ; a_begin != a_end; ++a_begin, ++oracle) {
      a_tmp[sbs[*oracle]++] = *a_begin;
    }
  }

  /**
   * Classifies all elements using a callback.
   */
  void classify(T* begin,
                T* splitter, T* sorted_splitter,
                oracle_type* oracle,
                TieBreakerRef<T>* begin_local_splitters,
                TieBreakerRef<T>* end_local_splitters,
                size_t* bucket_sizes, size_t log_num_buckets) const {
    static_assert(MAX_LOG_BUCKET <= 8, "Too many buckets.");
    switch (log_num_buckets) {
      case 1: classifyUnrolled<1>(begin, splitter, sorted_splitter,
                                  oracle, begin_local_splitters, end_local_splitters, bucket_sizes);
        break;
      case 2: classifyUnrolled<2>(begin, splitter, sorted_splitter,
                                  oracle, begin_local_splitters, end_local_splitters, bucket_sizes);
        break;
      case 3: classifyUnrolled<3>(begin, splitter, sorted_splitter,
                                  oracle, begin_local_splitters, end_local_splitters, bucket_sizes);
        break;
      case 4: classifyUnrolled<4>(begin, splitter, sorted_splitter,
                                  oracle, begin_local_splitters, end_local_splitters, bucket_sizes);
        break;
      case 5: classifyUnrolled<5>(begin, splitter, sorted_splitter,
                                  oracle, begin_local_splitters, end_local_splitters, bucket_sizes);
        break;
      case 6: classifyUnrolled<6>(begin, splitter, sorted_splitter,
                                  oracle, begin_local_splitters, end_local_splitters, bucket_sizes);
        break;
      case 7: classifyUnrolled<7>(begin, splitter, sorted_splitter,
                                  oracle, begin_local_splitters, end_local_splitters, bucket_sizes);
        break;
      case 8: classifyUnrolled<8>(begin, splitter, sorted_splitter,
                                  oracle, begin_local_splitters, end_local_splitters, bucket_sizes);
        break;
    }
  }

  /**
   * Classifies all elements using a callback.
   */
  template <int kLogBuckets>
  void classifyUnrolled(T* begin,
                        T* splitter, T* sorted_splitter,
                        oracle_type* oracle,
                        TieBreakerRef<T>* begin_local_splitters,
                        TieBreakerRef<T>* end_local_splitters,
                        size_t* bucket_sizes) const {
    using bucket_type = size_t;
    constexpr const bucket_type kNumBuckets = 1l << kLogBuckets;
    constexpr const bucket_type kNumTotalBuckets = 2 * kNumBuckets;
    constexpr const int kUnroll = UNROLL_CLASSIFIER;

    bucket_type b[kUnroll];

    // After the first iteration of the outer for-loop, this value
    // stores the bucket id of the last element which we have
    // classified. This value is the GID of the right local
    // splitter.
    bucket_type non_unrolled_bucket_id = 0;

    // Process stripes of elements separated by the local splitters.
    for (auto loc_splitter = begin_local_splitters;
         loc_splitter + 1 < end_local_splitters;
         ++loc_splitter) {
      // Reset GID of local splitters expect the first and the
      // last local splitter. Those splitters are dummy
      // splitters and have GID '-1' anyway.
      if (loc_splitter != begin_local_splitters) {
        loc_splitter->ref->GID = bucket_sizes[non_unrolled_bucket_id] - 1;
      }

      const auto stripe_size = (loc_splitter + 1)->GID - loc_splitter->GID;
      const auto loc_end = begin + stripe_size;
      const auto oracle_loc_end = oracle + stripe_size;

      ASSUME_NOT(begin > loc_end);
      ASSUME_NOT(oracle > oracle_loc_end);

      // We do not process the last 'kUnroll + 1' elements to
      // have at least one element left for the the non-unrolled
      // loop.  This element is used to be an element which is
      // followed by a local splitter.
      if (oracle + kUnroll + 1 <= oracle_loc_end) {
        ASSUME_NOT(begin + kUnroll + 1 > loc_end);
        ASSUME_NOT(oracle + kUnroll + 1 > oracle_loc_end);

        for (auto cutoff = loc_end - kUnroll - 1;
             begin <= cutoff;
             begin += kUnroll, oracle += kUnroll) {
          for (int i = 0; i < kUnroll; ++i)
            b[i] = 1;

          for (int l = 0; l < kLogBuckets; ++l)
            for (int i = 0; i < kUnroll; ++i)
              b[i] = 2 * b[i] + comp_(splitter[b[i]], begin[i]);

          for (int i = 0; i < kUnroll; ++i)
            // We don't want a branch here, so we use
            // '&'. We cast to bool before performing the
            // '&' operation to guarantee that the correct
            // bits are used.
            b[i] = 2 * b[i]
                   + (static_cast<bool>(b[i] <
                                        kNumTotalBuckets - 1) &
                      static_cast<bool>(!comp_(begin[i], sorted_splitter[b[i] - kNumBuckets])));

          for (int i = 0; i < kUnroll; ++i) {
            const auto bucket_id = b[i] - kNumTotalBuckets;
            *(oracle + i) = bucket_id;
            ++bucket_sizes[bucket_id];
          }
        }
      }

      ASSUME_NOT(begin > loc_end);
      for ( ; begin != loc_end; ++begin, ++oracle) {
        bucket_type b = 1;
        for (int l = 0; l < kLogBuckets; ++l)
          b = 2 * b + comp_(splitter[b], *begin);
        b = 2 * b
            + (static_cast<bool>(b <
                                 kNumTotalBuckets - 1) &
               static_cast<bool>(!comp_(*begin, sorted_splitter[b - kNumBuckets])));
        non_unrolled_bucket_id = b - kNumTotalBuckets;
        *oracle = non_unrolled_bucket_id;
        ++bucket_sizes[non_unrolled_bucket_id];
      }
    }
  }

  Comp comp_;
  int64_t local_index_begin_;
  int64_t local_index_end_;
};
}  // end namespace _internal
}  // end namespace Ams
