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
#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include "../../include/Tools/MpiTuple.hpp"
#include "../Tools/CommonMpi.hpp"
#include "LocalSampleCount.hpp"

#include <RBC.hpp>
#include <tlx/math.hpp>

namespace Ams {
namespace _internal {

int64_t getIndexFromRank(int64_t nprocs, int64_t rank, int64_t num_glob_els) {
  const int64_t min_num_els = num_glob_els / nprocs;
  const int64_t extra_elements = num_glob_els % nprocs;
  if (rank <= extra_elements) {
    return rank * (min_num_els + 1);
  } else {
    return rank * min_num_els + extra_elements;
  }
}

std::tuple<int64_t, int64_t> CalcLocalSamples(size_t num_glob_els, size_t total_samples,
                                              std::mt19937_64& sync_gen, RBC::Comm comm) {
  if (num_glob_els == 0) {
    return std::tuple<int64_t, int64_t>{ 0, 0 };
  }

  // Initialize random generator.
  // We create a new random generator for this implementation to branch in the tree.
  std::linear_congruential_engine<std::uint_fast64_t,
                                  6364136223846793005u,
                                  1442695040888963407u,
                                  0u> random_generator(sync_gen());

  // Generators are synced.
  {
#ifndef NDEBUG
    size_t val = sync_gen();
    assert(Common::aggregate<size_t>(val, Common::getMpiType(val), MPI_BAND, comm)
           ==
           Common::aggregate<size_t>(val, Common::getMpiType(val), MPI_BOR, comm));
    val = random_generator();
    assert(Common::aggregate<size_t>(val, Common::getMpiType(val), MPI_BAND, comm)
           ==
           Common::aggregate<size_t>(val, Common::getMpiType(val), MPI_BOR, comm));
#endif
  }

#ifndef NDEBUG
  size_t total_num_els = 0;
  RBC::Allreduce(&num_glob_els, &total_num_els, 1, Common::getMpiType(num_glob_els), MPI_SUM, comm);
  assert(total_num_els > 0);
#endif
  assert(total_samples > 0);
  assert(comm.getSize() > 1);

  int max_height = tlx::integer_log2_ceil(comm.getSize());


  int first_PE = 0;
  int last_PE = comm.getSize() - 1;
  int64_t local_samples = total_samples;
  int64_t prefix_sample_cnt = 0;

  for (int height = max_height; height > 0; height--) {
    if (first_PE + (1 << (height - 1)) > last_PE) {
      // right subtree is empty
    } else {
      int left_size = std::pow(2, height - 1);
      assert(left_size > 0);
      int64_t left_elements = getIndexFromRank(comm.getSize(), first_PE + left_size, num_glob_els)
                              - getIndexFromRank(comm.getSize(), first_PE, num_glob_els);
      int64_t right_elements = getIndexFromRank(comm.getSize(), last_PE + 1, num_glob_els)
                               - getIndexFromRank(comm.getSize(), first_PE + left_size,
                                                  num_glob_els);

      // If the PEs to the left do not have any elements, the
      // PEs to the right have no element either.
      assert(left_elements > 0 || right_elements == 0);

      int64_t samples_left = local_samples;
      int64_t samples_right = 0;
      if (right_elements > 0) {
        double percentage_left = static_cast<double>(left_elements)
                                 / static_cast<double>(left_elements + right_elements);

        std::binomial_distribution<int64_t> binom_distr(local_samples, percentage_left);
        samples_left = binom_distr(random_generator);
        samples_right = local_samples - samples_left;
      }

      int mid_PE = first_PE + std::pow(2, height - 1);
      if (comm.getRank() < mid_PE) {
        // left side
        last_PE = mid_PE - 1;
        local_samples = samples_left;
      } else {
        // right side
        first_PE = mid_PE;
        local_samples = samples_right;
        prefix_sample_cnt += samples_left;
        random_generator.seed(random_generator());
      }
    }
  }

  // // Sync generator -- PEs do not have the same tree height.
  // for (auto i = own_height; i != max_height; ++i) {
  //     sync_gen();
  // }

  assert(Common::aggregate<size_t>(local_samples, Common::getMpiType(local_samples), MPI_SUM, comm)
         == total_samples);

  return std::tuple<int64_t, int64_t>{ prefix_sample_cnt, local_samples };
}


size_t locSampleCountWithSameInputSize(size_t total_samples, std::mt19937_64& sync_gen,
                                       const RBC::Comm& comm) {
  assert(total_samples > 0);

  // Initialize random generator.
  // We create a new random generator for this implementation to branch in the tree.
  std::linear_congruential_engine<std::uint_fast64_t,
                                  6364136223846793005u,
                                  1442695040888963407u,
                                  0u> random_generator(sync_gen());

  // Generators are synced.
  {
#ifndef NDEBUG
    size_t val = sync_gen();
    assert(Common::aggregate<size_t>(val, Common::getMpiType(val), MPI_BAND, comm)
           ==
           Common::aggregate<size_t>(val, Common::getMpiType(val), MPI_BOR, comm));
    val = random_generator();
    assert(Common::aggregate<size_t>(val, Common::getMpiType(val), MPI_BAND, comm)
           ==
           Common::aggregate<size_t>(val, Common::getMpiType(val), MPI_BOR, comm));
#endif
  }

  const size_t max_height = tlx::integer_log2_ceil(comm.getSize());


  int begin_pe = 0;
  int end_pe = comm.getSize();
  size_t local_samples = total_samples;

  for (size_t height = max_height; height > 0; height--) {
    const int left_size = (1 << (height - 1));
    // right subtree is empty
    if (begin_pe + left_size < end_pe) {
      assert(left_size > 0);

      const double right_size = end_pe - begin_pe - left_size;

      assert(left_size > 0 && right_size > 0);

      const double ratio = static_cast<double>(left_size) / (left_size + right_size);

      std::binomial_distribution<size_t> binom_distr(local_samples, ratio);
      const size_t samples_left = binom_distr(random_generator);
      const size_t samples_right = local_samples - samples_left;

      const int mid_pe = begin_pe + std::pow(2, height - 1);
      if (comm.getRank() < mid_pe) {
        // left side
        end_pe = mid_pe;
        local_samples = samples_left;
      } else {
        // right side
        begin_pe = mid_pe;
        local_samples = samples_right;
        random_generator.seed(random_generator());
      }
    }
  }

  assert(Common::aggregate<size_t>(local_samples, Common::getMpiType(local_samples), MPI_SUM, comm)
         == total_samples);

  return local_samples;
}
}  // namespace _internal
}  // end namespace Ams
