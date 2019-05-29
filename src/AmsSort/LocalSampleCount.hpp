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
#include <random>
#include <tuple>
#include <vector>

#include <RBC.hpp>
#include <tlx/math.hpp>

namespace Ams {
namespace _internal {
// Calculates number of local samples without communication
// assuming that each process has the same number of elements.
size_t locSampleCountWithSameInputSize(size_t total_samples, std::mt19937_64& sync_gen,
                                       const RBC::Comm& comm);

// // Calculates number of local samples without communication
// // assuming that each process has the same number of
// // elements. If the total number of elements is not a multiple
// // of the number of processes, the elements are evenly
// // distributed and the first processes have one additional
// // element.
// size_t locSampleCountWithSameInputSize(size_t total_el_cnt, size_t total_samples,
//                                        std::mt19937_64& sync_gen,
//                                        const RBC::Comm& comn);

// Calculates number of local samples with communication (async generator).
template<class AmsTag>
size_t locSampleCountWithoutSameInputSize(size_t num_local_els, size_t total_samples,
                                          std::mt19937_64& async_gen, const RBC::Comm& comm);

// Calculates number of local samples with communication (async generator).
template<class AmsTag>
std::vector<int64_t> locSampleCountWithoutSameInputSize(const std::vector<int64_t>& num_local_els,
                                                        const std::vector<int64_t>& total_samples,
                                                        std::mt19937_64& async_gen,
                                                        const RBC::Comm& comm);

// Calculates number of local samples and the number of
// samples on processes with smaller index with communication
// (async generator).
template<class AmsTag>
std::tuple<int64_t, int64_t> locSampleCountWithoutSameInputSizeExscan(size_t num_local_els, size_t
                                                                      total_samples,
                                                                      std::mt19937_64& async_gen,
                                                                      const RBC::Comm& comm);

/* @brief Calculates number of elements on the first 'rank' PEs
 * (exclusive) assuming that the elements are evenly distributed.
 */
int64_t getIndexFromRank(int64_t nprocs, int64_t rank, int64_t num_glob_els);

std::tuple<int64_t, int64_t> CalcLocalSamples(size_t num_glob_els, size_t total_samples,
                                              std::mt19937_64& sync_gen, RBC::Comm comm);

template<class AmsTag>
size_t locSampleCountWithoutSameInputSize(size_t num_local_els, size_t total_samples,
                                          std::mt19937_64& async_gen, const RBC::Comm& comm) {
  if (total_samples == 0) {
    return 0;
  }

  if (comm.getSize() == 1) {
    return total_samples;
  }

  const int tag = AmsTag::kGeneral;

  // Implementation detail: we use signed values to inform processes
  // that there are not input elements at all (-1).

  std::vector<int64_t> load_l, load_r;
  int64_t tot = num_local_els;

  // Reduce.
  const int tailing_zeros = tlx::ffs(comm.getRank()) - 1;
  const int iterations = comm.getRank() > 0 ?
                         tailing_zeros : tlx::integer_log2_ceil(comm.getSize());

  MPI_Status status;

  int64_t tot_r = 0;
  for (int k = 0; k != iterations; ++k) {
    int target_id = comm.getRank() + (1 << k);
    if (target_id >= comm.getSize()) {
      continue;
    }
    RBC::Recv(&tot_r, 1, Common::getMpiType(tot_r), target_id, tag, comm, &status);

    // op
    load_r.push_back(tot_r);
    load_l.push_back(tot);
    tot += tot_r;
  }
  if (comm.getRank() > 0) {
    int target_id = comm.getRank() - (1 << tailing_zeros);
    RBC::Send(&tot, 1, Common::getMpiType(tot), target_id, tag, comm);
  }

  // Bcast

  // Set values for PE 0.
  int64_t tree_sample_cnt = total_samples;

  // Receive values for PE > 0.
  if (comm.getRank() > 0) {
    int src_id = comm.getRank() - (1 << tailing_zeros);
    RBC::Recv(&tree_sample_cnt, 1, Common::getMpiType(tree_sample_cnt), src_id, tag, comm,
              &status);
  }

  // this case and remove the if expression.
  if (comm.getRank() == 0 && load_r.back() == 0 && load_l.back() == 0 &&
      tree_sample_cnt > 0) {
    // There are no global elements at all.
    tree_sample_cnt = -1;
  }

  for (int kr = iterations; kr > 0; kr--) {
    const int k = kr - 1;
    int target_id = comm.getRank() + (1 << k);
    if (target_id >= comm.getSize()) {
      continue;
    }

    int64_t right_subtree_sample_cnt = 0;
    if (tree_sample_cnt < 0) {
      // There are no global elements at all.
      right_subtree_sample_cnt = -1;
    }
    if (tree_sample_cnt == 0) {
      right_subtree_sample_cnt = 0;
      tree_sample_cnt -= right_subtree_sample_cnt;
    } else if (load_r.back() == 0) {
      right_subtree_sample_cnt = 0;
      tree_sample_cnt -= right_subtree_sample_cnt;
    } else if (load_l.back() == 0) {
      right_subtree_sample_cnt = tree_sample_cnt;
      tree_sample_cnt -= right_subtree_sample_cnt;
    } else {
      const double right_p = load_r.back()
                             / static_cast<double>(load_l.back() + load_r.back());
      std::binomial_distribution<int64_t> distr(tree_sample_cnt, right_p);
      right_subtree_sample_cnt = distr(async_gen);
      tree_sample_cnt -= right_subtree_sample_cnt;
    }
    // The prefix sample count of our right child is our prefix
    // sample count plus the current sample count of our
    // tree. Atm, the current sample count of our tree is the
    // sample count of our tree minus the sample count of our
    // right subtree.
    RBC::Send(&right_subtree_sample_cnt,
              1,
              Common::getMpiType(right_subtree_sample_cnt),
              target_id,
              tag,
              comm);

    load_l.pop_back();
    load_r.pop_back();
  }

  assert(Common::aggregate<size_t>(tree_sample_cnt, Common::getMpiType(tree_sample_cnt),
                                   MPI_SUM, comm) == total_samples);

  return tree_sample_cnt;
}

template<class AmsTag>
std::vector<int64_t> locSampleCountWithoutSameInputSize(const std::vector<int64_t>& num_local_els,
                                                        const std::vector<int64_t>& total_samples,
                                                        std::mt19937_64& async_gen,
                                                        const RBC::Comm& comm) {
  if (comm.getSize() == 1) {
    return total_samples;
  }

  const size_t dim = total_samples.size();

  const int tag = AmsTag::kGeneral;

  // Implementation detail: we use signed values to inform processes
  // that there are not input elements at all (-1).

  std::vector<std::vector<int64_t> > load_l(dim), load_r(dim);
  std::vector<int64_t> tot = num_local_els;

  // Reduce.
  const int tailing_zeros = tlx::ffs(comm.getRank()) - 1;
  const int iterations = comm.getRank() > 0 ?
                         tailing_zeros : tlx::integer_log2_ceil(comm.getSize());

  MPI_Status status;

  std::vector<int64_t> tot_r(dim, 0);
  for (int k = 0; k != iterations; ++k) {
    int target_id = comm.getRank() + (1 << k);
    if (target_id >= comm.getSize()) {
      continue;
    }
    RBC::Recv(tot_r.data(), dim, Common::getMpiType(tot_r), target_id, tag, comm, &status);

    // op
    for (size_t i = 0; i != dim; ++i) {
      load_r[i].push_back(tot_r[i]);
      load_l[i].push_back(tot[i]);
      tot[i] += tot_r[i];
    }
  }
  if (comm.getRank() > 0) {
    int target_id = comm.getRank() - (1 << tailing_zeros);
    RBC::Send(tot.data(), dim, Common::getMpiType(tot), target_id, tag, comm);
  }

  // Bcast

  // Set values for PE 0.
  std::vector<int64_t> tree_sample_cnt = total_samples;

  // Receive values for PE > 0.
  if (comm.getRank() > 0) {
    int src_id = comm.getRank() - (1 << tailing_zeros);
    RBC::Recv(tree_sample_cnt.data(), dim,
              Common::getMpiType(tree_sample_cnt),
              src_id, tag, comm,
              &status);
  }

  // this case and remove the if expression.
  for (size_t i = 0; i != dim; ++i) {
    if (comm.getRank() == 0 && load_r[i].back() == 0 && load_l[i].back() == 0 &&
        tree_sample_cnt[i] > 0) {
      // There are no global elements at all.
      tree_sample_cnt[i] = -1;
    }
  }

  for (int kr = iterations; kr > 0; kr--) {
    const int k = kr - 1;
    int target_id = comm.getRank() + (1 << k);
    if (target_id >= comm.getSize()) {
      continue;
    }

    std::vector<int64_t> right_subtree_sample_cnt(dim, 0);
    for (size_t i = 0; i != dim; ++i) {
      if (tree_sample_cnt[i] < 0) {
        // There are no global elements at all.
        right_subtree_sample_cnt[i] = -1;
      }
      if (tree_sample_cnt[i] == 0) {
        right_subtree_sample_cnt[i] = 0;
        tree_sample_cnt[i] -= right_subtree_sample_cnt[i];
      } else if (load_r[i].back() == 0) {
        right_subtree_sample_cnt[i] = 0;
        tree_sample_cnt[i] -= right_subtree_sample_cnt[i];
      } else if (load_l[i].back() == 0) {
        right_subtree_sample_cnt[i] = tree_sample_cnt[i];
        tree_sample_cnt[i] -= right_subtree_sample_cnt[i];
      } else {
        const double right_p = load_r[i].back()
                               / static_cast<double>(load_l[i].back() + load_r[i].back());
        std::binomial_distribution<int64_t> distr(tree_sample_cnt[i], right_p);
        right_subtree_sample_cnt[i] = distr(async_gen);
        tree_sample_cnt[i] -= right_subtree_sample_cnt[i];
      }
    }

    // The prefix sample count of our right child is our prefix
    // sample count plus the current sample count of our
    // tree. Atm, the current sample count of our tree is the
    // sample count of our tree minus the sample count of our
    // right subtree.
    RBC::Send(right_subtree_sample_cnt.data(), dim,
              Common::getMpiType(right_subtree_sample_cnt),
              target_id, tag, comm);

    for (size_t i = 0; i != dim; ++i) {
      load_l[i].pop_back();
      load_r[i].pop_back();
    }
  }

  for (size_t i = 0; i != dim; ++i) {
    assert(Common::aggregate<int64_t>(tree_sample_cnt[i],
                                      Common::getMpiType(tree_sample_cnt),
                                      MPI_SUM, comm) == total_samples[i]);
  }

  return tree_sample_cnt;
}

template<class AmsTag>
std::tuple<int64_t, int64_t> locSampleCountWithoutSameInputSizeExscan(size_t num_local_els, size_t
                                                                      total_samples,
                                                                      std::mt19937_64& async_gen,
                                                                      const RBC::Comm& comm) {
  if (total_samples == 0) {
    return std::tuple<int64_t, int64_t>{ 0, 0 };
  }

  if (comm.getSize() == 1) {
    return std::tuple<int64_t, int64_t>{ 0, total_samples };
  }

  const int tag = AmsTag::kGeneral;

  // Implementation detail: we use signed values to inform processes
  // that there are not input elements at all (-1).

  std::vector<int64_t> load_l, load_r;
  int64_t tot = num_local_els;

  // Reduce.
  const int tailing_zeros = tlx::ffs(comm.getRank()) - 1;
  const int iterations = comm.getRank() > 0 ?
                         tailing_zeros : tlx::integer_log2_ceil(comm.getSize());

  MPI_Status status;

  int64_t tot_r = 0;
  for (int k = 0; k != iterations; ++k) {
    const int target_id = comm.getRank() + (1 << k);
    if (target_id >= comm.getSize()) {
      continue;
    }
    RBC::Recv(&tot_r, 1, Common::getMpiType(tot_r), target_id, tag, comm, &status);

    // op
    load_r.push_back(tot_r);
    load_l.push_back(tot);
    tot += tot_r;
  }
  if (comm.getRank() > 0) {
    const int target_id = comm.getRank() - (1 << tailing_zeros);
    RBC::Send(&tot, 1, Common::getMpiType(tot), target_id, tag, comm);
  }

  // Bcast

  using MpiTuple = Tools::Tuple<int64_t, int64_t>;
  MpiTuple tuple;
  MPI_Datatype mpi_tuple_type = MpiTuple::MpiType(
    Common::getMpiType<int64_t>(),
    Common::getMpiType<int64_t>());

  // Set values for PE 0.
  int64_t tree_sample_cnt = total_samples;
  int64_t prefix_sample_cnt = 0;

  // Receive values for PE > 0.
  if (comm.getRank() > 0) {
    const int src_id = comm.getRank() - (1 << tailing_zeros);
    RBC::Recv(&tuple, 1, mpi_tuple_type, src_id, tag, comm,
              &status);
    prefix_sample_cnt = tuple.first;
    tree_sample_cnt = tuple.second;
  }

  // this case and remove the if expression.
  if (comm.getRank() == 0 && load_r.back() == 0 && load_l.back() == 0 &&
      tree_sample_cnt > 0) {
    // There are no global elements at all.
    tree_sample_cnt = -1;
  }

  for (int kr = iterations; kr > 0; kr--) {
    const int k = kr - 1;
    int target_id = comm.getRank() + (1 << k);
    if (target_id >= comm.getSize()) {
      continue;
    }

    int64_t right_subtree_sample_cnt = 0;
    if (tree_sample_cnt < 0) {
      // There are no global elements at all.
      right_subtree_sample_cnt = -1;
    }
    if (tree_sample_cnt == 0) {
      right_subtree_sample_cnt = 0;
      tree_sample_cnt -= right_subtree_sample_cnt;
    } else if (load_r.back() == 0) {
      right_subtree_sample_cnt = 0;
      tree_sample_cnt -= right_subtree_sample_cnt;
    } else if (load_l.back() == 0) {
      right_subtree_sample_cnt = tree_sample_cnt;
      tree_sample_cnt -= right_subtree_sample_cnt;
    } else {
      double right_p = load_r.back()
                       / static_cast<double>(load_l.back() + load_r.back());
      std::binomial_distribution<int64_t> distr(tree_sample_cnt, right_p);
      right_subtree_sample_cnt = distr(async_gen);
      tree_sample_cnt -= right_subtree_sample_cnt;
    }
    // The prefix sample count of our right child is our prefix
    // sample count plus the current sample count of our
    // tree. Atm, the current sample count of our tree is the
    // sample count of our tree minus the sample count of our
    // right subtree.
    tuple.second = right_subtree_sample_cnt;
    RBC::Send(&tuple, 1, mpi_tuple_type, target_id, tag, comm);

    load_l.pop_back();
    load_r.pop_back();
  }

  assert(Common::aggregate<size_t>(tree_sample_cnt, Common::getMpiType(tree_sample_cnt), MPI_SUM,
                                   comm)
         == total_samples);

  return std::tuple<int64_t, int64_t>{ prefix_sample_cnt, tree_sample_cnt };
}
}     // namespace _internal
}  // end namespace Ams
