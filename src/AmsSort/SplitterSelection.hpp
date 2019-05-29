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
#include <cstddef>
#include <tuple>
#include <vector>

#include "../Tools/CommonMpi.hpp"
#include "RFis/RFis.hpp"

#include <RBC.hpp>

namespace Ams {
namespace SplitterSelection {
/*!
 * @brief Sorts the samples in parallel, selects splitters and
 * returns these splitters on each PE.
 *
 * @param loc_samples Local samples which must be sorted.
 * @param num_glob_samples Must be greater than 0 or num_glob_splitters is 0.
 */
template <class AmsData, class SampleType, class SplitterComp>
std::vector<SampleType> SplitterSelection(AmsData& ams_data,
                                          MPI_Datatype mpi_sample_type,
                                          const std::vector<SampleType>& loc_samples,
                                          SplitterComp splitter_comp,
                                          size_t num_glob_samples,
                                          size_t num_glob_splitters,
                                          bool use_two_tree);

namespace _internal {
template <class SampleType, class RankType>
std::vector<SampleType> SelectLocalSplittersFromRankedValues(const std::vector<SampleType>& samples,
                                                             const std::vector<RankType>& ranks,
                                                             size_t num_glob_samples, size_t
                                                             num_glob_splitters) {
  std::vector<SampleType> loc_splitters;

  const double splitter_dist = static_cast<double>(num_glob_samples) / (num_glob_splitters + 1);

  // Search each splitter.
  auto rank_it = ranks.begin();
  for (size_t splitter_idx = 1; splitter_idx != num_glob_splitters + 1;
       ++splitter_idx) {
    size_t sp_rank = splitter_dist * static_cast<double>(splitter_idx);
    while (rank_it != ranks.end() && *rank_it < sp_rank) {
      ++rank_it;
    }
    if (rank_it != ranks.end() && *rank_it == sp_rank) {
      const auto idx = rank_it - ranks.begin();
      loc_splitters.push_back(samples[idx]);
    }
  }

  return loc_splitters;
}

template <class T, class Comp>
std::vector<T> Allgathermerge(const std::vector<T>& in,
                              size_t tot_size,
                              MPI_Datatype mpi_type, Comp comp,
                              bool use_two_tree, const RBC::Comm& comm) {
  assert(std::is_sorted(in.begin(), in.end(), comp));

  std::function<void(void*, void*, void*, void*, void*)> op =
    [&comp](void* first1, void* last1,
            void* first2, void* last2,
            void* out) {
      std::merge(static_cast<T*>(first1),
                 static_cast<T*>(last1),
                 static_cast<T*>(first2),
                 static_cast<T*>(last2),
                 static_cast<T*>(out), comp);
    };

  // Return to avoid that in.data() or out.data() is a nullptr.
  if (tot_size == 0) {
    return std::vector<T>{ };
  }

  std::vector<T> out(tot_size);

  RBC::Gatherm(in.data(), in.size(), mpi_type, out.data(), out.size(), 0, op, comm);

  if (use_two_tree) {
    RBC::_internal::optimized::BcastTwotree(out.data(), out.size(), mpi_type, 0, comm);
  } else {
    RBC::Bcast(out.data(), out.size(), mpi_type, 0, comm);
  }

  return out;
}

template <class T, class Comp>
std::vector<T> Allgathermerge(const std::vector<T>& in,
                              MPI_Datatype mpi_type, Comp comp,
                              bool use_two_tree, const RBC::Comm& comm) {
  size_t tot_size;
  const size_t el_cnt = in.size();
  RBC::Allreduce(&el_cnt, &tot_size, 1, Common::getMpiType(el_cnt), MPI_SUM, comm);

  return Allgathermerge(in, tot_size, mpi_type, comp, use_two_tree, comm);
}

template <class AmsData, class SampleType, class SplitterComp>
std::vector<SampleType> SplitterSelectionFis(MPI_Datatype mpi_sample_type,
                                             const std::vector<SampleType>& loc_samples,
                                             SplitterComp splitter_comp,
                                             size_t num_glob_samples,
                                             size_t num_glob_splitters,
                                             bool use_two_tree,
                                             const RBC::Comm& comm) {
  using Tags = typename AmsData::Tags::RFisTags;
  
  assert(num_glob_splitters > 0);

  std::vector<SampleType> samples;
  std::vector<size_t> ranks;


  // Create a temporary communicator which ensures that the
  // RFisSorter does not split communicators even when RBC
  // collectives are disabled in the RBC communicator 'comm'.
  RBC::Comm rbc_comm, rbc_subcomm;
  // Splitting disabled, RBC collectives enabled
  RBC::Create_Comm_from_MPI(comm.get(), &rbc_comm, true, false, false);
  RBC::Comm_create_group(rbc_comm, &rbc_subcomm, comm.getMpiFirst(),
                         comm.getMpiLast(), comm.getStride());

  RBC::Comm col_comm;
  std::tie(samples, ranks, col_comm) =
    RFis::RankRowwise<Tags, SampleType, size_t, decltype(splitter_comp)>(
      mpi_sample_type, Common::getMpiType<size_t>(),
      loc_samples, splitter_comp, rbc_subcomm);

  const auto loc_splitters = SelectLocalSplittersFromRankedValues(
    samples, ranks, num_glob_samples,
    num_glob_splitters);

  return Allgathermerge(loc_splitters, mpi_sample_type,
                        splitter_comp, use_two_tree, col_comm);
}
}  // namespace _internal

template <class AmsData, class SampleType, class SplitterComp>
std::vector<SampleType> SplitterSelection(AmsData& ams_data,
                                          MPI_Datatype mpi_sample_type,
                                          const std::vector<SampleType>& loc_samples,
                                          SplitterComp splitter_comp,
                                          size_t num_glob_samples,
                                          size_t num_glob_splitters,
                                          bool use_two_tree) {
  return _internal::SplitterSelectionFis<AmsData>(mpi_sample_type,
                                         loc_samples, splitter_comp, num_glob_samples,
                                         num_glob_splitters, use_two_tree, ams_data.comm());
}
}  // end namespace SplitterSelection
}  // end namespace Ams
