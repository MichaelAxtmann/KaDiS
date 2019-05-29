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
#include <random>
#include <vector>

#include "TieBreaker.hpp"

namespace Ams {
namespace _internal {
template <class T>
std::vector<T> sampleUniform(const std::vector<T>& loc_data,
                             size_t sample_cnt,
                             std::mt19937_64& async_gen) {
  if (loc_data.empty()) {
    return std::vector<T>();
  }

  std::vector<T> samples;
  samples.reserve(sample_cnt);

  std::uniform_int_distribution<size_t> distr(0, loc_data.size() - 1);
  for (size_t i = 0; i != sample_cnt; ++i) {
    auto idx = distr(async_gen);
    samples.emplace_back(loc_data[idx]);
  }

  return samples;
}

template <class T>
std::vector<T> sampleUniform(const std::vector<T>& loc_data,
                             size_t range,
                             size_t sample_cnt,
                             std::mt19937_64& async_gen) {
  assert(range >= loc_data.size());

  if (loc_data.empty()) {
    return std::vector<T>();
  }

  std::vector<T> samples;
  samples.reserve(sample_cnt);

  std::uniform_int_distribution<size_t> distr(0, range - 1);
  for (size_t i = 0; i != sample_cnt; ++i) {
    auto idx = distr(async_gen);
    if (idx < loc_data.size()) {
      samples.emplace_back(loc_data[idx]);
    }
  }

  return samples;
}

template <class T>
std::vector<TieBreaker<T> > sampleUniformTieBreaker(const size_t myrank,
                                                    const std::vector<T>& loc_data,
                                                    size_t sample_cnt,
                                                    std::mt19937_64& async_gen) {
  if (loc_data.empty()) {
    return std::vector<TieBreaker<T> >();
  }

  std::vector<TieBreaker<T> > samples;
  samples.reserve(sample_cnt);

  std::uniform_int_distribution<size_t> distr(0, loc_data.size() - 1);
  const auto shifted_rank = TieBreaker<T>::GetOffset(myrank);
  for (size_t i = 0; i != sample_cnt; ++i) {
    auto idx = distr(async_gen);
    if (idx < loc_data.size()) {
      samples.emplace_back(loc_data[idx], shifted_rank + idx);
    }
  }

  return samples;
}

template <class T>
std::vector<TieBreaker<T> > sampleUniformTieBreaker(const size_t myrank,
                                                    const std::vector<T>& loc_data,
                                                    size_t range,
                                                    size_t sample_cnt,
                                                    std::mt19937_64& async_gen) {
  assert(range >= loc_data.size());

  if (loc_data.empty()) {
    return std::vector<TieBreaker<T> >();
  }

  std::vector<TieBreaker<T> > samples;
  samples.reserve(sample_cnt);

  std::uniform_int_distribution<size_t> distr(0, range - 1);
  const auto shifted_rank = TieBreaker<T>::GetOffset(myrank);
  for (size_t i = 0; i != sample_cnt; ++i) {
    auto idx = distr(async_gen);
    if (idx < loc_data.size()) {
      samples.emplace_back(loc_data[idx], shifted_rank + idx);
    }
  }

  return samples;
}
}  // namespace _internal
}  // end namespace Ams
