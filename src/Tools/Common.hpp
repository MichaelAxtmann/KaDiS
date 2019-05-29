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
#include <algorithm>
#include <cassert>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <tlx/container.hpp>

#define ASSUME_NOT(c) if (c) __builtin_unreachable()

namespace Common {
template <class T>
T ceilNthroot(T val, T base) {
  static_assert(std::is_integral<T>::value, "Integral required.");

  assert(val >= 0 && base > 0);

  if (val == 0) return 0;

  T pow = std::pow(static_cast<double>(val), 1. / base);

  size_t prod = pow;

  while (true) {
    for (T i = 1; i < base; ++i) {
      prod *= pow;
    }

    if (prod >= val) {
      return pow;
    }

    pow += 1;
    prod = pow;
  }
}

int int_floor_sqrt(int x);

template <class T>
T cube(T cube) {
  return cube * cube;
}

template <class T, class Comp>
void multiwayMerge(const std::vector<std::pair<T*, T*> >& ranges, T* target,
                   Comp comp) {
  for (const auto& range : ranges) {
    std::ignore = range;
    assert(std::is_sorted(range.first, range.second, std::forward<Comp>(comp)));
  }

  using Tree = tlx::LoserTreeCopy<false, T, decltype(comp)>;

  if (ranges.empty()) {
    return;
  }

  Tree lt(ranges.size(), comp);

  std::vector<T*> lt_iter(ranges.size());
  size_t remaining_inputs = 0;

  for (size_t i = 0; i < ranges.size(); ++i) {
    lt_iter[i] = ranges[i].first;

    if (lt_iter[i] == ranges[i].second) {
      lt.insert_start(nullptr, i, true);
    } else {
      lt.insert_start(&*lt_iter[i], i, false);
      ++remaining_inputs;
    }
  }

  lt.init();

  auto out = target;

  while (remaining_inputs != 0) {
    // take next smallest element out
    unsigned top = lt.min_source();
    *out = *lt_iter[top];
    ++out;

    ++lt_iter[top];

    if (lt_iter[top] != ranges[top].second) {
      lt.delete_min_insert(&*lt_iter[top], false);
    } else {
      lt.delete_min_insert(nullptr, true);
      --remaining_inputs;
    }
  }

  assert(std::is_sorted(target, out, comp));
}
}  // namespace Common
