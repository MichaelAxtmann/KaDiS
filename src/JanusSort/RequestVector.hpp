/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiS).
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (C) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
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

#include <cmath>
#include <vector>

#include <RBC.hpp>

namespace JanusSort {
class RequestVector {
 public:
  RequestVector() :
    base_size(8) {
    req_vecs.emplace_back(base_size);
    current_vec = 0;
    current_size = 0;
  }

  void push_back(RBC::Request& req) {
    req_vecs[current_vec][current_size] = req;
    ++current_size;
    if (current_size == req_vecs[current_vec].size()) {
      ++current_vec;
      size_t size = base_size * std::pow(2, current_vec);
      req_vecs.emplace_back(size);
      current_size = 0;
    }
  }

  void testAll(int* flag) {
    *flag = 1;
    for (size_t i = 0; i < current_vec; ++i) {
      int flag_vec;
      RBC::Testall(req_vecs[i].size(), req_vecs[i].data(), &flag_vec,
                   MPI_STATUSES_IGNORE);
      if (flag_vec == 0)
        *flag = 0;
    }
    int flag_vec;
    RBC::Testall(current_size, &req_vecs[current_vec][0], &flag_vec,
                 MPI_STATUSES_IGNORE);
    if (flag_vec == 0)
      *flag = 0;
  }

  void testAll() {
    int f;
    testAll(&f);
  }

  void waitAll() {
    int flag = 0;
    while (!flag)
      testAll(&flag);
  }

 private:
  size_t base_size;
  std::vector<std::vector<RBC::Request> > req_vecs;
  size_t current_vec;
  size_t current_size;
};
}  // namespace JanusSort
