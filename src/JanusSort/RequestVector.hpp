/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiS).
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (C) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************************/

#pragma once

#include <cmath>
#include <vector>

#include "../../include/RBC/RBC.hpp"

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
