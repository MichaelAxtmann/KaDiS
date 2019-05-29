/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiS).
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
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
#include <cstdint>
#include <functional>
#include <ostream>

#include "../Tools/CommonMpi.hpp"

#include <RBC.hpp>

namespace JanusSort {
template <class T>
class TbSplitter {
 public:
  TbSplitter() { }

  TbSplitter(const T& splitter, const int64_t gid) noexcept :
    splitter_(splitter),
    gid_(gid) { }

  int64_t GID() const {
    return gid_;
  }
  void setGid(int64_t gid) {
    gid_ = gid;
  }

  const T & Splitter() const {
    return splitter_;
  }
  T & Splitter() {
    return splitter_;
  }
  void setSplitter(const T& splitter) {
    splitter_ = splitter;
  }

  static MPI_Datatype mpiType(const MPI_Datatype& mpi_type) {
    const int nitems = 2;
    int blocklengths[2] = { 1, 1 };
    MPI_Datatype types[2];
    types[0] = mpi_type;
    types[1] = Common::getMpiType<decltype(gid_)>();
    MPI_Datatype mpi_tb_splitter_type;
    MPI_Aint offsets[2];

    offsets[0] = offsetof(TbSplitter<T>, splitter_);
    offsets[1] = offsetof(TbSplitter<T>, gid_);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                           &mpi_tb_splitter_type);
    MPI_Type_commit(&mpi_tb_splitter_type);

    return mpi_tb_splitter_type;
  }

  template <class Compare>
  bool compare(const TbSplitter& b, Compare&& comp) const {
    return comp(this->splitter_, b.splitter_) ||
           (!comp(b.splitter_, this->splitter_) && this->gid_ < b.gid_);
  }

 private:
  T splitter_;
  int64_t gid_;
};

template <class T>
std::ostream& operator<< (std::ostream& os,
                          const TbSplitter<T>& tbs) {
  os << "TbSplitter{ splitter=" << tbs.splitter_ << " gid=" << tbs.gid_
     << " }";

  return os;
}
}  // namespace JanusSort
