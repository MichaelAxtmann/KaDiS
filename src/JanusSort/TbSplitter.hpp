/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiS).
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (c) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
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

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>


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
    types[1] = MPI_LONG_LONG;
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
