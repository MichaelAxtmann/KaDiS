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
#include <cstddef>
#include <cstdlib>
#include <limits>

#include "../Tools/CommonMpi.hpp"

#include <RBC.hpp>

namespace Ams {
namespace _internal {
// Type made up of value_type and the global id.
template <class T>
struct TieBreaker {
  TieBreaker(const T& splitter, const int64_t& GID) noexcept :
    splitter(splitter),
    GID(GID) {
    assert(GID != std::numeric_limits<int64_t>::max());
  }

  TieBreaker() noexcept :
    splitter(T()),
    GID(std::numeric_limits<int64_t>::max()) { }

  // Don't change this type.

  // We need a signed type as we sometimes normalize ids by
  // substracting the minimal offset.
  using IdType = int64_t;

  // Offset of the first element of the PE with rank r.
  static IdType GetOffset(int64_t r) {
    return ((1l << sizeof(int) * 8) * r);
  }

  T splitter;
  IdType GID;

  static MPI_Datatype MpiType(const MPI_Datatype& mpi_type) {
    static MPI_Datatype splitter_type = MPI_DATATYPE_NULL;

    if (splitter_type == MPI_DATATYPE_NULL) {
      const int nitems = 2;
      int blocklengths[2] = { 1, 1 };
      MPI_Datatype types[2];
      types[0] = mpi_type;
      types[1] = Common::getMpiType<IdType>();
      MPI_Aint offsets[2] = { offsetof(TieBreaker<T>, splitter),
                              offsetof(TieBreaker<T>, GID) };

      MPI_Type_create_struct(nitems, blocklengths, offsets, types,
                             &splitter_type);
      MPI_Type_commit(&splitter_type);
    }

    return splitter_type;
  }
};

template <class T, class Comp>
class TieBreakerComparator {
 public:
  TieBreakerComparator() = delete;

  explicit TieBreakerComparator(const Comp comp_less) :
    comp_less_(comp_less)
  { }

  bool operator() (const TieBreaker<T>& a,
                   const TieBreaker<T>& b) const {
    return comp_less_(a.splitter, b.splitter) ||
           (!(b.splitter < a.splitter) && a.GID < b.GID);
    // return this->operator<(a, b);
  }

  bool operator() (const T& a,
                   const T& b) const {
    return comp_less_(a, b);
  }

  bool less(const TieBreaker<T>& a,
            const TieBreaker<T>& b) const {
    return comp_less_(a.splitter, b.splitter) ||
           (!(b.splitter < a.splitter) && a.GID < b.GID);
  }

  bool greater(const TieBreaker<T>& a,
               const TieBreaker<T>& b) const {
    return less(b, a);
  }

  bool greater_equal(const TieBreaker<T>& a,
                     const TieBreaker<T>& b) const {
    return !less(a, b);
  }

  bool smaller_equal(const TieBreaker<T>& a,
                     const TieBreaker<T>& b) const {
    return !greater(a, b);
  }

  bool equal(const TieBreaker<T>& a,
             const TieBreaker<T>& b) const {
    return a.GID == b.GID && !comp_less_(a.splitter, b.splitter) && !comp_less_(b.splitter,
                                                                                a.splitter);
  }

  bool not_equal(const TieBreaker<T>& a,
                 const TieBreaker<T>& b) const {
    return !equal(a, b);
  }

  bool greater(const TieBreaker<T>& s,
               const T& a,
               const int64_t& global_el_id) const {
    return comp_less_(a, s.splitter) || (!comp_less_(s.splitter, a) && s.GID > global_el_id);
  }

  bool less_equal(const TieBreaker<T>& s,
                  const T& a,
                  const int64_t& global_el_id) const {
    return !greater(s, a, global_el_id);
  }

  bool less(const TieBreaker<T>& s,
            const T& a,
            const int64_t& global_el_id) const {
    return comp_less_(s.splitter, a) ||
           (!comp_less_(a, s.splitter) && s.GID < global_el_id);
  }

  bool greater_equal(const TieBreaker<T>& s,
                     const T& a,
                     const int64_t& global_el_id) const {
    return !less(s, a, global_el_id);
  }

 private:
  const Comp comp_less_;
};

template <class T>
struct TieBreakerRef {
  int64_t GID;
  TieBreaker<T>* ref;

  explicit TieBreakerRef(TieBreaker<T>& split) :
    GID(split.GID),
    ref(&split) { }

  explicit TieBreakerRef(int64_t GID) :
    GID(GID),
    ref(NULL) { }

  TieBreakerRef() :
    GID(0),
    ref(NULL) { }

  TieBreakerRef& operator= (const TieBreakerRef& rhs) {
    if (this == &rhs) return *this;

    ref = rhs.ref;
    GID = ref ? ref->GID : rhs.GID;
    return *this;
  }

  bool operator< (const TieBreakerRef& splitter) const {
    return GID < splitter.GID;
  }
};
}  // end namespace _internal
}  // end namespace Ams
