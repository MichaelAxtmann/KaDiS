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

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <vector>

#include <RBC.hpp>

namespace Common {
template <class T>
T aggregate(T val, MPI_Datatype mpi_datatype, MPI_Op mpi_op, RBC::Comm comm) {
  T sum;
  RBC::Allreduce(&val, &sum, 1, mpi_datatype, mpi_op, comm);
  return sum;
}

template <class T>
constexpr MPI_Datatype getMpiType(T v = T{ }) {
  using TT = std::remove_reference_t<T>;
  using TTT = std::remove_pointer_t<TT>;
  using TTTT = std::remove_const_t<TTT>;
  std::ignore = v;
  if constexpr (std::is_same_v<TTTT, char>) {
    return MPI_CHAR;
  } else if constexpr (std::is_same_v<TTTT, char>) {
    return MPI_CHAR;
  } else if constexpr (std::is_same_v<TTTT, signed short int>) {
    return MPI_SHORT;
  } else if constexpr (std::is_same_v<TTTT, signed int>) {
    return MPI_INT;
  } else if constexpr (std::is_same_v<TTTT, signed long int>) {
    return MPI_LONG;
  } else if constexpr (std::is_same_v<TTTT, signed long long int>) {
    return MPI_LONG_LONG_INT;
  } else if constexpr (std::is_same_v<TTTT, signed char>) {
    return MPI_SIGNED_CHAR;
  } else if constexpr (std::is_same_v<TTTT, unsigned char>) {
    return MPI_UNSIGNED_CHAR;
  } else if constexpr (std::is_same_v<TTTT, unsigned short int>) {
    return MPI_UNSIGNED_SHORT;
  } else if constexpr (std::is_same_v<TTTT, unsigned int>) {
    return MPI_UNSIGNED;
  } else if constexpr (std::is_same_v<TTTT, unsigned long int>) {
    return MPI_UNSIGNED_LONG;
  } else if constexpr (std::is_same_v<TTTT, unsigned long long int>) {
    return MPI_UNSIGNED_LONG_LONG;
  } else if constexpr (std::is_same_v<TTTT, float>) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same_v<TTTT, double>) {
    return MPI_DOUBLE;
  } else if constexpr (std::is_same_v<TTTT, long double>) {
    return MPI_LONG_DOUBLE;
  } else if constexpr (std::is_same_v<TTTT, wchar_t>) {
    return MPI_WCHAR;
  } else if constexpr (std::is_same_v<TTTT, int8_t>) {
    return MPI_INT8_T;
  } else if constexpr (std::is_same_v<TTTT, int16_t>) {
    return MPI_INT16_T;
  } else if constexpr (std::is_same_v<TTTT, int32_t>) {
    return MPI_INT32_T;
  } else if constexpr (std::is_same_v<TTTT, int64_t>) {
    return MPI_INT64_T;
  } else if constexpr (std::is_same_v<TTTT, uint8_t>) {
    return MPI_UINT8_T;
  } else if constexpr (std::is_same_v<TTTT, uint16_t>) {
    return MPI_UINT16_T;
  } else if constexpr (std::is_same_v<TTTT, uint32_t>) {
    return MPI_UINT32_T;
  } else if constexpr (std::is_same_v<TTTT, uint64_t>) {
    return MPI_UINT64_T;
  } else {
    static_assert(!std::is_integral_v<TTTT>,
                  "Type not supported. An integral type is expected.");
    return MPI_INT;
  }
}

template <class T>
constexpr MPI_Datatype getMpiType(const std::vector<T>& v) {
  std::ignore = v;
  return getMpiType<T>();
}
}  // namespace Common
