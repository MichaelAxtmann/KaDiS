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

#include <mpi.h>

namespace Tools {
template <typename T1, typename T2>
struct Tuple {
  Tuple() { }

  Tuple(const T1& first, const T2& second) :
    first(first),
    second(second)
  { }

  using first_type = T1;
  using second_type = T2;

  T1 first;
  T2 second;

  static MPI_Datatype MpiType(MPI_Datatype mpi_type1,
                              MPI_Datatype mpi_type2) {
    static MPI_Datatype mpi_type = MPI_DATATYPE_NULL;

    if (mpi_type == MPI_DATATYPE_NULL) {
      using TupleType = Tuple<T1, T2>;

      const int nitems = 2;
      int blocklengths[2] = { 1, 1 };
      MPI_Datatype types[2];
      types[0] = mpi_type1;
      types[1] = mpi_type2;
      MPI_Aint disp[2] = {
        offsetof(TupleType, first),
        offsetof(TupleType, second)
      };

      MPI_Type_create_struct(nitems, blocklengths, disp, types,
                             &mpi_type);
      MPI_Type_commit(&mpi_type);
    }

    return mpi_type;
  }
};
}  // namespace Tools
