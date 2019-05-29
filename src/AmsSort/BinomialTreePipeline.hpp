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

#include <cstddef>

#include "../Tools/CommonMpi.hpp"

#include <RBC.hpp>

namespace BinomialTreePipeline {
template <class value_type>
value_type bcast(value_type el, bool pipe_own_el, MPI_Datatype mpi_datatype,
                 RBC::Comm comm);

namespace _internal {
template <class value_type>
struct SegScanPair {
  using template_type = value_type;
  using inner_type = int;

  template_type pipe_val_;
  template_type new_val_;
  inner_type myrank_;
  inner_type pipe_;
  inner_type nvid_;
};

template <class value_type>
void mpiPipeline(SegScanPair<value_type>* in, SegScanPair<value_type>* inout,
                 int* len, MPI_Datatype*) {
  int i;
  for (i = 0; i < *len; ++i) {
    auto& in_el = in[i];
    auto& inout_el = inout[i];
    SegScanPair<value_type> new_el;

    assert(in_el.myrank_ < inout_el.myrank_);
    new_el.pipe_ = in_el.pipe_ + inout_el.pipe_;
    new_el.myrank_ = inout_el.myrank_;

    if (inout_el.pipe_ > 0) {
      new_el.pipe_val_ = inout_el.pipe_val_;
      if (in_el.pipe_ > 0 && in_el.myrank_ >= inout_el.nvid_) {
        new_el.new_val_ = in_el.pipe_val_;
        new_el.nvid_ = in_el.myrank_;
      } else {
        new_el.new_val_ = inout_el.new_val_;
        new_el.nvid_ = inout_el.nvid_;
      }
    } else {
      new_el.pipe_val_ = in_el.pipe_val_;
      if (in_el.pipe_ > 0) {
        new_el.new_val_ = in_el.pipe_val_;
        new_el.nvid_ = in_el.myrank_;
      }
    }
    inout[i] = new_el;
  }
}
}  // end namespace _internal

/* @brief Partially pipeline element 'el'.
 *
 * Assume that process 'i' passes pipe_own_el = true and process
 * 'j' (j > i) is the next process that passes 'pipe_own_el =
 * true'. Then, process ['i+1'.. 'j'] returns 'el[i]'.
 *
 * E.g.:
 * Rank 0: pipe_own_el=1 el=0 result=0
 * Rank 1: pipe_own_el=0 el=1 result=0
 * Rank 2: pipe_own_el=0 el=2 result=0
 * Rank 3: pipe_own_el=1 el=3 result=0
 * Rank 4: pipe_own_el=0 el=4 result=3
 * Rank 5: pipe_own_el=0 el=5 result=3
 * Rank 6: pipe_own_el=1 el=6 result=3
 * Rank 7: pipe_own_el=0 el=7 result=6
 */
template <class value_type>
value_type bcast(value_type el, bool pipe_own_el, MPI_Datatype mpi_datatype,
                 RBC::Comm comm) {
  using SegScanPair = _internal::SegScanPair<value_type>;

  int myrank;
  RBC::Comm_rank(comm, &myrank);
  SegScanPair a, answer;

  static MPI_Op myOp = MPI_OP_NULL;
  static MPI_Datatype my_type = MPI_DATATYPE_NULL;

  if (my_type == MPI_DATATYPE_NULL) {
    MPI_Datatype types[5];
    int blocklen[5] = { 1, 1, 1, 1, 1 };
    MPI_Aint disp[5] = {
      offsetof(SegScanPair, pipe_val_),
      offsetof(SegScanPair, new_val_),
      offsetof(SegScanPair, myrank_),
      offsetof(SegScanPair, pipe_),
      offsetof(SegScanPair, nvid_)
    };

    types[0] = mpi_datatype;
    types[1] = mpi_datatype;
    types[2] = Common::getMpiType<typename SegScanPair::inner_type>();
    types[3] = Common::getMpiType<typename SegScanPair::inner_type>();
    types[4] = Common::getMpiType<typename SegScanPair::inner_type>();

    MPI_Type_create_struct(5, blocklen, disp, types, &my_type);
    MPI_Type_commit(&my_type);

    MPI_Op_create(reinterpret_cast<MPI_User_function*>(_internal::mpiPipeline<value_type>),
                  0, &myOp);
  }

  a.pipe_val_ = el;
  a.new_val_ = el;
  a.myrank_ = myrank;
  a.nvid_ = 0;
  a.pipe_ = pipe_own_el ? 1 : 0;
  RBC::Scan(&a, &answer, 1, my_type, myOp, comm);
  return answer.new_val_;
}
}  // end namespace BinomialTreePipeline
