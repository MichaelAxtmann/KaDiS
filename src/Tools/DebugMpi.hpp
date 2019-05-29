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
#include <vector>
#include <iostream>
#include <iterator>

#include <RBC.hpp>

namespace std {

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    if ( !v.empty() ) {
        os << '[';
        std::copy (v.begin(), v.end(), std::ostream_iterator<T>(os, ", "));
        os << "\b\b]";
    }
    return os;
}

} // namespace std

namespace DBG {
namespace _internal {

int Get_rank(MPI_Comm comm);
int Get_rank(const RBC::Comm& comm);

} // namespace _internal
} // namespace DBG

#define DBGMSTRD(comm, dbg, X)                                         \
    if(dbg) do {std::cout                                              \
            <<  "Comm=" << comm                                                     \
            << " grank=" << DBG::_internal::Get_rank(MPI_COMM_WORLD)               \
            << " rank=" << DBG::_internal::Get_rank(comm) << ": " << X      \
            << std::endl;} while(false)

#define DBGMSTR0(comm, X) DBGMSTRD(comm, false, X)
#define DBGMSTR1(comm, X) DBGMSTRD(comm, true, X)

#define DBGMSTRROOTD(comm, dbg, X) if (DBG::_internal::Get_rank(comm) == 0) DBGMSTRD(comm, dbg, X)
#define DBGMSTRROOT0(comm, X) DBGMSTRROOTD(comm, false, X)
#define DBGMSTRROOT1(comm, X) DBGMSTRROOTD(comm, true, X)


#define DBGMD(comm, dbg, X) DBGMSTRD(comm, dbg, #X << " = " << X)

#define DBGM0(comm, X) DBGMD(comm, false, X)
#define DBGM1(comm, X) DBGMD(comm, true, X)

#define DBGMROOTD(comm, dbg, X) if (DBG::_internal::Get_rank(comm) == 0) DBGMD(comm, dbg, X)
#define DBGMROOT0(comm, X) DBGMROOTD(comm, false, X)
#define DBGMROOT1(comm, X) DBGMROOTD(comm, true, X)


#define DBGMITD(comm, dbg, X, Y)                                        \
    if(dbg) do {\
            std::cout                                                   \
                    <<  "Comm=" << comm                                 \
                    << " grank=" << DBG::_internal::Get_rank(MPI_COMM_WORLD)       \
                    << " rank=" << DBG::_internal::Get_rank(comm) << ": ";         \
            printAll(X, Y);                                             \
                std::cout << std::endl; \
        } while(false)

#define DBGMIT0(comm, X, Y) DBGMITD(comm, false, X, Y)
#define DBGMIT1(comm, X, Y) DBGMITD(comm, true, X, Y)

#define DBGMITROOTD(comm, dbg, X, Y) if (DBG::_internal::Get_rank(comm) == 0) DBGMITD(comm, dbg, X, Y)
#define DBGMITROOT0(comm, X, Y) DBGMITROOTD(comm, false, X, Y)
#define DBGMITROOT1(comm, X, Y) DBGMITROOTD(comm, true, X, Y)

