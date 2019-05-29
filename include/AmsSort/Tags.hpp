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

#include "../RFis/Tags.hpp"

#ifndef KADIS_AMS_ONE_FACTOR_GROUP_BASED_EXCHANGE_TAG
#define KADIS_AMS_ONE_FACTOR_GROUP_BASED_EXCHANGE_TAG 10000
#endif

#ifndef KADIS_AMS_CAPACITY_DISTRIBUTE_SCHEMA_TAG
#define KADIS_AMS_CAPACITY_DISTRIBUTE_SCHEMA_TAG 10001
#endif

#ifndef KADIS_AMS_GENERAL_TAG
#define KADIS_AMS_GENERAL_TAG 10002
#endif

namespace Ams {
template <
  int OneFactorGroupBasedExchange_ = KADIS_AMS_ONE_FACTOR_GROUP_BASED_EXCHANGE_TAG,
  int CapacityDistributeSchema_ = KADIS_AMS_CAPACITY_DISTRIBUTE_SCHEMA_TAG,
  int General_ = KADIS_AMS_GENERAL_TAG,
  class RFisTags_ = RFis::Tags<>
  >
struct Tags {
  using RFisTags= RFisTags_;
    static constexpr const int kOneFactorGroupBasedExchange =
        OneFactorGroupBasedExchange_;
    static constexpr const int kCapacityDistributeSchema =
        CapacityDistributeSchema_;
    static constexpr const int kGeneral = General_;
};


#undef KADIS_AMS_ONE_FACTOR_GROUP_BASED_EXCHANGE_TAG
#undef KADIS_AMS_CAPACITY_DISTRIBUTE_SCHEMA_TAG
#undef KADIS_AMS_GENERAL_TAG
}  // namespace Ams
