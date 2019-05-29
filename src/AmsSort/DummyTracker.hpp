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

#include <string>

#include "../Tools/Dummies.hpp"

namespace Ams {
namespace _internal {
class DummyTracker {
 public:
  Tools::DummyTimer various_t;
  Tools::DummyTimer splitter_allgather_scan_t;
  Tools::DummyTimer overpartition_t;
  Tools::DummyTimer exchange_t;
  Tools::DummyTimer msg_assignment_t;
  Tools::DummyTimer sampling_t;
  Tools::DummyTimer partition_t;
  Tools::DummyTimer local_sort_t;
  Tools::DummyTimer split_comm_t;

  Tools::LocalDummyMeasure<size_t> overpartition_repeats_c_;

  /*
   * Received total number of elements per level.
   *
   */
  Tools::LocalDummyMeasure<size_t> receive_volume_c_;

  /*
   * Send total number of elements per level.
   *
   */
  Tools::LocalDummyMeasure<size_t> send_volume_c_;

  /*
   * Total number of received messages per level.
   *
   */
  Tools::LocalDummyMeasure<size_t> receive_messages_c_;

  /*
   * Total number of send messages per level.
   *
   */
  Tools::LocalDummyMeasure<size_t> send_messages_c_;

  /*
   * Total number of splitters per level.
   *
   */
  Tools::LocalDummyMeasure<size_t> splitters_c_;

  /*
   * Total number of samples per level.
   *
   */
  Tools::LocalDummyMeasure<size_t> samples_c_;
};
}  // namespace _internal
}  // namespace Ams
