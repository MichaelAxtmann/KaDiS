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
#include <cstring>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include <RBC.hpp>
#include <tlx/algorithm.hpp>
#include <tlx/math.hpp>

#include "../../include/AmsSort/Tags.hpp"
#include "../../include/Tools/MpiTuple.hpp"
#include "../Tools/CommonMpi.hpp"
#include "DistrRange.hpp"

namespace Alltoallv {
template <class Tracker, class T>
std::vector<std::pair<T*, T*> > MPIAlltoallvRanges(Tracker& tracker, std::vector<T>& v_send,
                                                   const std::vector<int>& sendcounts,
                                                   std::vector<T>& v_recv,
                                                   MPI_Datatype mpi_datatype,
                                                   const RBC::Comm& comm) {
  const size_t size = comm.getSize();

  std::vector<int> sdispls(size + 1, 0);

  std::vector<int> recvcounts(size, 0);
  std::vector<int> rdispls(size + 1, 0);

  tlx::exclusive_scan(sendcounts.data(), sendcounts.data() + size,
                      sdispls.data(), 0, std::plus<>{ });

  MPI_Alltoall(sendcounts.data(), 1, Common::getMpiType(sendcounts),
               recvcounts.data(), 1, Common::getMpiType(recvcounts), comm.get());

  tlx::exclusive_scan(recvcounts.data(), recvcounts.data() + size,
                      rdispls.data(), 0, std::plus<>{ });

  v_recv.resize(rdispls.back());

  MPI_Alltoallv(v_send.data(),
                sendcounts.data(),
                sdispls.data(),
                mpi_datatype,
                v_recv.data(),
                recvcounts.data(),
                rdispls.data(),
                mpi_datatype,
                comm.get());

  tracker.receive_messages_c_.add(size);
  tracker.receive_volume_c_.add(rdispls.back());

  tracker.send_messages_c_.add(size);
  tracker.send_volume_c_.add(sdispls.back());

  std::vector<std::pair<T*, T*> > ret;
  ret.reserve(size);
  for (size_t i = 0; i != size; ++i) {
    ret.emplace_back(v_recv.data() + rdispls[i], v_recv.data() + rdispls[i + 1]);
  }

  return ret;
}

template <class Tracker, class T>
void MPIAlltoallv(Tracker& tracker, std::vector<T>& v_send, const std::vector<int>& sendcounts,
                  std::vector<T>& v_recv, MPI_Datatype mpi_datatype, const RBC::Comm& comm) {
  const size_t size = comm.getSize();

  std::vector<int> sdispls(size, 0);

  std::vector<int> recvcounts(size, 0);
  std::vector<int> rdispls(size, 0);

  tlx::exclusive_scan(sendcounts.data(), sendcounts.data() + size - 1,
                      sdispls.data(), 0, std::plus<>{ });

  MPI_Alltoall(sendcounts.data(), 1, Common::getMpiType(sendcounts),
               recvcounts.data(), 1, Common::getMpiType(recvcounts), comm.get());

  tlx::exclusive_scan(recvcounts.data(), recvcounts.data() + size - 1,
                      rdispls.data(), 0, std::plus<>{ });

  v_recv.resize(rdispls.back() + recvcounts.back());

  MPI_Alltoallv(v_send.data(),
                sendcounts.data(),
                sdispls.data(),
                mpi_datatype,
                v_recv.data(),
                recvcounts.data(),
                rdispls.data(),
                mpi_datatype,
                comm.get());

  tracker.receive_messages_c_.add(size);
  tracker.receive_volume_c_.add(rdispls.back() + recvcounts.back());

  tracker.send_messages_c_.add(size);
  tracker.send_volume_c_.add(sdispls.back() + sendcounts.back());
}

namespace _internal {
inline int FirstTarget(int nprocs, int myrank) {
  return myrank > 0 ? nprocs - myrank : 0;
}


/* @brief Removes elements which pred evaluates to true from the input and writes them to the output.
 *
 * @param first Input iterator to the first position.
 * @param last Input iterator to the final position.
 * @param result_true Output iterator to the first position.
 * @param pred Binary function that accpets an element pointed by InputIterator.
 * @return Final positions of the input and output positions.
 */

// Similar to std::partition_copy but the 'false' output is written to the input.
template <class InputIterator, class OutputIterator, class UnaryPredicate>
std::pair<InputIterator, OutputIterator> partition_copy(InputIterator first, InputIterator last,
                                                        OutputIterator result_true,
                                                        UnaryPredicate pred) {
  InputIterator result_false = first;

  while (first != last) {
    if (pred(*first)) {
      *result_true = *first;
      ++result_true;
    } else {
      *result_false = *first;
      ++result_false;
    }
    ++first;
  }
  return { result_false, result_true };
}

/* @brief Reorders msgs by comparing the member variable 'pe'
 * according to the onefactor message exchange order.
 *
 * @param msgs Messages must already be sorted by variable 'pe'.
 */
template <class T, class ValExtractor>
std::vector<T> OneFactorReorder(const std::vector<T>& msgs, int myrank, int nprocs,
                                ValExtractor extr) {
  std::vector<T> ordered_msgs;
  ordered_msgs.reserve(msgs.size());

  // Messages are sorted by target process.
  assert(std::is_sorted(msgs.begin(), msgs.end(),
                        [extr](const auto& left, const auto& right) {
        return extr(left) < extr(right);
      }));

  const auto first_target = FirstTarget(nprocs, myrank);
  const auto middle = std::lower_bound(msgs.begin(), msgs.end(), first_target,
                                       [extr](const auto& val, const auto& first_target) {
        return extr(val) < first_target;
      });

  ordered_msgs.insert(ordered_msgs.end(), middle, msgs.end());
  ordered_msgs.insert(ordered_msgs.end(), msgs.begin(), middle);

  return ordered_msgs;
}
}  // end namespace _internal

/* @brief Message exchange
 *
 * @param out_msgs Outgoing messages of size zero are not allowed!
 * The outgoing messages must be sorted by their target processes.
 *
 * Use this function if you know an upper bound of incoming
 * elements (no resize of the receive vector) and if you know an
 * upper bound of incoming messages.
 *
 * @param v_recv Receive vector. The size(!) of the vector must be
 * equal or larger than the number of elements which we will
 * receive.
 *
 */
template <class AmsTag, class Tracker, class T>
void exchangeWithoutRecvSizes(Tracker&& tracker,
                              const std::vector<T>& v_send,
                              const DistrRanges& out_msgs,
                              std::vector<T>& v_recv,
                              size_t max_incoming_msgs,
                              size_t max_incoming_els,
                              MPI_Datatype mpi_datatype,
                              const RBC::Comm& comm) {
  v_recv.resize(max_incoming_els);

  // No messages of size zero.
  assert(std::find_if(out_msgs.begin(), out_msgs.end(),
                      [](const auto& msg) {
      return msg.size == 0;
    }) == out_msgs.end());

  // Messages are sorted by target process.
  assert(std::is_sorted(out_msgs.begin(), out_msgs.end(),
                        [](const auto& left, const auto& right) {
      return left.pe < right.pe;
    }));

  const auto reordered_out_msgs = _internal::OneFactorReorder(out_msgs, comm.getRank(),
                                                              comm.getSize(), [](const auto& val) {
      return val.pe;
    });

  std::vector<MPI_Request> send_requests;
  send_requests.reserve(reordered_out_msgs.size());

  DistrRanges local_ranges;

  for (const auto& out_msg : reordered_out_msgs) {
    if (out_msg.pe != comm.getRank()) {
      send_requests.emplace_back();
      RBC::Issend(v_send.data() + out_msg.offset, out_msg.size, mpi_datatype,
                  out_msg.pe, AmsTag::kOneFactorGroupBasedExchange, comm,
                  &send_requests.back());
    } else {
      local_ranges.emplace_back(out_msg);
    }
  }

  // We don't know how many messages we will receive. We use a list
  // as we want (non-amortized) constant push-back.
  std::vector<MPI_Request> recv_requests;
  recv_requests.reserve(max_incoming_msgs);
  T* recv_ptr = v_recv.data();
  int num_recv_msgs = 0;

  // As long as my send operations are not finished, I test
  // send/recv requests and wait for incoming messages.
  size_t num_send_completed = 0;
  size_t num_recv_completed = 0;
  while (num_send_completed < send_requests.size()) {
    MPI_Status status;
    int flag;
    RBC::Iprobe(MPI_ANY_SOURCE, AmsTag::kOneFactorGroupBasedExchange,
                comm, &flag, &status);
    if (flag) {
      int size;
      MPI_Get_count(&status, mpi_datatype, &size);
      auto source = comm.MpiRankToRangeRank(status.MPI_SOURCE);
      recv_requests.emplace_back();
      assert(recv_requests.size() <= max_incoming_msgs);
      assert(recv_ptr + size - v_recv.data() <= static_cast<std::ptrdiff_t>(v_recv.size()));
      RBC::Irecv(recv_ptr, size, mpi_datatype, source,
                 AmsTag::kOneFactorGroupBasedExchange, comm,
                 &recv_requests.back());
      recv_ptr += size;
      ++num_recv_msgs;
    }

    if (num_send_completed < send_requests.size()) {
      int succ = 0;
      MPI_Test(send_requests.data() + num_send_completed, &succ, MPI_STATUS_IGNORE);
      if (succ) {
        ++num_send_completed;
      }
    }

    if (num_recv_completed < recv_requests.size()) {
      int succ = 0;
      MPI_Test(recv_requests.data() + num_recv_completed, &succ, MPI_STATUS_IGNORE);
      if (succ) {
        ++num_recv_completed;
      }
    }
  }

  // My outgoing messages have been received. Start nonblocking
  // barrier while testing receive requests and waiting for new
  // incoming messages.
  RBC::Request rbarrier;
  RBC::Ibarrier(comm, &rbarrier);
  int done = 0;
  while (!done) {
    MPI_Status status;
    int flag;
    RBC::Iprobe(MPI_ANY_SOURCE, AmsTag::kOneFactorGroupBasedExchange,
                comm, &flag, &status);
    if (flag) {
      int size;
      MPI_Get_count(&status, mpi_datatype, &size);
      const auto source = comm.MpiRankToRangeRank(status.MPI_SOURCE);
      assert(recv_requests.size() < max_incoming_msgs);
      recv_requests.emplace_back();
      assert(recv_ptr + size <= v_recv.data() + v_recv.size());
      RBC::Irecv(recv_ptr, size, mpi_datatype, source,
                 AmsTag::kOneFactorGroupBasedExchange, comm,
                 &recv_requests.back());
      recv_ptr += size;
      ++num_recv_msgs;
    }

    if (num_recv_completed < recv_requests.size()) {
      int succ = 0;
      MPI_Test(recv_requests.data() + num_recv_completed, &succ, MPI_STATUS_IGNORE);
      if (succ) {
        ++num_recv_completed;
      }
    }

    RBC::Test(&rbarrier, &done, MPI_STATUS_IGNORE);
  }

  // As the nonblocking barrier is finished, each outgoing message
  // of all processes are received or receiving has been
  // started. Thus, we just have to finish the pending requests.
  if (num_recv_completed < recv_requests.size()) {
    MPI_Waitall(recv_requests.size() - num_recv_completed,
                recv_requests.data() + num_recv_completed,
                MPI_STATUSES_IGNORE);
  }

  // Copy local data.
  for (const auto& range : local_ranges) {
    assert(recv_ptr + range.size <= v_recv.data() + v_recv.size());
    recv_ptr = std::copy(v_send.data() + range.offset,
                         v_send.data() + range.offset + range.size,
                         recv_ptr);
  }

  // This barrier avoids influences of the execution of this
  // function and further executions of this functions. Without a
  // barrier, we could receive a message (from a further execution
  // of this function), the barrier has already been finished but we
  // did not execute the final test.
  RBC::Barrier(comm);

  // Shrink receive vector to receive volume.
  v_recv.resize(recv_ptr - v_recv.data());

  tracker.send_messages_c_.add(reordered_out_msgs.size());
  tracker.send_volume_c_.add(v_send.size());

  tracker.receive_messages_c_.add(num_recv_msgs);
  tracker.receive_volume_c_.add(v_recv.size());
}

/* @brief Message exchange
 *
 * Message exchange. Messages are sent and received according to
 * the one-factor message-exchange ordering. Number of incoming
 * elements is unknown.
 *
 * Use this function if you know an upper bound of incoming
 * elements (no resize of the receive vector) and if you don't
 * know the total number of incoming messages (handled by a list
 * of receive requests).  @param out_msgs Outgoing messages of
 * size zero are not allowed!  The outgoing messages must be
 * sorted by their target processes.
 *
 * @param v_recv Receive vector. The size of the vector must be
 * equal or larger than the number of elements which we will
 * receive.
 */
template <class AmsTag, class Tracker, class T>
std::vector<std::pair<T*, T*> >
exchangeWithoutRecvSizesReturnRanges(Tracker&& tracker, const std::vector<T>& v_send,
                                     const DistrRanges& out_msgs, std::vector<T>& v_recv,
                                     size_t max_num_recv_msgs, size_t max_num_recv_els,
                                     MPI_Datatype mpi_datatype, const RBC::Comm& comm) {
  v_recv.resize(max_num_recv_els);

  // No messages of size zero.
  assert(std::find_if(out_msgs.begin(), out_msgs.end(),
                      [](const auto& msg) {
      return msg.size == 0;
    }) == out_msgs.end());

  // Messages are sorted by target process.
  assert(std::is_sorted(out_msgs.begin(), out_msgs.end(),
                        [](const auto& left, const auto& right) {
      return left.pe < right.pe;
    }));

  const auto reordered_out_msgs = _internal::OneFactorReorder(out_msgs, comm.getRank(),
                                                              comm.getSize(), [](const auto& val) {
      return val.pe;
    });

  std::vector<MPI_Request> send_requests;
  send_requests.reserve(reordered_out_msgs.size());

  DistrRanges local_ranges;

  for (const auto& out_msg : reordered_out_msgs) {
    if (out_msg.pe != comm.getRank()) {
      send_requests.emplace_back();
      RBC::Issend(v_send.data() + out_msg.offset, out_msg.size, mpi_datatype,
                  out_msg.pe, AmsTag::kOneFactorGroupBasedExchange, comm,
                  &send_requests.back());
    } else {
      local_ranges.emplace_back(out_msg);
    }
  }

  // We don't know how many messages we will receive. We use a list
  // as MPI_Request is not allowed to be copied and we want
  // (non-amortized) push-back.
  std::vector<MPI_Request> recv_requests;
  recv_requests.reserve(max_num_recv_msgs);
  auto recv_ptr = v_recv.data();
  int num_recv_msgs = 0;
  std::vector<std::pair<T*, T*> > recv_ranges;
  recv_ranges.reserve(max_num_recv_msgs);

  // As long as my send operations are not finished, I test
  // send/recv requests and wait for incoming messages.
  size_t num_send_completed = 0;
  size_t num_recv_completed = 0;
  while (num_send_completed < send_requests.size()) {
    MPI_Status status;
    int flag;
    RBC::Iprobe(MPI_ANY_SOURCE, AmsTag::kOneFactorGroupBasedExchange,
                comm, &flag, &status);
    if (flag) {
      int size;
      MPI_Get_count(&status, mpi_datatype, &size);
      auto source = comm.MpiRankToRangeRank(status.MPI_SOURCE);
      recv_requests.emplace_back();
      assert(recv_ptr + size - v_recv.data() <= static_cast<std::ptrdiff_t>(v_recv.size()));
      RBC::Irecv(recv_ptr, size, mpi_datatype, source,
                 AmsTag::kOneFactorGroupBasedExchange, comm,
                 &recv_requests.back());
      recv_ranges.emplace_back(recv_ptr, recv_ptr + size);
      recv_ptr += size;
      ++num_recv_msgs;
    }

    if (num_send_completed < send_requests.size()) {
      int succ = 0;
      MPI_Test(send_requests.data() + num_send_completed, &succ, MPI_STATUS_IGNORE);
      if (succ) {
        ++num_send_completed;
      }
    }

    if (num_recv_completed < recv_requests.size()) {
      int succ = 0;
      MPI_Test(recv_requests.data() + num_recv_completed, &succ, MPI_STATUS_IGNORE);
      if (succ) {
        ++num_recv_completed;
      }
    }
  }

  // My outgoing messages have been received. Start nonblocking
  // barrier while testing receive requests and waiting for new
  // incoming messages.
  RBC::Request rbarrier;
  RBC::Ibarrier(comm, &rbarrier);
  int done = 0;
  while (!done) {
    MPI_Status status;
    int flag;
    RBC::Iprobe(MPI_ANY_SOURCE, AmsTag::kOneFactorGroupBasedExchange,
                comm, &flag, &status);
    if (flag) {
      int size;
      MPI_Get_count(&status, mpi_datatype, &size);
      const auto source = comm.MpiRankToRangeRank(status.MPI_SOURCE);
      recv_requests.emplace_back();
      assert(recv_ptr + size <= v_recv.data() + v_recv.size());
      RBC::Irecv(recv_ptr, size, mpi_datatype, source,
                 AmsTag::kOneFactorGroupBasedExchange, comm,
                 &recv_requests.back());
      recv_ranges.emplace_back(recv_ptr, recv_ptr + size);
      recv_ptr += size;
      ++num_recv_msgs;
    }

    if (num_recv_completed < recv_requests.size()) {
      int succ = 0;
      MPI_Test(recv_requests.data() + num_recv_completed, &succ, MPI_STATUS_IGNORE);
      if (succ) {
        ++num_recv_completed;
      }
    }

    RBC::Test(&rbarrier, &done, MPI_STATUS_IGNORE);
  }

  // As the nonblocking barrier is finished, each outgoing message
  // of all processes are received or receiving has been
  // started. Thus, we just have to finish the pending requests.
  if (num_recv_completed < recv_requests.size()) {
    MPI_Waitall(recv_requests.size() - num_recv_completed,
                recv_requests.data() + num_recv_completed,
                MPI_STATUSES_IGNORE);
  }

  // Copy local data.
  for (const auto& range : local_ranges) {
    assert(recv_ptr + range.size <= v_recv.data() + v_recv.size());
    std::memcpy(recv_ptr,
                v_send.data() + range.offset,
                range.size * sizeof(T));
    recv_ranges.emplace_back(recv_ptr, recv_ptr + range.size);
    recv_ptr += range.size;
  }

  // This barrier avoids influences of the execution and subsequent executions.
  //
  // Example: All processes call the implicit barrier. Thread 0
  // finishes the barrier (and this function) and invokes this
  // function again, sending to thread 1. Thread 1 does not finish
  // the barrier and tests for a new incoming message. In this case,
  // thread 1 receives a message from a different invocation of this
  // function.
  RBC::Barrier(comm);

  // Shrink receive vector to receive volume.
  v_recv.resize(recv_ptr - v_recv.data());

  tracker.send_messages_c_.add(reordered_out_msgs.size());
  tracker.send_volume_c_.add(v_send.size());

  tracker.receive_messages_c_.add(num_recv_msgs);
  tracker.receive_volume_c_.add(v_recv.size());

  return recv_ranges;
}

/* @brief Element-wise Alltoallv
 *
 * Idea: A message m_ij belonging to process j which is stored on
 * process i has to be shifted to the right by s_ij = (j - i + nprocs)
 * % nprocs processes. In iteration r in [ 0, ceil(log(nprocs)) ),
 * m_ij is sent to processor (i + (1 << r)) % nprocs, if the r'th bit
 * of s_ij is set. Observation: After iteration r, the bits [0, r] of
 * the new shifts s_ij are zero.
 *
 * @param data Array of tuples. The first value is the target.
 * @param mpi_type MPI datatype of Tools::Tuple<int, T>.
 *
 * A variant of the log-latency store-and-forward algorithm by
 * Jehoshua Bruck et al, IEEE TPDS, Nov. 1997.
 * This algorithm is also used by MPICH for MPI_Alltoall if the message length is small.
 */
template <class T>
void storeAndForward(std::vector<Tools::Tuple<int, T> >& data, MPI_Datatype mpi_type,
                     int tag, const RBC::Comm& comm) {
  const size_t nprocs = comm.getSize();
  const size_t myrank = comm.getRank();

  const int its = tlx::integer_log2_ceil(comm.getSize());

  std::vector<Tools::Tuple<int, T> > send_data;

  for (int it = 0; it != its; ++it) {
    // Partition data
    const int mask = 1 << it;
    send_data.resize(data.size());
    const auto[end_keep, end_send] =
      _internal::partition_copy(data.begin(), data.end(), send_data.begin(),
                                [mask, myrank, nprocs](const auto& el) {
        const int total_shift = (el.first - myrank + nprocs) % nprocs;
        return total_shift & mask;
      });

    const int target = (myrank + mask) % nprocs;
    const int source = (myrank - mask + nprocs) % nprocs;

    int keep_cnt = end_keep - data.begin();
    int send_cnt = end_send - send_data.begin();
    int recv_cnt = 0;
    RBC::Sendrecv(&send_cnt, 1, Common::getMpiType(send_cnt), target, tag,
                  &recv_cnt, 1, Common::getMpiType(recv_cnt), source, tag,
                  comm, MPI_STATUS_IGNORE);

    data.resize(keep_cnt + recv_cnt);
    RBC::Sendrecv(send_data.data(), send_cnt, mpi_type, target, tag,
                  data.data() + keep_cnt, recv_cnt, mpi_type, source, tag,
                  comm, MPI_STATUS_IGNORE);
  }
}

template <class AmsTag, class Tracker, class T>
void exchangeWithRecvSizes(Tracker&& tracker, const std::vector<T>& v_send,
                           const DistrRanges& out_msgs, std::vector<T>& v_recv,
                           MPI_Datatype mpi_datatype, const RBC::Comm& comm) {
  // No messages of size zero.
  assert(std::find_if(out_msgs.begin(), out_msgs.end(),
                      [](const auto& msg) {
      return msg.size == 0;
    }) == out_msgs.end());

  // Messages are sorted by target process.
  assert(std::is_sorted(out_msgs.begin(), out_msgs.end(),
                        [](const auto& left, const auto& right) {
      return left.pe < right.pe;
    }));

  /*
   * Exchange receive sizes.
   */

  using InnerType = Tools::Tuple<int, int>;
  MPI_Datatype inner_type = InnerType::MpiType(
    Common::getMpiType<InnerType::first_type>(),
    Common::getMpiType<InnerType::second_type>());

  using InMsgType = Tools::Tuple<int, InnerType>;
  MPI_Datatype in_msg_mpi_type = InMsgType::MpiType(
    Common::getMpiType<InMsgType::first_type>(),
    inner_type);

  // Construct vector to exchange sizes.
  std::vector<InMsgType> in_msgs;
  in_msgs.reserve(out_msgs.size());
  for (const auto& out_msg : out_msgs) {
    in_msgs.emplace_back(out_msg.pe,
                         Tools::Tuple<int, int>{ comm.getRank(), static_cast<int>(out_msg.size) });
  }

  // Exchange sizes and then sort sizes by the source process.
  storeAndForward(in_msgs, in_msg_mpi_type, AmsTag::kGeneral, comm);
  std::sort(in_msgs.begin(), in_msgs.end(), [](const auto& l, const auto& r) {
      return l.second.first < r.second.first;
    });


  /*
   * Post receive messages
   */

  // Calculate receive volume.
  const int num_recv_els = std::accumulate(in_msgs.begin(), in_msgs.end(), int{ 0 },
                                           [](const int sum, const auto& in_msg) {
      return sum + in_msg.second.second;
    });

  // Adjust receive vector.
  v_recv.resize(num_recv_els);

  // Allocate enough requests for send/receive messages.
  std::vector<MPI_Request> requests;
  requests.reserve(out_msgs.size() + in_msgs.size());

  // Calculate the first message which we will send according to the
  // 1-factor data exchange scheme.
  const int first_target = _internal::FirstTarget(comm.getSize(), comm.getRank());
  const auto first_recv_msg = std::lower_bound(in_msgs.begin(), in_msgs.end(), first_target,
                                               [](const InMsgType& msg, const int& target) {
      return msg.second.first < target;
    });

  // Post receive requests.
  auto recv_begin = v_recv.data();
  for (auto in_msg = first_recv_msg; in_msg != in_msgs.end(); ++in_msg) {
    const int size = in_msg->second.second;
    const int source = in_msg->second.first;

    assert(requests.capacity() > requests.size());
    RBC::Irecv(recv_begin,
               size,
               mpi_datatype,
               source,
               AmsTag::kGeneral,
               comm,
               &requests.emplace_back());

    recv_begin += size;
  }

  // Post receive requests.
  for (auto in_msg = in_msgs.begin(); in_msg != first_recv_msg; ++in_msg) {
    const auto size = in_msg->second.second;
    const auto source = in_msg->second.first;

    assert(requests.capacity() > requests.size());
    RBC::Irecv(recv_begin,
               size,
               mpi_datatype,
               source,
               AmsTag::kGeneral,
               comm,
               &requests.emplace_back());

    recv_begin += size;
  }

  // Calculate the first message which we will send according to the
  // 1-factor data exchange scheme.
  const auto first_send_msg = std::lower_bound(out_msgs.begin(), out_msgs.end(), first_target,
                                               [](const auto& msg, const auto target) {
      return msg.pe < target;
    });

  // Post send requests.
  for (auto out_msg = first_send_msg; out_msg != out_msgs.end(); ++out_msg) {
    const auto size = out_msg->size;
    const auto offset = out_msg->offset;
    const auto target = out_msg->pe;

    assert(requests.capacity() > requests.size());
    RBC::Isend(v_send.data() + offset,
               size,
               mpi_datatype,
               target,
               AmsTag::kGeneral,
               comm,
               &requests.emplace_back());
  }

  // Post send requests.
  for (auto out_msg = out_msgs.begin(); out_msg != first_send_msg; ++out_msg) {
    const auto size = out_msg->size;
    const auto offset = out_msg->offset;
    const auto target = out_msg->pe;

    assert(requests.capacity() > requests.size());
    RBC::Isend(v_send.data() + offset,
               size,
               mpi_datatype,
               target,
               AmsTag::kGeneral,
               comm,
               &requests.emplace_back());
  }

  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

  tracker.send_messages_c_.add(out_msgs.size());
  tracker.send_volume_c_.add(v_send.size());

  tracker.receive_messages_c_.add(in_msgs.size());
  tracker.receive_volume_c_.add(v_recv.size());
}

template <class AmsTag, class Tracker, class T> 
std::vector<std::pair<T*, T*> > exchangeWithRecvSizesReturnRanges(Tracker&& tracker,
                                                                  const std::vector<T>& v_send,
                                                                  const DistrRanges& out_msgs,
                                                                  std::vector<T>& v_recv,
                                                                  MPI_Datatype mpi_datatype,
                                                                  const RBC::Comm& comm) {
  // No messages of size zero.
  assert(std::find_if(out_msgs.begin(), out_msgs.end(),
                      [](const auto& msg) {
      return msg.size == 0;
    }) == out_msgs.end());

  // Messages are sorted by target process.
  assert(std::is_sorted(out_msgs.begin(), out_msgs.end(),
                        [](const auto& left, const auto& right) {
      return left.pe < right.pe;
    }));

  /*
   * Exchange receive sizes.
   */

  using InnerType = Tools::Tuple<int, int>;
  MPI_Datatype inner_type = InnerType::MpiType(
    Common::getMpiType<InnerType::first_type>(),
    Common::getMpiType<InnerType::second_type>());

  using InMsgType = Tools::Tuple<int, InnerType>;
  MPI_Datatype in_msg_mpi_type = InMsgType::MpiType(
    Common::getMpiType<InMsgType::first_type>(),
    inner_type);

  // Construct vector to exchange sizes.
  std::vector<InMsgType> in_msgs;
  in_msgs.reserve(out_msgs.size());
  for (const auto& out_msg : out_msgs) {
    in_msgs.emplace_back(out_msg.pe,
                         Tools::Tuple<int, int>{ comm.getRank(), static_cast<int>(out_msg.size) });
  }

  // Exchange sizes and sort then sizes by the source process.
  storeAndForward(in_msgs, in_msg_mpi_type, AmsTag::kGeneral, comm);
  std::sort(in_msgs.begin(), in_msgs.end(), [](const auto& l, const auto& r) {
      return l.second.first < r.second.first;
    });

  std::vector<std::pair<T*, T*> > recv_ranges;
  recv_ranges.reserve(in_msgs.size());


  /*
   * Post receive messages
   */

  // Calculate receive volume.
  const int num_recv_els = std::accumulate(in_msgs.begin(), in_msgs.end(), int{ 0 },
                                           [](const int sum, const auto& in_msg) {
      return sum + in_msg.second.second;
    });

  // Adjust receive vector.
  v_recv.resize(num_recv_els);

  // Allocate enough requests for send/receive messages.
  std::vector<MPI_Request> requests;
  requests.reserve(out_msgs.size() + in_msgs.size());

  // Calculate the first message which we will send according to the
  // 1-factor data exchange scheme.
  const int first_target = _internal::FirstTarget(comm.getSize(), comm.getRank());
  const auto first_recv_msg = std::lower_bound(in_msgs.begin(), in_msgs.end(), first_target,
                                               [](const InMsgType& msg, const int& target) {
      return msg.second.first < target;
    });

  // Post receive requests.
  auto recv_begin = v_recv.data();
  for (auto in_msg = first_recv_msg; in_msg != in_msgs.end(); ++in_msg) {
    const int size = in_msg->second.second;
    const int source = in_msg->second.first;

    assert(requests.capacity() > requests.size());
    RBC::Irecv(recv_begin,
               size,
               mpi_datatype,
               source,
               AmsTag::kGeneral,
               comm,
               &requests.emplace_back());

    recv_ranges.emplace_back(recv_begin, recv_begin + size);

    recv_begin += size;
  }

  // Post receive requests.
  for (auto in_msg = in_msgs.begin(); in_msg != first_recv_msg; ++in_msg) {
    const auto size = in_msg->second.second;
    const auto source = in_msg->second.first;

    assert(requests.capacity() > requests.size());
    RBC::Irecv(recv_begin,
               size,
               mpi_datatype,
               source,
               AmsTag::kGeneral,
               comm,
               &requests.emplace_back());

    recv_ranges.emplace_back(recv_begin, recv_begin + size);

    recv_begin += size;
  }

  // Calculate the first message which we will send according to the
  // 1-factor data exchange scheme.
  const auto first_send_msg = std::lower_bound(out_msgs.begin(), out_msgs.end(), first_target,
                                               [](const auto& msg, const auto target) {
      return msg.pe < target;
    });

  // Post send requests.
  for (auto out_msg = first_send_msg; out_msg != out_msgs.end(); ++out_msg) {
    const auto size = out_msg->size;
    const auto offset = out_msg->offset;
    const auto target = out_msg->pe;

    assert(requests.capacity() > requests.size());
    RBC::Isend(v_send.data() + offset,
               size,
               mpi_datatype,
               target,
               AmsTag::kGeneral,
               comm,
               &requests.emplace_back());
  }

  // Post send requests.
  for (auto out_msg = out_msgs.begin(); out_msg != first_send_msg; ++out_msg) {
    const auto size = out_msg->size;
    const auto offset = out_msg->offset;
    const auto target = out_msg->pe;

    assert(requests.capacity() > requests.size());
    RBC::Isend(v_send.data() + offset,
               size,
               mpi_datatype,
               target,
               AmsTag::kGeneral,
               comm,
               &requests.emplace_back());
  }

  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

  tracker.send_messages_c_.add(out_msgs.size());
  tracker.send_volume_c_.add(v_send.size());

  tracker.receive_messages_c_.add(in_msgs.size());
  tracker.receive_volume_c_.add(v_recv.size());

  return recv_ranges;
}

template <class AmsTag, class Tracker, class T>
void exchangeWithRecvSizesAndPorts(Tracker&& tracker, const std::vector<T>& v_send,
                                   const DistrRanges& out_msgs, std::vector<T>& v_recv,
                                   MPI_Datatype mpi_datatype, const RBC::Comm& comm) {
  // No messages of size zero.
  assert(std::find_if(out_msgs.begin(), out_msgs.end(),
                      [](const auto& msg) {
      return msg.size == 0;
    }) == out_msgs.end());

  // Messages are sorted by target process.
  assert(std::is_sorted(out_msgs.begin(), out_msgs.end(),
                        [](const auto& left, const auto& right) {
      return left.pe < right.pe;
    }));

  /*
   * Exchange receive sizes.
   */

  using InnerType = Tools::Tuple<int, int>;
  MPI_Datatype inner_type = InnerType::MpiType(
    Common::getMpiType<InnerType::first_type>(),
    Common::getMpiType<InnerType::second_type>());

  using InMsgType = Tools::Tuple<int, InnerType>;
  MPI_Datatype in_msg_mpi_type = InMsgType::MpiType(
    Common::getMpiType<InMsgType::first_type>(),
    inner_type);

  // Construct vector to exchange sizes.
  std::vector<InMsgType> in_msgs;
  in_msgs.reserve(out_msgs.size());
  for (const auto& out_msg : out_msgs) {
    in_msgs.emplace_back(out_msg.pe, Tools::Tuple<int, int>{ comm.getRank(),
                                                             static_cast<int>(out_msg.size) });
  }

  // Exchange sizes and then sort sizes by the source process.
  storeAndForward(in_msgs, in_msg_mpi_type, AmsTag::kGeneral, comm);
  std::sort(in_msgs.begin(), in_msgs.end(), [](const auto& l, const auto& r) {
      return l.second.first < r.second.first;
    });

  // Calculate receive volume.
  const int num_recv_els = std::accumulate(in_msgs.begin(), in_msgs.end(), int{ 0 },
                                           [](const int sum, const auto& in_msg) {
      return sum + in_msg.second.second;
    });

  // Adjust receive vector.
  v_recv.resize(num_recv_els);

  // Reorder recv messages according to the 1-factor algorithm by the source process.
  in_msgs = _internal::OneFactorReorder(in_msgs, comm.getRank(), comm.getSize(),
                                        [](const auto& msg) {
      return msg.second.first;
    });

  // Reorder send messages according to the 1-factor algorithm by the source process.
  const auto ordered_out_msgs = _internal::OneFactorReorder(out_msgs,
                                                            comm.getRank(),
                                                            comm.getSize(),
                                                            [](const auto& msg) {
      return msg.pe;
    });

  /*
   * Message exchange
   */

  const size_t port_cnt = 16;
  std::vector<MPI_Request> req(2 * port_cnt, MPI_REQUEST_NULL);

  auto in_it = in_msgs.begin();
  auto out_it = ordered_out_msgs.begin();

  const auto in_it_end = in_msgs.end();
  const auto out_it_end = ordered_out_msgs.end();

  const auto inr_begin = req.data();
  const auto inr_end = req.data() + port_cnt;
  const auto outr_begin = req.data() + port_cnt;
  const auto outr_end = req.data() + 2 * port_cnt;

  auto inr = inr_begin;
  auto outr = outr_begin;

  auto in_arr_ptr = v_recv.data();

  // Post as many incoming messages as possible.
  for (size_t i = 0; i != std::min(port_cnt, in_msgs.size()); ++i) {
    const auto size = in_it->second.second;
    const auto source = in_it->second.first;

    assert(in_arr_ptr + size <= v_recv.data() + v_recv.size());
    assert(inr < inr_end);

    RBC::Irecv(in_arr_ptr, size, mpi_datatype, source, AmsTag::kGeneral, comm, inr);

    in_arr_ptr += size;
    ++inr;
    ++in_it;
  }
  if (inr == inr_end) inr = inr_begin;

  // Post as many outgoing messages as possible.
  for (size_t i = 0; i != std::min(port_cnt, ordered_out_msgs.size()); ++i) {
    const auto size = out_it->size;
    assert(out_it->offset <= v_send.size());
    const auto ptr = v_send.data() + out_it->offset;
    assert(size + out_it->offset <= v_send.size());
    const auto target = out_it->pe;

    assert(outr < outr_end);
    assert(out_it < out_it_end);

    RBC::Isend(ptr, size, mpi_datatype, target, AmsTag::kGeneral, comm, outr);

    ++outr;
    ++out_it;
  }
  if (outr == outr_end) outr = outr_begin;

  // Post all send and receive requests while testing existing requests.
  while (in_it != in_it_end || out_it != out_it_end) {
    /*
     * Process incoming messages.
     */

    // Test message.
    if (*inr != MPI_REQUEST_NULL) {
      int flag = 0;
      MPI_Test(inr, &flag, MPI_STATUS_IGNORE);

      if (flag) *inr = MPI_REQUEST_NULL;
    }

    // Start receiving a message.
    if (in_it != in_it_end && *inr == MPI_REQUEST_NULL) {
      const auto size = in_it->second.second;
      const auto source = in_it->second.first;

      assert(in_arr_ptr + size <= v_recv.data() + v_recv.size());
      assert(inr < inr_end);

      RBC::Irecv(in_arr_ptr, size, mpi_datatype, source, AmsTag::kGeneral, comm, inr);

      in_arr_ptr += size;
      ++in_it;
    }

    ++inr;
    if (inr == inr_end) inr = inr_begin;

    /*
     * Process outgoing messages.
     */

    // Test message.
    if (*outr != MPI_REQUEST_NULL) {
      int flag = 0;
      MPI_Test(outr, &flag, MPI_STATUS_IGNORE);

      if (flag) *outr = MPI_REQUEST_NULL;
    }

    // Start sending a message.
    if (out_it != out_it_end && *outr == MPI_REQUEST_NULL) {
      const auto size = out_it->size;
      assert(size + out_it->offset <= v_send.size());
      const auto ptr = v_send.data() + out_it->offset;
      assert(out_it->offset <= v_send.size());
      const auto target = out_it->pe;

      assert(outr < outr_end);
      assert(out_it < out_it_end);

      RBC::Isend(ptr, size, mpi_datatype, target, AmsTag::kGeneral, comm, outr);

      ++out_it;
    }

    ++outr;
    if (outr == outr_end) outr = outr_begin;
  }

  assert(in_arr_ptr == v_recv.data() + v_recv.size());

  // Wait until remaining requests are finished.
  MPI_Waitall(req.size(), req.data(), MPI_STATUSES_IGNORE);

  tracker.send_messages_c_.add(out_msgs.size());
  tracker.send_volume_c_.add(v_send.size());

  tracker.receive_messages_c_.add(in_msgs.size());
  tracker.receive_volume_c_.add(v_recv.size());
}

template <class AmsTag, class Tracker, class T>
std::vector<std::pair<T*, T*> > exchangeWithRecvSizesAndPortsReturnRanges(Tracker&& tracker,
                                                                          const std::vector<T>& v_send,
                                                                          const DistrRanges& out_msgs,
                                                                          std::vector<T>& v_recv,
                                                                          MPI_Datatype mpi_datatype,
                                                                          const RBC::Comm& comm) {
  // No messages of size zero.
  assert(std::find_if(out_msgs.begin(), out_msgs.end(),
                      [](const auto& msg) {
      return msg.size == 0;
    }) == out_msgs.end());

  // Messages are sorted by target process.
  assert(std::is_sorted(out_msgs.begin(), out_msgs.end(),
                        [](const auto& left, const auto& right) {
      return left.pe < right.pe;
    }));

  /*
   * Exchange receive sizes.
   */

  using InnerType = Tools::Tuple<int, int>;
  MPI_Datatype inner_type = InnerType::MpiType(
    Common::getMpiType<InnerType::first_type>(),
    Common::getMpiType<InnerType::second_type>());

  using InMsgType = Tools::Tuple<int, InnerType>;
  MPI_Datatype in_msg_mpi_type = InMsgType::MpiType(
    Common::getMpiType<InMsgType::first_type>(),
    inner_type);

  // Construct vector to exchange sizes.
  std::vector<InMsgType> in_msgs;
  in_msgs.reserve(out_msgs.size());
  for (const auto& out_msg : out_msgs) {
    in_msgs.emplace_back(out_msg.pe, Tools::Tuple<int, int>{ comm.getRank(),
                                                             static_cast<int>(out_msg.size) });
  }

  // Exchange sizes and sort then sizes by the source process.
  storeAndForward(in_msgs, in_msg_mpi_type, AmsTag::kGeneral, comm);
  std::sort(in_msgs.begin(), in_msgs.end(), [](const auto& l, const auto& r) {
      return l.second.first < r.second.first;
    });

  std::vector<std::pair<T*, T*> > recv_ranges;
  recv_ranges.reserve(in_msgs.size());


  // Calculate receive volume.
  const int num_recv_els = std::accumulate(in_msgs.begin(), in_msgs.end(), int{ 0 },
                                           [](const int sum, const auto& in_msg) {
      return sum + in_msg.second.second;
    });

  // Adjust receive vector.
  v_recv.resize(num_recv_els);

  // Reorder recv messages according to the 1-factor algorithm by the source process.
  in_msgs = _internal::OneFactorReorder(in_msgs, comm.getRank(), comm.getSize(),
                                        [](const auto& msg) {
      return msg.second.first;
    });

  assert(num_recv_els == std::accumulate(in_msgs.begin(), in_msgs.end(), int{ 0 },
                                         [](const int sum, const auto& in_msg) {
      return sum + in_msg.second.second;
    }));

  // Reorder send messages according to the 1-factor algorithm by the source process.
  const auto ordered_out_msgs = _internal::OneFactorReorder(out_msgs,
                                                            comm.getRank(),
                                                            comm.getSize(),
                                                            [](const auto& msg) {
      return msg.pe;
    });

  /*
   * Message exchange
   */

  const size_t port_cnt = 16;
  std::vector<MPI_Request> req(2 * port_cnt, MPI_REQUEST_NULL);

  auto in_it = in_msgs.begin();
  auto out_it = ordered_out_msgs.begin();

  const auto in_it_end = in_msgs.end();
  const auto out_it_end = ordered_out_msgs.end();

  const auto inr_begin = req.data();
  const auto inr_end = req.data() + port_cnt;
  const auto outr_begin = req.data() + port_cnt;
  const auto outr_end = req.data() + 2 * port_cnt;

  auto inr = inr_begin;
  auto outr = outr_begin;

  auto in_arr_ptr = v_recv.data();

  // Post as many incoming messages as possible.
  for (size_t i = 0; i != std::min(port_cnt, in_msgs.size()); ++i) {
    const auto size = in_it->second.second;
    const auto source = in_it->second.first;

    assert(in_arr_ptr + size <= v_recv.data() + v_recv.size());
    assert(inr < inr_end);

    RBC::Irecv(in_arr_ptr, size, mpi_datatype, source, AmsTag::kGeneral, comm, inr);

    recv_ranges.emplace_back(in_arr_ptr, in_arr_ptr + size);

    in_arr_ptr += size;
    ++inr;
    ++in_it;
  }
  if (inr == inr_end) inr = inr_begin;

  // Post as many outgoing messages as possible.
  for (size_t i = 0; i != std::min(port_cnt, ordered_out_msgs.size()); ++i) {
    const auto size = out_it->size;
    assert(out_it->offset <= v_send.size());
    const auto ptr = v_send.data() + out_it->offset;
    assert(size + out_it->offset <= v_send.size());
    const auto target = out_it->pe;

    assert(outr < outr_end);
    assert(out_it < out_it_end);

    RBC::Isend(ptr, size, mpi_datatype, target, AmsTag::kGeneral, comm, outr);

    ++outr;
    ++out_it;
  }
  if (outr == outr_end) outr = outr_begin;

  // Post all send and receive requests while testing existing requests.
  while (in_it != in_it_end || out_it != out_it_end) {
    /*
     * Process incoming messages.
     */

    // Test message.
    if (*inr != MPI_REQUEST_NULL) {
      int flag = 0;
      MPI_Test(inr, &flag, MPI_STATUS_IGNORE);

      if (flag) *inr = MPI_REQUEST_NULL;
    }

    // Start receiving a message.
    if (in_it != in_it_end && *inr == MPI_REQUEST_NULL) {
      const auto size = in_it->second.second;
      const auto source = in_it->second.first;

      assert(in_arr_ptr + size <= v_recv.data() + v_recv.size());
      assert(inr < inr_end);

      RBC::Irecv(in_arr_ptr, size, mpi_datatype, source, AmsTag::kGeneral, comm, inr);

      recv_ranges.emplace_back(in_arr_ptr, in_arr_ptr + size);

      in_arr_ptr += size;
      ++in_it;
    }

    ++inr;
    if (inr == inr_end) inr = inr_begin;

    /*
     * Process outgoing messages.
     */

    // Test message.
    if (*outr != MPI_REQUEST_NULL) {
      int flag = 0;
      MPI_Test(outr, &flag, MPI_STATUS_IGNORE);

      if (flag) *outr = MPI_REQUEST_NULL;
    }

    // Start sending a message.
    if (out_it != out_it_end && *outr == MPI_REQUEST_NULL) {
      const auto size = out_it->size;
      assert(size + out_it->offset <= v_send.size());
      const auto ptr = v_send.data() + out_it->offset;
      assert(out_it->offset <= v_send.size());
      const auto target = out_it->pe;

      assert(outr < outr_end);
      assert(out_it < out_it_end);
      RBC::Isend(ptr, size, mpi_datatype, target, AmsTag::kGeneral, comm, outr);

      ++out_it;
    }

    ++outr;
    if (outr == outr_end) outr = outr_begin;
  }

  assert(in_arr_ptr == v_recv.data() + v_recv.size());

  // Wait until remaining requests are finished.
  MPI_Waitall(req.size(), req.data(), MPI_STATUSES_IGNORE);

  tracker.send_messages_c_.add(out_msgs.size());
  tracker.send_volume_c_.add(v_send.size());

  tracker.receive_messages_c_.add(in_msgs.size());
  tracker.receive_volume_c_.add(v_recv.size());

  return recv_ranges;
}
}  // end namespace Alltoallv
