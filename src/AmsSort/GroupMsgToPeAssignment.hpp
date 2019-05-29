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
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "../../include/AmsSort/Configuration.hpp"
#include "../Tools/CommonMpi.hpp"
#include "Alltoall.hpp"
#include "BinomialTreePipeline.hpp"
#include "Bitonic/Bitonic.hpp"

#include <RBC.hpp>
#include <tlx/algorithm.hpp>

namespace Ams {
namespace _internal {
namespace GroupMsgToPeAssignment {
template <class AmsTag, class Tracker>
// DistrRanges are sorted by target process. There is at most one range for each target process.
std::tuple<DistrRanges, size_t, size_t> detAssignment(const std::vector<size_t>& loc_group_el_cnts,
                                                      const std::vector<size_t>& glob_group_el_cnts,
                                                      const std::vector<size_t>& group_sizes,
                                                      const std::vector<size_t>& group_sizes_exscan,
                                                      size_t my_group_idx,
                                                      size_t my_group_rank,
                                                      size_t residual_capacity,
                                                      const Ams::DistributionStrategy
                                                      distr_strategy,
                                                      Tracker& tracker,
                                                      RBC::Comm comm, RBC::Comm group_comm);

// DistrRanges are sorted by target process. There is at most one range for each target process.
std::tuple<DistrRanges, size_t, size_t> simpleAssignment(const std::vector<size_t>& group_sizes,
                                                         const std::vector<size_t>&
                                                         group_sizes_exscan,
                                                         const std::vector<size_t>&
                                                         loc_group_el_cnts,
                                                         const std::vector<size_t>&
                                                         glob_group_el_cnts,
                                                         size_t my_group_idx,
                                                         bool use_two_tree,
                                                         RBC::Comm comm);

// DistrRanges are sorted by target process. There is at most one range for each target process.
std::tuple<DistrRanges, size_t, size_t> simpleAssignment(const std::vector<size_t>& group_sizes,
                                                         const std::vector<size_t>&
                                                         group_sizes_exscan,
                                                         const std::vector<size_t>&
                                                         loc_group_el_cnts,
                                                         const std::vector<size_t>&
                                                         loc_group_el_distr_scan,
                                                         const std::vector<size_t>&
                                                         glob_group_el_cnts,
                                                         size_t my_group_idx,
                                                         RBC::Comm comm);

namespace _internal {
struct DataRequest {
  DataRequest() :
    size(0),
    pe(0)
  { }

  DataRequest(size_t my_size, size_t my_pe) :
    size(my_size),
    pe(my_pe)
  { }

  size_t size;
  size_t pe;
};

struct GroupDataRequest {
 public:
  GroupDataRequest() :
    size(0),
    group(0),
    pe(0) { }
  GroupDataRequest(size_t my_size, size_t my_group_id, size_t my_pe) :
    size(my_size),
    group(my_group_id),
    pe(my_pe) { }

  static MPI_Datatype mpiType() {
    static MPI_Datatype my_type = MPI_DATATYPE_NULL;

    if (my_type == MPI_DATATYPE_NULL) {
      MPI_Datatype types[3];
      int blocklen[3] = { 1, 1, 1 };
      MPI_Aint disp[3] = {
        offsetof(GroupDataRequest, size),
        offsetof(GroupDataRequest, group),
        offsetof(GroupDataRequest, pe)
      };

      types[0] = Common::getMpiType<decltype(size)>();
      types[1] = Common::getMpiType<decltype(group)>();
      types[2] = Common::getMpiType<decltype(pe)>();

      MPI_Type_create_struct(3, blocklen, disp, types, &my_type);
      MPI_Type_commit(&my_type);
    }

    return my_type;
  }

  bool operator== (const GroupDataRequest& r) const {
    return size == r.size && group == r.group &&
           pe == r.pe;
  }

  size_t getSize() const {
    return size;
  }

  size_t getGroup() const {
    return group;
  }

  int getPe() const {
    return pe;
  }

 private:
  size_t size;
  size_t group;
  int pe;
};

struct Piece {
 public:
  Piece() :
    begin_(0),
    end_(0),
    pe_(0),
    type_(Piece::residualType()) { }

  int pe() const { return pe_; }

  size_t begin() const { return begin_; }

  size_t end() const { return end_; }

  size_t size() const { return end_ - begin_; }

  // void setPe(size_t pe) { pe_ = pe; }

  void setBegin(size_t begin) { begin_ = begin; }

  void setEnd(size_t end) { end_ = end; }

  static Piece requestPiece(size_t begin, size_t end, size_t pe) {
    return Piece(begin, end, pe, Piece::requestType());
  }

  static Piece residualPiece(size_t begin, size_t end, size_t pe) {
    return Piece(begin, end, pe, Piece::residualType());
  }

  bool isRequest() const { return type_ == Piece::requestType(); }
  bool isResidual() const { return type_ == Piece::residualType(); }
  bool isEmpty() const { return begin_ == end_; }
  bool isTypeEqualTo(const Piece& p) const { return type_ == p.type_; }

  GroupDataRequest satisfyRequest(Piece& x_piece, size_t group_id) {
    assert(isRequest());
    assert(x_piece.isResidual());
    assert(x_piece.begin() == this->begin());

    const size_t satisfiable_request_size =
      std::min(size(), x_piece.size());
    GroupDataRequest msg(satisfiable_request_size, group_id,
                         x_piece.pe_);

    removeLeftCapacity(satisfiable_request_size);
    x_piece.removeLeftCapacity(satisfiable_request_size);

    assert(isEmpty() || x_piece.isEmpty());
    assert(msg.getSize() > 0);
    return msg;
  }

  static MPI_Datatype mpiType() {
    static MPI_Datatype my_type = MPI_DATATYPE_NULL;
    if (my_type == MPI_DATATYPE_NULL) {
      MPI_Datatype types[4];
      int blocklen[4] = { 1, 1, 1, 1 };
      MPI_Aint disp[4] = {
        offsetof(Piece, begin_),
        offsetof(Piece, end_),
        offsetof(Piece, pe_),
        offsetof(Piece, type_)
      };

      types[0] = Common::getMpiType<decltype(begin_)>();
      types[1] = Common::getMpiType<decltype(end_)>();
      types[2] = Common::getMpiType<decltype(pe_)>();
      types[3] = Common::getMpiType<decltype(type_)>();

      MPI_Type_create_struct(4, blocklen, disp, types, &my_type);
      MPI_Type_commit(&my_type);
    }

    return my_type;
  }

 private:
  Piece(size_t begin, size_t end, size_t pe, size_t type) :
    begin_(begin),
    end_(end),
    pe_(pe),
    type_(type) { }

  static int requestType() { return 1; }
  static int residualType() { return 0; }

  void removeLeftCapacity(size_t cap) {
    assert(size() >= cap);
    begin_ += cap;
  }

  size_t begin_;
  size_t end_;
  int pe_;
  int type_;
};

/* @brief Merges sorted X pieces and Y pieces and generates request answers.
 *
 * The X pieces (Y pieces) cover a stripe of residual elements
 * (request elements).  The residual stripe may begin before the
 * request stripe starts (the first X piece extends into the first Y
 * piece) or may end after the request stripe ends (the last X piece
 * extends into the last Y piece).
 */
void mergeXY(std::vector<Piece>& x_pieces,
             std::vector<Piece>& y_pieces,
             size_t my_group_idx,
             std::vector<GroupDataRequest>& answers,
             DistrRanges& answer_targets);

/* @brief Calculates send descriptions for a message assignment.
 *
 * @param assignments Vector of assignments of "element volume -> target process".
 *                    The vector 'assignments' stores one assignment (not more!)
 *                    for a target process. The assignments for one process group
 *                    are stored next to each other and the assignments for
 *                    one process group are sorted by the target process id.
 *
 * @return Send descriptions for outgoing messages. The descriptions are ordered
 *         by the receiving processes.
 */
DistrRanges msgAssignmentToSendDescr(const std::vector<GroupDataRequest>& assignments,
                                     const std::vector<size_t>& group_sizes_exscan,
                                     std::vector<size_t> loc_group_el_cnts_exscan);

std::vector<Piece> orderXPieces(const std::vector<Piece>& x_pieces);

/* @brief Creates Y pieces of requests.
 *
 * @param lmsgs_request_messages Requests sorted by requesting processes
 * @return Y pieces sorted by requesting processes.
 *
 */
std::vector<Piece> createYPieces(size_t lmsgs_request_exscan,
                                 const std::vector<int>& recv_partners,
                                 const std::vector<size_t>& recv_sizes,
                                 const size_t max_small_msg_size);

inline bool isSmallMsg(size_t msg_size, size_t max_small_msg_size) {
  return msg_size <= max_small_msg_size && msg_size > 0;
}

/*  @brief Maximum size of a small message.
 */
size_t maxSmallMsgSize(size_t group_size,
                       size_t residual_capacity, size_t nprocs);

template <class AmsTag, class Tracker>
void distributeMergedXPieces(const std::vector<int>& target_pes, const std::vector<Piece>& send_xs,
                             const size_t rbegin, const size_t rend, std::vector<Piece>& recv_xs,
                             Tracker& tracker, RBC::Comm group_comm) {
  std::vector<RBC::Request> send_requests(send_xs.size());
  // send x pieces
  for (size_t i = 0; i < send_xs.size(); ++i) {
    auto target_pe = target_pes[i];
    RBC::Isend(send_xs.data() + i, 1, Piece::mpiType(), target_pe,
               AmsTag::kCapacityDistributeSchema, group_comm,
               send_requests.data() + i);
  }

  // Recv pieces
  const auto getContributingResidual =
    [](const Piece& candidate, size_t rbegin, size_t rend) {
      const auto cbegin = candidate.begin();
      const auto cend = candidate.end();

      const auto end = std::min(rend, cend);
      const auto begin = std::max(rbegin, cbegin);

      assert(end > begin);

      return end - begin;
    };

#ifndef NDEBUG
  const auto isContributing = [&getContributingResidual](
    Piece candidate, size_t rbegin, size_t rend) {
                                assert(candidate.isResidual());
                                return getContributingResidual(candidate, rbegin, rend) > 0;
                              };
#endif

  size_t request_count = rend - rbegin;
  int sent = 0;
  while (request_count) {
    MPI_Status status;
    int flag = 0;
    RBC::Iprobe(MPI_ANY_SOURCE, AmsTag::kCapacityDistributeSchema, group_comm, &flag, &status);
    if (flag) {
      recv_xs.push_back(Piece{ });
      int source = group_comm.MpiRankToRangeRank(status.MPI_SOURCE);
      RBC::Recv(&recv_xs.back(), 1, Piece::mpiType(), source,
                AmsTag::kCapacityDistributeSchema, group_comm, MPI_STATUS_IGNORE);
      assert(isContributing(recv_xs.back(), rbegin, rend));
      size_t contributed_residual =
        getContributingResidual(recv_xs.back(), rbegin, rend);
      assert(contributed_residual > 0);
      assert(contributed_residual <= request_count);
      request_count -= contributed_residual;
    }

    if (!sent) {
      RBC::Testall(send_requests.size(), send_requests.data(), &sent,
                   MPI_STATUSES_IGNORE);
    }
  }

  if (!sent) {
    RBC::Waitall(send_requests.size(), send_requests.data(),
                 MPI_STATUSES_IGNORE);
  }

  tracker.send_messages_c_.add(send_xs.size());
  tracker.send_volume_c_.add(send_xs.size());

  tracker.receive_messages_c_.add(recv_xs.size());
  tracker.receive_volume_c_.add(recv_xs.size());
}

std::vector<int> recvPartners(int nprocs, int my_group_size, int my_group_rank);
}  // end namespace _internal

/*
 * 1. Store small messages locally.
 * 2. Route message requests to group with one-factor algorithm.
 * 3. Calculate residual capacity.
 * 4. Distributed scan over request sizes and residual capacities.
 * 5. Store first value of "local request size prefix sum vector" y and "local residual capacity prefix sum vector" x.
 * 6. Merge (sorted) X and Y sequence with bitonic merge. Each process again has two pieces a and b.
 * 7. Send piece b to next process. Thus, each process knows piece 'prev_piece' which comes before piece a.
 * 8. Detect each x piece 'x_remember' which is directly followed by a y piece.
 * 9. Partial broadcast of 'x_remember' stored at process i to processes 'i+1'..'i+j', where 'i+j' is the first process which stores a y piece. The received piece is called 'x_bcasted'.
 * 10.
 *
 * @return Returns send descriptions and maximum number of receive messages.
 */
template <class AmsTag, class Tracker>
std::tuple<DistrRanges, size_t, size_t> detAssignment(const std::vector<size_t>& loc_group_el_cnts,
                                                      const std::vector<size_t>& glob_group_el_cnts,
                                                      const std::vector<size_t>& group_sizes,
                                                      const std::vector<size_t>& group_sizes_exscan,
                                                      const size_t my_group_idx,
                                                      const size_t my_group_rank,
                                                      const size_t residual_capacity,
                                                      const Ams::DistributionStrategy
                                                      distr_strategy,
                                                      Tracker& tracker,
                                                      RBC::Comm comm, RBC::Comm group_comm) {
  std::ignore = glob_group_el_cnts;

    #ifndef NDEBUG
  for (size_t i = 0; i != glob_group_el_cnts.size(); ++i) {
    assert(residual_capacity * group_sizes[i] >= glob_group_el_cnts[i]);
  }
    #endif

  using Piece = _internal::Piece;

  const size_t nprocs = comm.getSize();
  const size_t myrank = comm.getRank();
  const size_t group_cnt = group_sizes.size();
  const size_t my_group_size = group_sizes[my_group_idx];

  const auto loc_group_el_cnts_exscan = [&loc_group_el_cnts]() {
                                          std::vector<size_t> loc_group_el_cnts_exscan(
                                            loc_group_el_cnts.size() + 1);
                                          tlx::exclusive_scan(loc_group_el_cnts.begin(),
                                                              loc_group_el_cnts.end(),
                                                              loc_group_el_cnts_exscan.begin(), 0);
                                          return loc_group_el_cnts_exscan;
                                        } ();

  size_t max_num_recv_msgs = 0;

  /******************************************************************************/
  /* Route send requests to group.                                              */
  /*                                                                            */
  /* We send one message to each group.                                         */
  /* We receive messages from processes [ begin, end )                          */
  /* begin := my_group_rank * tlx::div_ceil(nprocs, group_sizes[my_group_idx])  */
  /* end   := min((my_group_rank + 1) * tlx::div_ceil(                          */
  /*                  nprocs, group_sizes[my_group_idx]), nprocs )              */
  /******************************************************************************/

  const std::vector<int> recv_partners = _internal::recvPartners(nprocs, my_group_size,
                                                                 my_group_rank);

  const int first_1factor_group = Alltoallv::_internal::FirstTarget(group_cnt, my_group_idx);

  std::vector<MPI_Request> requests(group_cnt + recv_partners.size());

  // Receive requests in order. Post receive requests in one-factor order.
  std::vector<size_t> recv_sizes(recv_partners.size());
  const int recv_offset = std::lower_bound(
    recv_partners.begin(), recv_partners.end(),
    group_sizes_exscan[first_1factor_group]) - recv_partners.begin();

  for (size_t i = 0; i != recv_partners.size(); ++i) {
    auto rotated_idx = i + recv_offset;
    if (rotated_idx >= recv_partners.size()) {
      rotated_idx -= recv_partners.size();
    }

    const auto source = recv_partners[rotated_idx];

    RBC::Irecv(recv_sizes.data() + rotated_idx, 1,
               Common::getMpiType(recv_sizes),
               source, AmsTag::kOneFactorGroupBasedExchange,
               comm, requests.data() + i);
  }

  // Send requests in one-factor order.
  for (size_t i = 0; i != group_cnt; ++i) {
    auto rotated_idx = i + first_1factor_group;
    if (rotated_idx >= group_cnt) {
      rotated_idx -= group_cnt;
    }

    const auto target = group_sizes_exscan[rotated_idx]
                        + myrank % group_sizes[rotated_idx];

    RBC::Isend(loc_group_el_cnts.data() + rotated_idx, 1,
               Common::getMpiType(loc_group_el_cnts),
               target, AmsTag::kOneFactorGroupBasedExchange,
               comm, requests.data() + recv_sizes.size() + i);
  }
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

  tracker.send_messages_c_.add(group_cnt);
  tracker.send_volume_c_.add(group_cnt);

  tracker.receive_messages_c_.add(recv_partners.size());
  tracker.receive_volume_c_.add(recv_partners.size());

  // Request answers (of type GroupDataRequest) contain the group
  // index such that the receiver can sort the answers by their target
  // groups using a bucket sort algorithm.
  std::vector<_internal::GroupDataRequest> answers;
  DistrRanges answer_targets;

  if (my_group_size == 1) {
    // Just one process in the group. All processes send their
    // data to this process.
    for (size_t i = 0; i != recv_sizes.size(); ++i) {
      // Empty answers are not allowed as the receiver would not
      // wait for such answers.
      // 'answers' must be ordered by process ids.
      if (recv_sizes[i] > 0) {
        assert(i == 0 || recv_partners[i - 1] < recv_partners[i]);
        answer_targets.emplace_back(recv_partners[i], answer_targets.size(), 1);
        answers.emplace_back(recv_sizes[i], my_group_idx, my_group_rank);
      }
    }

    max_num_recv_msgs = answers.size();

    tracker.send_messages_c_.add(0);
    tracker.send_volume_c_.add(0);

    tracker.receive_messages_c_.add(0);
    tracker.receive_volume_c_.add(0);
  } else if (residual_capacity == 0) {
    assert(recv_partners.empty());

    tracker.send_messages_c_.add(0);
    tracker.send_volume_c_.add(0);

    tracker.receive_messages_c_.add(0);
    tracker.receive_volume_c_.add(0);
  } else {
    // We have to assign messages.

    /******************************************************************************/
    /* Split requests into large and small requests.                              */
    /******************************************************************************/

    size_t smsg_capacity = 0;
    size_t lmsgs_request_volume = 0;
    size_t smsg_cnt = 0;

    const size_t my_max_small_msg_size = _internal::maxSmallMsgSize(
      group_sizes[my_group_idx], residual_capacity, nprocs);
    for (size_t i = 0; i != recv_sizes.size(); ++i) {
      if (_internal::isSmallMsg(recv_sizes[i], my_max_small_msg_size)) {
        answer_targets.emplace_back(recv_partners[i], answers.size(), 1);
        answers.emplace_back(recv_sizes[i], my_group_idx, my_group_rank);

        ++smsg_cnt;
        smsg_capacity += recv_sizes[i];
      } else if (recv_sizes[i] > 0) {
        lmsgs_request_volume += recv_sizes[i];
      }
    }

    // Add number of incoming messages during the data exchange.
    max_num_recv_msgs = smsg_cnt + 2 + tlx::div_ceil(residual_capacity,
                                                     my_max_small_msg_size + 1);

    assert(residual_capacity >= 2 * smsg_capacity);

    const size_t lmsg_capacity = residual_capacity - smsg_capacity;
    assert(lmsg_capacity > 0);

    /******************************************************************************
     * Calculate exclusive prefix sum over remaining residuals and
     * large message request sizes of this group.
     ******************************************************************************/

    size_t lmsg_request_exscan = 0;
    size_t lmsg_capacity_exscan = 0;
    size_t lmsg_total_request = 0;
    auto scan = [group_comm, my_group_rank](size_t lmsgs_request_count,
                                            size_t remaining_residual) {
                  size_t tmp[2] = { lmsgs_request_count, remaining_residual };
                  size_t tmp_exscan[2];
                  size_t tmp_sum[2];

                  ScanAndBcast(tmp, tmp_exscan, tmp_sum, 2,
                               Common::getMpiType(tmp[0]), MPI_SUM, group_comm);

                  tmp_exscan[0] -= lmsgs_request_count;
                  tmp_exscan[1] -= remaining_residual;

                  return std::tuple<size_t, size_t, size_t>(
                    tmp_exscan[0], tmp_exscan[1], tmp_sum[0]);
                };
    std::tie(lmsg_request_exscan, lmsg_capacity_exscan, lmsg_total_request) =
      scan(lmsgs_request_volume, lmsg_capacity);

#ifndef NDEBUG
    if (my_group_rank + 1 == my_group_size) {
      const size_t total_request = lmsg_request_exscan + lmsgs_request_volume;
      const size_t total_capacity = lmsg_capacity_exscan + lmsg_capacity;

      assert(total_request <= total_capacity);
    }
#endif

    /******************************************************************************
     * Create X and Y piece locally. X pieces as well as Y pieces form two sorted
     * global sequences.
     * Merge those two sorted sequences with bitonic merging.
     ******************************************************************************/

    std::vector<Piece> sorted_xys = {
      // X piece.
      Piece::residualPiece(lmsg_capacity_exscan,
                           lmsg_capacity_exscan + lmsg_capacity,
                           my_group_rank),
      // Y piece.
      Piece::requestPiece(lmsg_request_exscan,
                          lmsg_request_exscan + lmsgs_request_volume,
                          my_group_rank)
    };

    // Pieces of the same type are ordered by their process id. A
    // residual piece is smaller than a request piece, iff the
    // residual piece begins before the request piece ends!! This
    // is crucial!!!
    auto comp = [](const Piece& p1, const Piece& p2) {
                  if (p1.isTypeEqualTo(p2)) {
                    return p1.pe() < p2.pe();
                  } else if (p1.isResidual()) {  // and p2.IsRequest() == true
                    return p1.begin() < p2.end();
                  } else {  // p1.IsRequest() and p2.IsResidual()
                    return p1.end() <= p2.begin();
                  }
                };

    assert(sorted_xys.size() == 2);
    // Distribute the pieces such that the first half of the PEs hold
    // a decreasing sequence of the Xs and the second half of the PEs
    // hold a increasing sequence of the Ys.
    // This leads to a bitonic decreasing sequence of size 2 * my_group_size
    // with each PE holding 2 elements.
    MPI_Request requests[4] = { MPI_REQUEST_NULL };
    // Send first element to first half
    RBC::Isend(&sorted_xys[0], 1, Piece::mpiType(),
               (my_group_size - 1 - my_group_rank) / 2,
               AmsTag::kGeneral, group_comm, &requests[0]);
    // Send second element to second half
    RBC::Isend(&sorted_xys[1], 1, Piece::mpiType(),
               (my_group_size + my_group_rank) / 2,
               AmsTag::kGeneral, group_comm, &requests[1]);
    // Receive elements
    std::vector<Piece> bitonic_sequence(2);
    for (size_t i = 0; i < 2; i++) {
      size_t elm_id = 2 * my_group_rank + i;
      size_t recv_rank;
      if (elm_id < my_group_size)
        recv_rank = my_group_size - 1 - elm_id;
      else
        recv_rank = elm_id - my_group_size;
      RBC::Irecv(&bitonic_sequence[i], 1, Piece::mpiType(), recv_rank,
                 AmsTag::kGeneral, group_comm, &requests[2 + i]);
    }
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    sorted_xys = std::move(bitonic_sequence);

    // Sort the sequence
    std::sort(sorted_xys.begin(), sorted_xys.end(), comp);
    Bitonic::SortBitonicSequence(sorted_xys, Piece::mpiType(), AmsTag::kGeneral,
                                 group_comm, comp, true, true);
    assert(sorted_xys.size() == 2);

    /******************************************************************************
     * Send second piece of local merged subsequence to next process.
     ******************************************************************************/

    Piece prev_piece;
    RBC::Sendrecv(sorted_xys.data() + 1, 1, Piece::mpiType(),
                  (my_group_rank + 1) % my_group_size,
                  AmsTag::kGeneral, &prev_piece, 1,
                  Piece::mpiType(),
                  (my_group_rank + my_group_size - 1) % my_group_size,
                  AmsTag::kGeneral, group_comm, MPI_STATUS_IGNORE);
    // Reset previous piece on PE 0.
    if (my_group_rank == 0) {
      prev_piece = Piece::residualPiece(0, 0, 0);
    }

    /******************************************************************************
     *
     * Broadcast X pieces which are followed by a Y piece to subsequent processes.
     * Reason: A X piece covers multiple Y pieces.
     * Correctness: A X piece x1 can cover many subsequent Y pieces, until a X piece
     *              x2 occurs. x1 may also cover the Y piece which comes after x2.
     *
     * Example:
     * Input:  X1 Y    | X2 X3| Y  X4| Y  Y | Y  Y | Y  Y | X5 X6| X7 X8| Y X9 ...
     * Bcast:  X1      |      | X3   | X4   |      |      |      |      | X8
     * Result: (0,0,0) | X1   | X1   | X3   | X4   | X4   | X4   | X4   | X4  | X8

     * Process i broadcasts piece X in two cases:
     *            i-1    i
     *  - Case 1: X   | Y ?
     *  - Case 2: ?   | X Y
     ******************************************************************************/

    // If a y piece is merged after a x piece, store x piece and remember.
    bool has_xy_sequence = false;
    Piece x_remember;
    if (sorted_xys.front().isResidual() && sorted_xys.back().isRequest()) {
      x_remember = sorted_xys.front();
      has_xy_sequence = true;
    } else if (prev_piece.isResidual() && sorted_xys.front().isRequest()) {
      x_remember = prev_piece;
      has_xy_sequence = true;
    }
    // First piece must be remembered. Reset previous piece on PE 0.
    if (my_group_rank == 0 && !has_xy_sequence) {
      x_remember = Piece::residualPiece(0, 0, 0);
      has_xy_sequence = true;
    }

    // Partial broadcast of remembered x pieces.
    Piece x_bcasted = BinomialTreePipeline::bcast<Piece>(
      x_remember, has_xy_sequence, Piece::mpiType(), group_comm);
    // Reset x bcasted on PE 0.
    if (my_group_rank == 0) {
      x_bcasted = Piece::residualPiece(0, 0, 0);
    }

    /******************************************************************************
     * Assign x pieces to y pieces.
     *
     * 1. Calculate number of Y pieces on previous processes.
     * 2. Iterate over X pieces and assign them to the previous Y piece.
     ******************************************************************************/

    std::vector<Piece> send_xs;
    std::vector<int> targets;

    // Count Y pieces in sorted_xys ex sum
    int num_prev_ys = [group_comm, my_group_rank, &sorted_xys]() {
                        int num_ypieces = 0;
                        for (auto& piece : sorted_xys) {
                          if (piece.isRequest()) num_ypieces++;
                        }

                        int scan = 0;
                        RBC::Exscan(&num_ypieces, &scan, 1,
                                    Common::getMpiType(num_ypieces), MPI_SUM, group_comm);
                        if (my_group_rank == 0) {
                          return 0;
                        } else {
                          return scan;
                        }
                      } ();

    auto const isContributing = [](const Piece& candidate, const Piece& piece) {
                                  // Candidate is not empty.
                                  // Piece is not empty.
                                  // Candidate covers piece at some position.
                                  assert(candidate.isResidual() && piece.isRequest());
                                  return !piece.isEmpty()
                                         &&
                                         (candidate.end() > piece.begin()
                                          &&
                                          piece.end() > candidate.begin());
                                };
    auto const isCovering = [lmsg_total_request](const Piece& xpiece) {
                              // Check whether xpiece is covering any request. As the
                              // total residual capacity may be larger than the total
                              // request capacity. There may be some xpieces (the last
                              // ones) which do not cover any requests. We have to
                              // filter out those residuals. Otherwise, those xpieces
                              // would all be send to the last process of the group!
                              assert(xpiece.isResidual());

                              return !xpiece.isEmpty() && xpiece.begin() < lmsg_total_request;
                            };

    if (sorted_xys.front().isResidual() && sorted_xys.back().isRequest()) {
      const auto& XB = x_bcasted;
      const auto& X1 = sorted_xys.front();
      const auto& Y2 = sorted_xys.back();

      // Example: XB Y X X |X1 Y2|

      if (isContributing(XB, Y2)) {
        send_xs.push_back(XB);
        targets.push_back(num_prev_ys);
      }

      if (isCovering(X1)) {
        send_xs.push_back(X1);
        targets.push_back(num_prev_ys);
      }
    } else if (sorted_xys.front().isRequest() &&
               sorted_xys.back().isResidual()) {
      const auto& XB = x_bcasted;
      const auto& Y1 = sorted_xys.front();
      const auto& X2 = sorted_xys.back();

      // Example: XB Y X X |Y1 X2|

      if (isContributing(XB, Y1)) {
        send_xs.push_back(XB);
        targets.push_back(num_prev_ys);
      }

      ++num_prev_ys;

      if (isCovering(X2)) {
        send_xs.push_back(X2);
        targets.push_back(num_prev_ys);
      }
    } else if (sorted_xys.front().isResidual() &&
               sorted_xys.back().isResidual()) {
      const auto& X1 = sorted_xys.front();
      const auto& X2 = sorted_xys.back();

      if (isCovering(X1)) {
        send_xs.push_back(X1);
        targets.push_back(num_prev_ys);
      }

      if (isCovering(X2)) {
        send_xs.push_back(X2);
        targets.push_back(num_prev_ys);
      }
    } else if (sorted_xys.front().isRequest() &&
               sorted_xys.back().isRequest()) {
      const auto& XB = x_bcasted;
      const auto& Y1 = sorted_xys.front();
      const auto& Y2 = sorted_xys.back();

      if (prev_piece.isResidual()) {
        const auto& XP = prev_piece;

        // Example: XB Y Y Y X X XP |Y1 Y2|

        // Send XB if it contributes to Y1. Note that XB can
        // not contribute to Y2 as the previous piece is a X
        // piece.

        if (isContributing(XB, Y1)) {
          send_xs.push_back(XB);
          targets.push_back(num_prev_ys);
        }

        ++num_prev_ys;

        // Send XP if it contributes to Y2. Note that we are
        // not responsible for XP contributing to Y1 as this
        // is checked by the previous processor.

        if (isContributing(XP, Y2)) {
          send_xs.push_back(XP);
          targets.push_back(num_prev_ys);
        }
      } else {
        // Example: XB Y Y YP |Y1 Y2|

        // XB may contribute to Y1 and to Y2.

        if (isContributing(XB, Y1)) {
          send_xs.push_back(XB);
          targets.push_back(num_prev_ys);
        }

        ++num_prev_ys;

        if (isContributing(XB, Y2)) {
          send_xs.push_back(XB);
          targets.push_back(num_prev_ys);
        }
      }
    }

    const size_t max_pieces_recv_cnt = 2 + tlx::div_ceil(lmsgs_request_volume,
                                                         my_max_small_msg_size + 1);

    std::vector<Piece> recv_xs;
    recv_xs.reserve(max_pieces_recv_cnt);

    Ams::_internal::GroupMsgToPeAssignment::_internal::distributeMergedXPieces<AmsTag>(
      targets, send_xs, lmsg_request_exscan, lmsg_request_exscan + lmsgs_request_volume,
      recv_xs, tracker, group_comm);

    // No pieces of size zero.
    assert(std::find_if(recv_xs.begin(), recv_xs.end(),
                        [](const auto& piece) {
            return piece.size() == 0;
          }) == recv_xs.end());

    recv_xs = orderXPieces(recv_xs);

    // We do not expect duplicates here.
    assert(std::adjacent_find(recv_xs.begin(), recv_xs.end(),
                              [](const auto& left, const auto& right) {
            return left.pe() == right.pe();
          }) == recv_xs.end());

    /******************************************************************************
     * Construct Y pieces
     ******************************************************************************/

    // Push back large pieces (Y).
    auto y_pieces = _internal::createYPieces(lmsg_request_exscan, recv_partners,
                                             recv_sizes, my_max_small_msg_size);

    // No pieces of size zero.
    assert(std::find_if(y_pieces.begin(), y_pieces.end(),
                        [](const auto& piece) {
            return piece.size() == 0;
          }) == y_pieces.end());

    /******************************************************************************
     * Create request answers.
     * 1. Create large message answers by merging X and Y pieces.
     * 2. Merge large message answers with small message answers
     *    to a sorted sequence of answers.
     ******************************************************************************/

    assert(smsg_cnt == answer_targets.size());
    assert(std::is_sorted(answer_targets.begin(), answer_targets.end(),
                          [](const DistrRange& l, const DistrRange& r) {
            return l.pe < r.pe;
          }));

    mergeXY(recv_xs, y_pieces, my_group_idx, answers, answer_targets);

    assert(std::is_sorted(answer_targets.begin() + smsg_cnt, answer_targets.end(),
                          [](const DistrRange& l, const DistrRange& r) {
            return l.pe < r.pe;
          }));

    // Sort targets of small requests and large requests.
    DistrRanges tmp_answer_targets(answer_targets.size());
    std::merge(answer_targets.begin(), answer_targets.begin() + smsg_cnt,
               answer_targets.begin() + smsg_cnt, answer_targets.end(),
               tmp_answer_targets.begin(),
               [](const DistrRange& l, const DistrRange& r) {
            return l.pe < r.pe;
          });
    answer_targets = std::move(tmp_answer_targets);
  }

  assert(std::is_sorted(answer_targets.begin(), answer_targets.end(),
                        [](const DistrRange& l, const DistrRange& r) {
          return l.pe < r.pe;
        }));

  // No pieces of size zero.
  assert(std::find_if(answer_targets.begin(), answer_targets.end(),
                      [](const auto& piece) {
          return piece.size == 0;
        }) == answer_targets.end());

  // No pieces of size zero.
  assert(std::find_if(answers.begin(), answers.end(),
                      [](const auto& piece) {
          return piece.getSize() == 0;
        }) == answers.end());

  // Calculate maximum number of incoming request answers.
  size_t max_assigned_msgs = 0;
  for (size_t i = 0; i != group_cnt; ++i) {
    const auto max = _internal::maxSmallMsgSize(
      group_sizes[i], residual_capacity, nprocs);
    const auto size = loc_group_el_cnts[i];

    // +2 for overlapping residuals.
    max_assigned_msgs += 2 + tlx::div_ceil(size, max + 1);
  }

  std::vector<_internal::GroupDataRequest> msg_meta;   // (max_assigned_msgs);

  if (distr_strategy == DistributionStrategy::EXCHANGE_WITHOUT_RECV_SIZES) {
    Alltoallv::exchangeWithoutRecvSizes<AmsTag>(tracker, answers, answer_targets, msg_meta,
                                        group_cnt, max_assigned_msgs,
                                        _internal::GroupDataRequest::mpiType(), comm);
  } else if (distr_strategy == DistributionStrategy::EXCHANGE_WITH_RECV_SIZES) {
    Alltoallv::exchangeWithRecvSizes<AmsTag>(tracker, answers, answer_targets, msg_meta,
                                     _internal::GroupDataRequest::mpiType(), comm);
  } else {
    Alltoallv::exchangeWithRecvSizesAndPorts<AmsTag>(tracker, answers, answer_targets, msg_meta,
                                             _internal::GroupDataRequest::mpiType(), comm);
  }

    #ifndef NDEBUG
  // Number of assigned elements must be equal to requested number of elements.
  std::vector<size_t> loc_group_el_cnts_debug(group_cnt, 0);
  for (const auto& msg : msg_meta) {
    loc_group_el_cnts_debug[msg.getGroup()] += msg.getSize();
  }
  for (size_t i = 0; i != loc_group_el_cnts.size(); ++i) {
    assert(loc_group_el_cnts[i] == loc_group_el_cnts_debug[i]);
  }
    #endif


  // No pieces of size zero.
  assert(std::find_if(msg_meta.begin(), msg_meta.end(),
                      [](const auto& piece) {
          return piece.getSize() == 0;
        }) == msg_meta.end());

  const DistrRanges out_msgs = msgAssignmentToSendDescr(
    msg_meta, group_sizes_exscan, loc_group_el_cnts_exscan);

  // Sorted by target processes.
  assert(std::is_sorted(out_msgs.begin(), out_msgs.end(),
                        [](const auto& left, const auto& right) {
          return left.pe < right.pe;
        }));

  // No pieces of size zero.
  assert(std::find_if(out_msgs.begin(), out_msgs.end(),
                      [](const auto& piece) {
          return piece.size == 0;
        }) == out_msgs.end());

  // At most one message is send to a target process.
  assert(std::adjacent_find(out_msgs.begin(), out_msgs.end(),
                            [](const auto& left, const auto& right) {
          return left.pe == right.pe;
        }) == out_msgs.end());

  return { out_msgs, max_num_recv_msgs, residual_capacity };
}
}  // end namespace GroupMsgToPeAssignment
}  // end namespace _internal
}  // end namespace Ams
