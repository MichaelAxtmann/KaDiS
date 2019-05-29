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

#include "GroupMsgToPeAssignment.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <tuple>
#include <vector>

#include "../Bitonic/Bitonic.hpp"
#include "BinomialTreePipeline.hpp"
#include "DistrRange.hpp"

#include <RBC.hpp>
#include <tlx/algorithm.hpp>

namespace Ams {
namespace _internal {
namespace GroupMsgToPeAssignment {
/*
 * @return Each outgoing message contains at least one
 * element. The outgoing messages are sorted by their target
 * processes. There is no target process which is covered by two
 * different outgoing messages.
 */
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
                                                         RBC::Comm comm) {
  int nprocs, myrank;
  RBC::Comm_size(comm, &nprocs);
  RBC::Comm_rank(comm, &myrank);

  DistrRanges out_msgs;

  const size_t num_groups = loc_group_el_cnts.size();

  // calculate the number of items each PE of a group shall receive: if
  // the
  // number of PEs does not divide glob_group_el_cnts[r], then we
  // distribute
  // the
  // empty slots evenly unto the last PEs (instead of just the last one),
  // by
  // raising the number of items of all PEs by one.

  std::vector<size_t> max_recv_el_cnt(num_groups);
  for (size_t i = 0; i < num_groups; i++) {
    max_recv_el_cnt[i] = tlx::div_ceil(glob_group_el_cnts[i], group_sizes[i]);
  }

  // calculate distribution of parts to other PEs.

  size_t offset = 0;

  for (size_t r = 0; r < num_groups; ++r) {
    // calculate distribute of this PE's items into group r.

    size_t remaining_group_els = loc_group_el_cnts[r];
    size_t num_previous_group_els = loc_group_el_distr_scan[r] - remaining_group_els;

    // calculate the number of "overflow" PEs in this group (PEs with
    // one item more than the rest).
    size_t num_large_pes = glob_group_el_cnts[r] % group_sizes[r];
    if (num_large_pes == 0) num_large_pes = group_sizes[r];

    while (remaining_group_els != 0) {
      // compute groupPE and residual
      size_t groupPE = 0;
      size_t residual = 0;

      // send elements to an overflow PE
      if (num_previous_group_els <
          num_large_pes * max_recv_el_cnt[r]) {
        groupPE = num_previous_group_els / max_recv_el_cnt[r];
        residual = (max_recv_el_cnt[r] -
                    (num_previous_group_els % max_recv_el_cnt[r]));
      } else {
        assert(num_previous_group_els >=
               num_large_pes * max_recv_el_cnt[r]);
        size_t rbeginSmall = num_previous_group_els -
                             num_large_pes * max_recv_el_cnt[r];
        groupPE = num_large_pes +
                  (rbeginSmall / (max_recv_el_cnt[r] - 1));
        residual = (max_recv_el_cnt[r] - 1 -
                    (rbeginSmall % (max_recv_el_cnt[r] - 1)));
      }
      size_t destPE = group_sizes_exscan[r] + groupPE;        // absolute PE

      const size_t send_count = std::min(remaining_group_els, residual);

      assert(send_count > 0);
      out_msgs.emplace_back(destPE, offset, send_count);

      offset += send_count;
      remaining_group_els -= send_count;
      num_previous_group_els += send_count;

      // We have sent a message to the last process of the
      // group. There should not be any elements left for this
      // group.
      assert(groupPE != group_sizes[r] - 1 || remaining_group_els == 0);
    }
  }

  // No messages of size zero.
  assert(std::find_if(out_msgs.begin(), out_msgs.end(),
                      [](const auto& msg) {
          return msg.size == 0;
        }) == out_msgs.end());

  // Sorted by target processes.
  assert(std::is_sorted(out_msgs.begin(), out_msgs.end(),
                        [](const auto& left, const auto& right) {
          return left.pe < right.pe;
        }));

  // At most one message is send to a target process.
  assert(std::adjacent_find(out_msgs.begin(), out_msgs.end(),
                            [](const auto& left, const auto& right) {
          return left.pe == right.pe;
        }) == out_msgs.end());

  return { out_msgs, std::min<size_t>(max_recv_el_cnt[my_group_idx], nprocs),
           max_recv_el_cnt[my_group_idx] };
}

std::tuple<DistrRanges, size_t, size_t> simpleAssignment(const std::vector<size_t>& group_sizes,
                                                         const std::vector<size_t>&
                                                         group_sizes_exscan,
                                                         const std::vector<size_t>&
                                                         loc_group_el_cnts,
                                                         const std::vector<size_t>&
                                                         glob_group_el_cnts,
                                                         size_t my_group_idx,
                                                         bool use_two_tree,
                                                         RBC::Comm comm) {
  std::vector<size_t> loc_group_el_dist_scan(loc_group_el_cnts.size());

  if (use_two_tree) {
    RBC::_internal::optimized::ScanTwotree(loc_group_el_cnts.data(),
                                           loc_group_el_dist_scan.data(),
                                           loc_group_el_dist_scan.size(),
                                           Common::getMpiType(loc_group_el_cnts),
                                           MPI_SUM, comm);
  } else {
    RBC::Scan(loc_group_el_cnts.data(),
              loc_group_el_dist_scan.data(), loc_group_el_dist_scan.size(),
              Common::getMpiType(loc_group_el_cnts), MPI_SUM, comm);
  }

  return simpleAssignment(group_sizes, group_sizes_exscan,
                          loc_group_el_cnts, loc_group_el_dist_scan,
                          glob_group_el_cnts, my_group_idx, comm);
}

namespace _internal {
/* @brief Merges sorted X pieces and Y pieces and generates request answers.
 *
 * The X pieces (Y pieces) cover a stripe of residual elements
 * (request elements).  The residual stripe may begin before the
 * request stripe starts (the first X piece extends into the first Y
 * piece) or may end after the request stripe ends (the last X piece
 * extends into the last Y piece).
 *
 * To guarantee correctness of this function, x_pieces as well as
 * y_pieces are not allowed to contain empty pieces.
 */
void mergeXY(std::vector<Piece>& x_pieces,
             std::vector<Piece>& y_pieces,
             size_t my_group_idx,
             std::vector<GroupDataRequest>& answers,
             DistrRanges& answer_targets) {
  // No pieces of size zero.
  assert(std::find_if(y_pieces.begin(), y_pieces.end(),
                      [](const auto& piece) {
            return piece.size() == 0;
          }) == y_pieces.end());

  // No pieces of size zero.
  assert(std::find_if(x_pieces.begin(), x_pieces.end(),
                      [](const auto& piece) {
            return piece.size() == 0;
          }) == x_pieces.end());

  // If we do not have Y pieces, we did not receive X pieces.
  assert(x_pieces.empty() == y_pieces.empty());
  if (x_pieces.empty()) {
    return;
  }

  size_t x_idx = 0;
  size_t y_idx = 0;
  // auto xptr = x_pieces.begin();
  // auto yptr = y_pieces.begin();

  // At least one x piece and one y piece.
  // First x piece must cover tip of first y piece.
  assert(x_pieces[x_idx].end() > y_pieces[y_idx].begin());
  assert(x_pieces[x_idx].begin() <= y_pieces[y_idx].begin());

  // We use the function SatisfyRequest to create
  // GroupDataRequests. This function requires that both pieces
  // start at the same position. However, the first X piece may
  // start before the first Y piece starts.
  x_pieces[x_idx].setBegin(y_pieces[y_idx].begin());

  while (x_idx < x_pieces.size() && y_idx < y_pieces.size()) {
    // If we still have a Y piece, there must be a X piece which
    // begins at the same position.
    assert(x_idx < x_pieces.size());
    assert(!x_pieces[x_idx].isEmpty());
    assert(!y_pieces[y_idx].isEmpty());
    assert(x_pieces[x_idx].begin() == y_pieces[y_idx].begin());

    answers.push_back(y_pieces[y_idx].satisfyRequest(x_pieces[x_idx], my_group_idx));

    assert(answers.back().getSize() > 0);

    if (!answer_targets.empty() &&
        answer_targets.back().pe == y_pieces[y_idx].pe()) {
      // There is already a message for the process of yptr.
      ++(answer_targets.back().size);
    } else {
      // We have to create a new message for the process of yptr.
      answer_targets.emplace_back(y_pieces[y_idx].pe(),
                                  answers.size() - 1, 1);
    }

    if (x_pieces[x_idx].isEmpty()) {
      ++x_idx;
    }
    if (y_pieces[y_idx].isEmpty()) {
      ++y_idx;
    }
  }

  // At most one X piece is left. This X piece extended into the
  // last Y piece.
  assert(x_pieces.size() - x_idx <= 1);
}

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
                                     std::vector<size_t> loc_group_el_cnts_exscan) {
  const auto group_cnt = group_sizes_exscan.size() - 1;
  std::vector<DistrRanges> grouped_descrs(group_cnt);

  for (auto& ass : assignments) {
    const auto group_pe_id = ass.getPe();
    const auto group_id = ass.getGroup();
    assert(group_id < group_sizes_exscan.size());
    const auto pe = group_sizes_exscan[group_id] + group_pe_id;
    assert(ass.getGroup() < loc_group_el_cnts_exscan.size());
    const auto offset = loc_group_el_cnts_exscan[ass.getGroup()];
    const auto size = ass.getSize();

    assert(group_id < group_cnt);

    // The messages for one group are ordered by processes.
    // We send at most one message to each process.
    // Case 'we send a small message': one message for the group
    // Case 'we send a large message':

    grouped_descrs[group_id].emplace_back(pe, offset, size);
    loc_group_el_cnts_exscan[ass.getGroup()] += size;
  }

  DistrRanges descrs;
  for (const auto& group_descrs : grouped_descrs) {
    for (const auto& descr : group_descrs) {
      assert(descrs.empty() || descrs.back().pe < descr.pe);
      descrs.push_back(descr);
    }
  }

  return descrs;
}

std::vector<Piece> orderXPieces(const std::vector<Piece>& x_pieces) {
  if (x_pieces.empty()) {
    return std::vector<Piece>{ };
  }

  auto min_pe = x_pieces.front().pe();
  for (size_t i = 1; i != x_pieces.size(); ++i) {
    min_pe = std::min(min_pe, x_pieces[i].pe());
  }

    #ifndef NDEBUG
  for (const auto piece : x_pieces) {
    assert(piece.pe() >= min_pe);
    assert(piece.pe() < min_pe + static_cast<int>(x_pieces.size()));
  }
  std::vector<bool> contains(x_pieces.size(), false);
  for (const auto piece : x_pieces) {
    contains[piece.pe() - min_pe] = true;
  }
  for (const auto contain : contains) {
    assert(contain);
  }
    #endif

  std::vector<Piece> ordered(x_pieces.size(), Piece{ });

  for (const auto& piece : x_pieces) {
    ordered[piece.pe() - min_pe] = piece;
  }

  assert(std::is_sorted(ordered.begin(), ordered.end(),
                        [](const Piece& p1, const Piece& p2) {
            return p1.pe() < p2.pe();
          }));

  return ordered;
}

/* @brief Creates Y pieces of requests.
 *
 * @param lmsgs_request_messages Requests sorted by requesting processes
 * @return Y pieces sorted by requesting processes.
 *
 */
std::vector<Piece> createYPieces(size_t lmsgs_request_exscan,
                                 const std::vector<int>& recv_partners,
                                 const std::vector<size_t>& recv_sizes,
                                 const size_t max_small_msg_size) {
  assert(std::is_sorted(recv_partners.begin(), recv_partners.end()));

  std::vector<Piece> y_pieces;
  size_t offset = lmsgs_request_exscan;
  for (size_t i = 0; i != recv_sizes.size(); ++i) {
    const auto size = recv_sizes[i];
    if (size > max_small_msg_size) {
      y_pieces.push_back(
        Piece::requestPiece(
          offset,
          offset + size,
          recv_partners[i]));
      offset += size;
    }
  }
  return y_pieces;
}

/*  @brief Maximum size of a small message.
 */
size_t maxSmallMsgSize(size_t group_size,
                       size_t residual_capacity, size_t nprocs) {
  // ceiling required: Some processes of this group could get one additional message.
  const size_t max_incoming_small_msgs = tlx::div_ceil(nprocs, group_size);
  // flooring required: Avoids overflow.
  const size_t smsg_residual_capacity = residual_capacity / 2;
  // flooring required: Avoids overflow.
  return smsg_residual_capacity / max_incoming_small_msgs;
}

std::vector<int> recvPartners(int nprocs, int my_group_size, int my_group_rank) {
  std::vector<int> recv_partners;

  while (my_group_rank < nprocs) {
    recv_partners.emplace_back(my_group_rank);
    my_group_rank += my_group_size;
  }

  return recv_partners;
}
}  // namespace _internal
}  // end namespace GroupMsgToPeAssignment
}  // end namespace _internal
}  // end namespace Ams
