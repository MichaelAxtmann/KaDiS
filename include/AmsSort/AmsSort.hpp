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

#include <functional>
#include <random>
#include <vector>

#include "Configuration.hpp"
#include "Tags.hpp"


#include <RBC.hpp>

namespace Ams {
/* @brief AMS-sort algorithm.
 *
 * AMS-sort algorithm from the publication "Robust Massively Parallel
 * Sorting"; https://doi.org/10.1145/2755573.2755595). We recommend
 * the default settings. The parameter 'k' determines the buckets on
 * each recursion level.
 *
 * @param comm MPI communicator
 *
 * @param mpi_type MPI datatype of element type T
 *
 * @param data Input and output vector
 *
 * @param k Number of buckets on each level. If the number of
 *   processes is not a power of k, the algorithm will adjust k
 *   appropriately.
 *
 * @param comp Comparator
 *
 * @param imbalance Maximum imbalance of the output. The value has to
 *   be larger than one.
 *
 * @param use_dma Flag which enables the deterministic message
 *   assignment algorithm. default = true
 *
 * @param Partitioning strategy. default part =
 *   PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING. See
 *   Configuration.hpp for more strategies.
 *
 * @param Data Distribution strategy. default distr =
 *   DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS. See
 *   Configuration.hpp for more strategies.
 *
 * @param use_ips4o Flag enables ips4o as base case sorting algorithm
 *   instead of std::sort. default = true.
 *
 * @param use_two_tree Use 2-tree collective operations from the RBC
 *   library instead of naive binomial tree collectives. Default = true.
 */
template <class T, class Comp = std::less<T> , class AmsTags = Ams::Tags<>>
void sort(MPI_Datatype mpi_type,
          std::vector<T>& data,
          int k,
          std::mt19937_64& async_gen,
          MPI_Comm comm,
          Comp comp = Comp(),
          double imbalance = 1.10,
          bool use_dma = true,
          PartitioningStrategy part = PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
          DistributionStrategy distr = DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS,
          bool use_ips4o = true,
          bool use_two_tree = true);

/* @brief AMS-sort algorithm.
 *
 * AMS-sort algorithm from the publication "Robust Massively Parallel
 * Sorting"; https://doi.org/10.1145/2755573.2755595).  We recommend
 * the default settings. The parameter 'ks' is a vector thatdetermines
 * the buckets on each recursion level.
 *
 * @param comm MPI communicator
 *
 * @param mpi_type MPI datatype of element type T
 *
 * @param data Input and output vector
 *
 * @param ks Vector of number of buckets k on each level. If the
 *   number of processes is not a power of k, the algorithm will
 *   adjust k appropriately.
 *
 * @param comp Comparator
 *
 * @param imbalance Maximum imbalance of the output. The value has to
 *   be larger than one.
 *
 * @param use_dma Flag which enables the deterministic message
 *   assignment algorithm. default = true
 *
 * @param Partitioning strategy. default part =
 *   PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING. See
 *   Configuration.hpp for more strategies.
 *
 * @param Data Distribution strategy. default distr =
 *   DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS. See
 *   Configuration.hpp for more strategies.
 *
 * @param use_ips4o Flag enables ips4o as base case sorting algorithm
 *   instead of std::sort. default = true.
 *
 * @param use_two_tree Use 2-tree collective operations from the RBC
 *   library instead of naive binomial tree collectives. Default = true.
 */
template <class T, class Comp = std::less<T> , class AmsTags = Ams::Tags<>>
void sort(MPI_Datatype mpi_type, std::vector<T>& data,
          std::vector<int>& ks,
          std::mt19937_64& async_gen,
          MPI_Comm comm,
          Comp comp = Comp(),
          double imbalance = 1.10,
          bool use_dma = true,
          PartitioningStrategy part = PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
          DistributionStrategy distr = DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS,
          bool use_ips4o = true,
          bool use_two_tree = true);

/* @brief AMS-sort algorithm.
 *
 * AMS-sort algorithm from the publication "Robust Massively Parallel
 * Sorting"; https://doi.org/10.1145/2755573.2755595).  We recommend
 * the default settings. The parameter 'k' determines the buckets on
 * each recursion level.
 *
 * @param comm MPI communicator
 *
 * @param mpi_type MPI datatype of element type T
 *
 * @param data Input and output vector
 *
 * @param l Number of recursionlevels. AMS-sort calculates the bucket
 *   sizes internally such that the sizes are about the same on each
 *   recursion level.
 *
 * @param comp Comparator
 *
 * @param imbalance Maximum imbalance of the output. The value has to
 *   be larger than one.
 *
 * @param use_dma Flag which enables the deterministic message
 *   assignment algorithm. default = true
 *
 * @param Partitioning strategy. default part =
 *   PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING. See
 *   Configuration.hpp for more strategies.
 *
 * @param Data Distribution strategy. default distr =
 *   DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS. See
 *   Configuration.hpp for more strategies.
 *
 * @param use_ips4o Flag enables ips4o as base case sorting algorithm
 *   instead of std::sort. default = true.
 *
 * @param use_two_tree Use 2-tree collective operations from the RBC
 *   library instead of naive binomial tree collectives. Default = true.
 */
template <class T, class Comp = std::less<T> , class AmsTags = Ams::Tags<>>
void sortLevel(MPI_Datatype mpi_type, std::vector<T>& data, int l,
               std::mt19937_64& async_gen,
               MPI_Comm comm,
               Comp comp = Comp(),
               double imbalance = 1.10,
               bool use_dma = true,
               PartitioningStrategy part =
                 PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
               DistributionStrategy distr =
                 DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS,
               bool use_ips4o = true,
               bool use_two_tree = true);

/* @brief AMS-sort algorithm.
 *
 * AMS-sort algorithm from the publication "Robust Massively Parallel
 * Sorting"; https://doi.org/10.1145/2755573.2755595). We recommend
 * the default settings. The parameter 'k' determines the buckets on
 * each recursion level.
 *
 * @param comm RBC communicator
 *
 * @param mpi_type MPI datatype of element type T
 *
 * @param data Input and output vector
 *
 * @param k Number of buckets on each level. If the number of
 *   processes is not a power of k, the algorithm will adjust k
 *   appropriately.
 *
 * @param comp Comparator
 *
 * @param imbalance Maximum imbalance of the output. The value has to
 *   be larger than one.
 *
 * @param use_dma Flag which enables the deterministic message
 *   assignment algorithm. default = true
 *
 * @param Partitioning strategy. default part =
 *   PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING. See
 *   Configuration.hpp for more strategies.
 *
 * @param Data Distribution strategy. default distr =
 *   DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS. See
 *   Configuration.hpp for more strategies.
 *
 * @param use_ips4o Flag enables ips4o as base case sorting algorithm
 *   instead of std::sort. default = true.
 *
 * @param use_two_tree Use 2-tree collective operations from the RBC
 *   library instead of naive binomial tree collectives. Default = true.
 */
template <class T, class Comp = std::less<T> , class AmsTags = Ams::Tags<>>
void sort(MPI_Datatype mpi_type,
          std::vector<T>& data,
          int k,
          std::mt19937_64& async_gen,
          const RBC::Comm& comm,
          Comp comp = Comp(),
          double imbalance = 1.10,
          bool use_dma = true,
          PartitioningStrategy part = PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
          DistributionStrategy distr = DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS,
          bool use_ips4o = true,
          bool use_two_tree = true);

/* @brief AMS-sort algorithm.
 *
 * AMS-sort algorithm from the publication "Robust Massively Parallel
 * Sorting"; https://doi.org/10.1145/2755573.2755595).  We recommend
 * the default settings. The parameter 'ks' is a vector thatdetermines
 * the buckets on each recursion level.
 *
 * @param comm RBC communicator
 *
 * @param mpi_type MPI datatype of element type T
 *
 * @param data Input and output vector
 *
 * @param ks Vector of number of buckets k on each level. If the
 *   number of processes is not a power of k, the algorithm will
 *   adjust k appropriately.
 *
 * @param comp Comparator
 *
 * @param imbalance Maximum imbalance of the output. The value has to
 *   be larger than one.
 *
 * @param use_dma Flag which enables the deterministic message
 * assignment algorithm. default = true
 *
 * @param Partitioning strategy. default part =
 *   PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING. See
 *   Configuration.hpp for more strategies.
 *
 * @param Data Distribution strategy. default distr =
 *   DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS. See
 *   Configuration.hpp for more strategies.
 *
 * @param use_ips4o Flag enables ips4o as base case sorting algorithm
 *   instead of std::sort. default = true.
 *
 * @param use_two_tree Use 2-tree collective operations from the RBC
 *   library instead of naive binomial tree collectives. Default = true.
 */
template <class T, class Comp = std::less<T> , class AmsTags = Ams::Tags<>>
void sort(MPI_Datatype mpi_type, std::vector<T>& data,
          std::vector<int>& ks,
          std::mt19937_64& async_gen,
          const RBC::Comm& comm,
          Comp comp = Comp(),
          double imbalance = 1.10,
          bool use_dma = true,
          PartitioningStrategy part = PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
          DistributionStrategy distr = DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS,
          bool use_ips4o = true,
          bool use_two_tree = true);

/* @brief AMS-sort algorithm.
 *
 * AMS-sort algorithm from the publication "Robust Massively Parallel
 * Sorting"; https://doi.org/10.1145/2755573.2755595).  We recommend
 * the default settings. The parameter 'k' determines the buckets on
 * each recursion level.
 *
 * @param comm RBC communicator
 *
 * @param mpi_type MPI datatype of element type T
 *
 * @param data Input and output vector
 *
 * @param l Number of recursionlevels. AMS-sort calculates the bucket
 *   sizes internally such that the sizes are about the same on each
 *   recursion level.
 *
 * @param comp Comparator
 *
 * @param imbalance Maximum imbalance of the output. The value has to
 *   be larger than one.
 *
 * @param use_dma Flag which enables the deterministic message
 *   assignment algorithm. default = true
 *
 * @param Partitioning strategy. default part =
 *   PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING. See
 *   Configuration.hpp for more strategies.
 *
 * @param Data Distribution strategy. default distr =
 *   DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS. See
 *   Configuration.hpp for more strategies.
 *
 * @param use_ips4o Flag enables ips4o as base case sorting algorithm
 *   instead of std::sort. default = true.
 *
 * @param use_two_tree Use 2-tree collective operations from the RBC
 *   library instead of naive binomial tree collectives. Default = true.
 */
template <class T, class Comp = std::less<T> , class AmsTags = Ams::Tags<> >
void sortLevel(MPI_Datatype mpi_type, std::vector<T>& data, int l,
               std::mt19937_64& async_gen,
               const RBC::Comm& comm,
               Comp comp = Comp(),
               double imbalance = 1.10,
               bool use_dma = true,
               PartitioningStrategy part =
                 PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
               DistributionStrategy distr =
                 DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS,
               bool use_ips4o = true,
               bool use_two_tree = true);

template <class T, class Tracker, class Comp = std::less<T> , class AmsTags = Ams::Tags<> >
void sortTracker(MPI_Datatype mpi_type, std::vector<T>& data, int k,
                 std::mt19937_64& async_gen,
                 Tracker& tracker,
                 const RBC::Comm& comm,
                 Comp comp = Comp(),
                 double imbalance = 1.10,
                 bool use_dma = true,
                 PartitioningStrategy part =
                   PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
                 DistributionStrategy distr =
                   DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS,
                 bool use_ips4o = true,
                 bool use_two_tree = true);

template <class T, class Tracker, class Comp = std::less<T> , class AmsTags = Ams::Tags<> >
void sortTracker(MPI_Datatype mpi_type, std::vector<T>& data,
                 std::vector<int>& ks,
                 std::mt19937_64& async_gen,
                 Tracker& tracker,
                 const RBC::Comm& comm,
                 Comp comp = Comp(),
                 double imbalance = 1.10,
                 bool use_dma = true,
                 PartitioningStrategy part =
                   PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
                 DistributionStrategy distr =
                   DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS,
                 bool use_ips4o = true,
                 bool use_two_tree = true);

template <class T, class Tracker, class Comp = std::less<> , class AmsTags = Ams::Tags<> >
void sortTrackerLevel(MPI_Datatype mpi_type, std::vector<T>& data, int l,
                      std::mt19937_64& async_gen,
                      Tracker& tracker,
                      const RBC::Comm& comm,
                      Comp comp = Comp(),
                      double imbalance = 1.10,
                      bool use_dma = true,
                      PartitioningStrategy part =
                        PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
                      DistributionStrategy distr =
                        DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS,
                      bool use_ips4o = true,
                      bool use_two_tree = true);

template <class T, class Tracker, class Comp = std::less<T> , class AmsTags = Ams::Tags<> >
void sortTracker(MPI_Datatype mpi_type, std::vector<T>& data, int k,
                 std::mt19937_64& async_gen,
                 Tracker& tracker,
                 MPI_Comm comm,
                 Comp comp = Comp(),
                 double imbalance = 1.10,
                 bool use_dma = true,
                 PartitioningStrategy part =
                   PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
                 DistributionStrategy distr =
                   DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS,
                 bool use_ips4o = true,
                 bool use_two_tree = true);

template <class T, class Tracker, class Comp = std::less<T> , class AmsTags = Ams::Tags<> >
void sortTracker(MPI_Datatype mpi_type, std::vector<T>& data,
                 std::vector<int>& ks,
                 std::mt19937_64& async_gen,
                 Tracker& tracker,
                 MPI_Comm comm,
                 Comp comp = Comp(),
                 double imbalance = 1.10,
                 bool use_dma = true,
                 PartitioningStrategy part =
                   PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
                 DistributionStrategy distr =
                   DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS,
                 bool use_ips4o = true,
                 bool use_two_tree = true);

template <class T, class Tracker, class Comp = std::less<> , class AmsTags = Ams::Tags<> >
void sortTrackerLevel(MPI_Datatype mpi_type, std::vector<T>& data, int l,
                      std::mt19937_64& async_gen,
                      Tracker& tracker,
                      MPI_Comm comm,
                      Comp comp = Comp(),
                      double imbalance = 1.10,
                      bool use_dma = true,
                      PartitioningStrategy part =
                        PartitioningStrategy::INPLACE_AND_EQUAL_BUCKET_PARTITIONING,
                      DistributionStrategy distr =
                        DistributionStrategy::EXCHANGE_WITH_RECV_SIZES_AND_PORTS,
                      bool use_ips4o = true,
                      bool use_two_tree = true);
}  // namespace Ams

#include "../../src/AmsSort/AmsSort.hpp"
