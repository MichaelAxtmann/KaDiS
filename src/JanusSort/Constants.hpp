/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiS).
 *
 * Copyright (c) 2016-2017, Armin Wiebigke <armin.wiebigke@gmail.com>
 * Copyright (C) 2016-2019, Michael Axtmann <michael.axtmann@kit.edu>
 * Copyright (c) 2016-2017, Tobias Heuer <tobias.heuer@gmx.net>
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************************/

#pragma once

namespace JanusSort {
namespace Constants {
// tags for communication
const int
  DISTR_SAMPLE_COUNT = 1000050,
  PIVOT_GATHER = 1000051,
  PIVOT_BCAST = 1000052,
  CALC_EXCH = 1000053,
  EXCHANGE_DATA_ASSIGNMENT = 1000054,
  EXCHANGE_SMALL = 1000055,
  EXCHANGE_LARGE = 1000056,
  TWO_PE = 1000057;
}  // namespace Constants
}  // namespace JanusSort
