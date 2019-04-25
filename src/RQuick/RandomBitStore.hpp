/*****************************************************************************
 * This file is part of the Project Karlsruhe Distributed Sorting Library
 * (KaDiS).
 *
 * Copyright (c) 2019, Michael Axtmann <michael.axtmann@kit.edu>
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

#include <cstddef>
#include <random>

class RandomBitStore {
 public:
  RandomBitStore();
  uint_fast64_t getNextBit(std::mt19937_64& async_gen);

 private:
  size_t pos_;
  uint_fast64_t bits_;
};

inline
RandomBitStore::RandomBitStore() :
  pos_(8 * sizeof(uint_fast64_t))
{ }

inline
uint_fast64_t RandomBitStore::getNextBit(std::mt19937_64& async_gen) {
  if (pos_ == 8 * sizeof(uint_fast64_t)) {
    bits_ = async_gen();
    pos_ = 0;
  }

  const auto res = bits_ & 1;
  bits_ = bits_ >> 1;
  ++pos_;
  return res;
}
