#pragma once
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

namespace Gbuilder {
namespace Gpu {
template <typename key_type = uint32_t, uint32_t table_size = 1 << 11>
struct HashTable {
  // NOTE(shiwen): the key start at 1. Empty is 0.
  key_type list_[table_size];
  static constexpr uint32_t Kempty = 0;

  __device__ __forceinline__ uint32_t hash(key_type const& key) {
    // NOTE(shiwen): the table size must be 2^n.
    return key & (table_size - 1);
  }

  // TODO(shiwen): discard this...
  __device__ __forceinline__ bool test_and_set(key_type const& key) {
    auto slot = hash(key);
    auto old_key = atomicCAS(&list_[slot], Kempty, key + 1);
    while (old_key != Kempty && old_key != key + 1) {
      slot = (slot + 1) & (table_size - 1);
      old_key = atomicCAS(&list_[slot], Kempty, key + 1);
    }
    if (old_key == Kempty) {
      return true;
    }
    return false;
  }

  // NOTE(shiwen): poor performance, fix this.
  __device__ __forceinline__ bool warp_level_test_and_set(
      key_type const& key, uint32_t const& lane_id) {
    auto res = false;
    if (lane_id == 0) {
      auto slot = hash(key);
      // auto old_key = atomicCAS(&list_[slot], Kempty, key + 1);
      auto old_key = list_[slot];
      if (old_key == Kempty) {
        list_[slot] = key + 1;
      }
      while (old_key != Kempty && old_key != key + 1) {
        slot = (slot + 1) & (table_size - 1);
        // old_key = atomicCAS(&list_[slot], Kempty, key + 1);
        old_key = list_[slot];
        if (old_key == Kempty) {
          list_[slot] = key + 1;
        }
      }
      if (old_key == Kempty) {
        res = true;
      } else {
        res = false;
      }
    }
    return __shfl_sync(0XFFFFFFFF, res, 0);
  }

  __device__ __forceinline__ bool thread_level_set(key_type const& key) {
    auto slot = hash(key);
    auto old_key = atomicCAS(&list_[slot], Kempty, key + 1);
    while (old_key != Kempty && old_key != key + 1) {
      slot = (slot + 1) & (table_size - 1);
      old_key = atomicCAS(&list_[slot], Kempty, key + 1);
    }
    if (old_key == Kempty) {
      return true;
    }
    return false;
  }

  // NOTE(shiwen): for sync reset
  __device__ __forceinline__ bool reset_sync(uint32_t const& lane_id) {
    // FIXME(shiwen): maybe other cuda API..?
    constexpr uint32_t lane_width = 32;
    for (auto i = lane_id; i < table_size; i += lane_width) {
      list_[i] = Kempty;
    }
  }

  // // NOTE(shiwen): for async reset
  // __device__ __forceinline__ bool reset_async() {}

  // // NOTE(shiwen): barrier
  // __device__ __forceinline__ bool sync() {}
};
}  // namespace Gpu
}  // namespace Gbuilder
