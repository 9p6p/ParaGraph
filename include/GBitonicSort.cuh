/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

// TODO: This shouldn't be calling RAFT detail APIs
#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

namespace Gbuilder {
namespace Gpu {

namespace detail {

template <class K, class V>
__device__ __forceinline__ void swap_if_needed(K& k0, V& v0, K& k1, V& v1,
                                               bool const asc) {
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    auto const tmp_k = k0;
    k0 = k1;
    k1 = tmp_k;
    auto const tmp_v = v0;
    v0 = v1;
    v1 = tmp_v;
  }
}

template <class K, class V>
__device__ __forceinline__ void swap_if_needed(K& k0, V& v0,
                                               unsigned const lane_offset,
                                               bool const asc) {
  auto k1 = __shfl_xor_sync(~0u, k0, lane_offset);
  auto v1 = __shfl_xor_sync(~0u, v0, lane_offset);
  if ((k0 != k1) && ((k0 < k1) != asc)) {
    k0 = k1;
    v0 = v1;
  }
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
struct warp_merge_core {
  __device__ __forceinline__ void operator()(K k[N], V v[N],
                                             std::uint32_t const range,
                                             bool const asc) {
    auto const lane_id = threadIdx.x % warp_size;

    if (range == 1) {
      for (std::uint32_t b = 2; b <= N; b <<= 1) {
        for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
#pragma unroll
          for (std::uint32_t i = 0; i < N; i++) {
            std::uint32_t j = i ^ c;
            if (i >= j) continue;
            auto const line_id = i + (N * lane_id);
            auto const p = static_cast<bool>(line_id & b) ==
                           static_cast<bool>(line_id & c);
            assert(i < N && j < N);
            swap_if_needed(k[i], v[i], k[j], v[j], p);
          }
        }
      }
      return;
    }

    std::uint32_t const b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      auto const p =
          static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
      for (std::uint32_t i = 0; i < N; i++) {
        assert(i < N);

        swap_if_needed(k[i], v[i], c, p);
      }
    }
    auto const p = ((lane_id & b) == 0);
    for (std::uint32_t c = N / 2; c >= 1; c >>= 1) {
#pragma unroll
      for (std::uint32_t i = 0; i < N; i++) {
        std::uint32_t j = i ^ c;
        if (i >= j) continue;
        assert(i < N && j < N);

        swap_if_needed(k[i], v[i], k[j], v[j], p);
      }
    }
  }
};

template <class K, class V, unsigned warp_size>
struct warp_merge_core<K, V, 6, warp_size> {
  __device__ __forceinline__ void operator()(K k[6], V v[6],
                                             std::uint32_t const range,
                                             bool const asc) {
    constexpr unsigned N = 6;
    auto const lane_id = threadIdx.x % warp_size;

    if (range == 1) {
      for (std::uint32_t i = 0; i < N; i += 3) {
        auto const p = (i == 0);
        swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], p);
        swap_if_needed(k[1 + i], v[1 + i], k[2 + i], v[2 + i], p);
        swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], p);
      }
      auto const p = ((lane_id & 1) == 0);
      for (std::uint32_t i = 0; i < 3; i++) {
        std::uint32_t j = i + 3;
        swap_if_needed(k[i], v[i], k[j], v[j], p);
      }
      for (std::uint32_t i = 0; i < N; i += 3) {
        swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], p);
        swap_if_needed(k[1 + i], v[1 + i], k[2 + i], v[2 + i], p);
        swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], p);
      }
      return;
    }

    std::uint32_t const b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      auto const p =
          static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
      for (std::uint32_t i = 0; i < N; i++) {
        swap_if_needed(k[i], v[i], c, p);
      }
    }
    auto const p = ((lane_id & b) == 0);
    for (std::uint32_t i = 0; i < 3; i++) {
      std::uint32_t j = i + 3;
      swap_if_needed(k[i], v[i], k[j], v[j], p);
    }
    for (std::uint32_t i = 0; i < N; i += N / 2) {
      swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], p);
      swap_if_needed(k[1 + i], v[1 + i], k[2 + i], v[2 + i], p);
      swap_if_needed(k[0 + i], v[0 + i], k[1 + i], v[1 + i], p);
    }
  }
};

template <class K, class V, unsigned warp_size>
struct warp_merge_core<K, V, 3, warp_size> {
  __device__ __forceinline__ void operator()(K k[3], V v[3],
                                             std::uint32_t const range,
                                             bool const asc) {
    constexpr unsigned N = 3;
    auto const lane_id = threadIdx.x % warp_size;

    if (range == 1) {
      auto const p = ((lane_id & 1) == 0);
      swap_if_needed(k[0], v[0], k[1], v[1], p);
      swap_if_needed(k[1], v[1], k[2], v[2], p);
      swap_if_needed(k[0], v[0], k[1], v[1], p);
      return;
    }

    std::uint32_t const b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      auto const p =
          static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
      for (std::uint32_t i = 0; i < N; i++) {
        swap_if_needed(k[i], v[i], c, p);
      }
    }
    auto const p = ((lane_id & b) == 0);
    swap_if_needed(k[0], v[0], k[1], v[1], p);
    swap_if_needed(k[1], v[1], k[2], v[2], p);
    swap_if_needed(k[0], v[0], k[1], v[1], p);
  }
};

template <class K, class V, unsigned warp_size>
struct warp_merge_core<K, V, 2, warp_size> {
  __device__ __forceinline__ void operator()(K k[2], V v[2],
                                             std::uint32_t const range,
                                             bool const asc) {
    constexpr unsigned N = 2;
    auto const lane_id = threadIdx.x % warp_size;

    if (range == 1) {
      auto const p = ((lane_id & 1) == 0);
      swap_if_needed(k[0], v[0], k[1], v[1], p);
      return;
    }

    std::uint32_t const b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      auto const p =
          static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
      for (std::uint32_t i = 0; i < N; i++) {
        swap_if_needed(k[i], v[i], c, p);
      }
    }
    auto const p = ((lane_id & b) == 0);
    swap_if_needed(k[0], v[0], k[1], v[1], p);
  }
};

template <class K, class V, unsigned warp_size>
struct warp_merge_core<K, V, 1, warp_size> {
  __device__ __forceinline__ void operator()(K k[1], V v[1],
                                             std::uint32_t const range,
                                             bool const asc) {
    auto const lane_id = threadIdx.x % warp_size;
    std::uint32_t const b = range;
    for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
      auto const p =
          static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
      swap_if_needed(k[0], v[0], c, p);
    }
  }
};

}  // namespace detail

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ __forceinline__ void warp_merge(K k[N], V v[N], unsigned range,
                                           bool const asc = true) {
  detail::warp_merge_core<K, V, N, warp_size>{}(k, v, range, asc);
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ __forceinline__ void warp_sort(K k[N], V v[N],
                                          bool const asc = true) {
#pragma unroll
  for (std::uint32_t range = 1; range <= warp_size; range <<= 1) {
    warp_merge<K, V, N, warp_size>(k, v, range, asc);
  }
}

}  // namespace Gpu
}  // namespace Gbuilder
