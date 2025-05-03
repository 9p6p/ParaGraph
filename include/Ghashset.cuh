#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

namespace Gbuilder {
namespace Gpu {

template <typename cell_type = int32_t, typename id_type = uint32_t,
          uint32_t hash_set_size = 10 * 1024 * 1024>
struct HashSet {
  using cell_type_allied = cell_type;
  HashSet() {
    cudaMalloc(&d_hash_set_array_, hash_set_size * sizeof(cell_type));
    cudaMemset(d_hash_set_array_, Empty, hash_set_size * sizeof(cell_type));
  }

  ~HashSet() { cudaFree(d_hash_set_array_); }

  auto __device__ check_empty_or_insert(id_type u, cell_type* hash_set_array)
      -> bool {
    return !atomicCAS(&hash_set_array[u], 0, 1);
  }

  cell_type* d_hash_set_array_{nullptr};
  enum Kloc { Device, Host };
  enum Kempty { Empty = 0 };
};

}  // namespace Gpu
}  // namespace Gbuilder
