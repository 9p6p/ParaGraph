#pragma once

#include "GBitonicSort.cuh"
#include "Ghashset.cuh"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <unistd.h>

namespace Gbuilder {
namespace Gpu {

template <uint32_t grid_size, uint32_t block_size, uint32_t query_num,
          uint32_t topk, uint32_t max_in_degree, uint32_t tomb = 0xFFFFFFFF,
          typename data_type = float, typename id_type = uint32_t>
__global__ void match_kernel(id_type const* __restrict__ gt_ids,
                             id_type* __restrict__ top1_match_graph,
                             id_type* __restrict__ match_match_graph) {
  // NOTE(shiwen): max_in_degree is same as topk.
  static_assert(max_in_degree == topk);
  constexpr auto stride = block_size * grid_size;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;

  // TODO(shiwen): use shared memory.
  for (auto query_idx = thread_idx; query_idx < query_num;
       query_idx += stride) {
    auto top1_base_id = gt_ids[query_idx * topk + 0];
    auto gt_idx = 1;
    auto base_id = gt_ids[query_idx * topk + gt_idx];
    // top1 match
    if (atomicCAS(&top1_match_graph[top1_base_id * max_in_degree + 0], tomb,
                  base_id) == tomb) {
      for (gt_idx = 2; gt_idx < topk; gt_idx++) {
        base_id = gt_ids[query_idx * topk + gt_idx];
        auto neighbor_idx = gt_idx - 1;
        top1_match_graph[top1_base_id * max_in_degree + neighbor_idx] = base_id;
      }
    }
    // greedy match
    gt_idx = 0;
    base_id = gt_ids[query_idx * topk + gt_idx];
    auto greedy_match_idx = 1;
    auto greedy_match_base_id = gt_ids[query_idx * topk + greedy_match_idx];
    // find the greedy_match_idx and its base id
    while (
        greedy_match_idx < topk &&
        atomicCAS(&match_match_graph[greedy_match_base_id * max_in_degree + 0],
                  tomb, base_id) != tomb) {
      greedy_match_idx++;
      greedy_match_base_id = gt_ids[query_idx * topk + greedy_match_idx];
    }
    // find the greedy match.
    if (greedy_match_idx < topk) {
      auto neighbor_idx = 1;
      for (gt_idx = 0; gt_idx < topk; gt_idx++) {
        if (gt_idx == greedy_match_idx) {
          continue;
        }
        base_id = gt_ids[query_idx * topk + gt_idx];
        match_match_graph[greedy_match_base_id * max_in_degree + neighbor_idx] =
            base_id;
        neighbor_idx++;
      }
    }
  }
}

// NOTE(shiwen): just for match top1.
template <uint32_t grid_size, uint32_t block_size, uint32_t query_num,
          uint32_t topk, uint32_t max_degree, uint32_t tomb = 0xFFFFFFFF,
          typename data_type = float, typename id_type = uint32_t>
__global__ void match_top1_kernel(id_type* __restrict__ gt_ids,
                                  id_type* __restrict__ top1_match_graph) {
  // NOTE(shiwen): max_degree is same as topk.
  // static_assert(max_degree == topk - 1);
  constexpr auto stride = block_size * grid_size;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;

  // TODO(shiwen): use shared memory.
  for (auto query_idx = thread_idx; query_idx < query_num;
       query_idx += stride) {
    auto top1_base_id = gt_ids[query_idx * topk];
    auto gt_idx = 1;
    auto neighbor_base_id = gt_ids[query_idx * topk + gt_idx];
    // top1 match
    if (atomicCAS(&top1_match_graph[top1_base_id * max_degree], tomb,
                  neighbor_base_id) == tomb) {
      for (gt_idx = 2; gt_idx < topk; gt_idx++) {
        neighbor_base_id = gt_ids[query_idx * topk + gt_idx];
        auto neighbor_idx = gt_idx - 1;
        top1_match_graph[top1_base_id * max_degree + neighbor_idx] =
            neighbor_base_id;
      }
    }
  }
}

// NOTE(shiwen): just for match top1.
template <uint32_t grid_size, uint32_t block_size, uint32_t query_num,
          uint32_t topk, uint32_t max_degree, uint32_t match_real_degree,
          uint32_t tomb = 0xFFFFFFFF, typename data_type = float,
          typename id_type = uint32_t>
__global__ void match_top1_kernel_v0(id_type* __restrict__ gt_ids,
                                     id_type* __restrict__ top1_match_graph) {
  // NOTE(shiwen): max_degree is same as topk.
  static_assert(max_degree >= match_real_degree);
  constexpr auto stride = block_size * grid_size;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;

  // TODO(shiwen): use shared memory.
  for (auto query_idx = thread_idx; query_idx < query_num;
       query_idx += stride) {
    auto top1_base_id = gt_ids[query_idx * topk];
    auto gt_idx = 1;
    auto neighbor_base_id = gt_ids[query_idx * topk + gt_idx];
    // top1 match
    if (atomicCAS(&top1_match_graph[top1_base_id * max_degree], tomb,
                  neighbor_base_id) == tomb) {
      for (gt_idx = 2; gt_idx < topk; gt_idx++) {
        neighbor_base_id = gt_ids[query_idx * topk + gt_idx];
        auto neighbor_idx = gt_idx - 1;
        top1_match_graph[top1_base_id * max_degree + neighbor_idx] =
            neighbor_base_id;
      }
      for (auto i = match_real_degree - 1; i < max_degree; i++) {
        top1_match_graph[top1_base_id * max_degree + i] = tomb;
      }
    }
  }
}

// // each warp is assigned to calculate all the distance between base_vector_id
// // and its neighbor.
// template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
//           uint32_t max_in_degree, uint32_t tomb = 0xFFFFFFFF, uint32_t dim,
//           uint32_t shared_memory_size, typename data_type = float,
//           typename id_type = uint32_t>
// __global__ void compute_ip_distance_kernel(
//     data_type const* __restrict__ base_data, id_type const* __restrict__
//     graph, data_type* __restrict__ neighbor_distance) {
//   constexpr auto lane_width = 32;
//   constexpr auto warp_per_block = block_size / lane_width;
//   // each warp need to store the base vector and the neighbor vector to
//   // calculate the distance between them.
//   constexpr auto shared_memory_size_per_warp = dim * sizeof(data_type) * 2;
//   constexpr auto global_warp_num = block_size * grid_size / lane_width;
//   // NOTE(shiwen): check the allocation of shared memory is correct.
//   static_assert(shared_memory_size ==
//                 shared_memory_size_per_warp * warp_per_block);

//   // NOTE(shiwen): use offset instead of bytes address.
//   extern __shared__ data_type sdata[];

//   auto const global_warp_id =
//       (threadIdx.x + blockDim.x * blockIdx.x) / lane_width;
//   auto const local_warp_id = threadIdx.x / lane_width;
//   auto const lane_id = threadIdx.x % lane_width;
//   auto const base_vector_offset =
//       local_warp_id * shared_memory_size_per_warp / sizeof(data_type);
//   auto const neighbor_vector_offset = base_vector_offset + dim;

//   for (auto base_vector_id = global_warp_id; base_vector_id < base_num;
//        base_vector_id += global_warp_num) {
//     for (auto i = lane_id; i < dim; i += lane_width) {
//       assert(i < dim);
//       sdata[base_vector_offset + i] = base_data[base_vector_id * dim + i];
//     }

//     // NOTE(shiwen): in case of dim % 32 != 0
//     // TODO(shiwen): need this primitive?
//     __syncwarp();

//     for (auto neighbor_idx = 0; neighbor_idx < max_in_degree; neighbor_idx++)
//     {
//       assert(neighbor_idx < max_in_degree);
//       auto neighbor_base_id =
//           graph[base_vector_id * max_in_degree + neighbor_idx];
//       if (neighbor_base_id == tomb) {
//         break;
//       }
//       for (auto i = lane_id; i < dim; i += lane_width) {
//         assert(i < dim);
//         sdata[neighbor_vector_offset + i] =
//             base_data[neighbor_base_id * dim + i];
//       }

//       // NOTE(shiwen): in case of dim % 32 != 0
//       // TODO(shiwen): need this primitive?
//       __syncwarp();

//       data_type sum = 0;
//       for (auto i = lane_id; i < dim; i += lane_width) {
//         assert(i < dim);
//         sum +=
//             sdata[base_vector_offset + i] * sdata[neighbor_vector_offset +
//             i];
//       }
//       // warp level reduce.
//       for (auto offset = 16; offset > 0; offset >>= 1) {
//         sum += __shfl_down_sync(0xffffffff, sum, offset);
//       }
//       if (lane_id == 0) {
//         assert(neighbor_idx < max_in_degree);
//         neighbor_distance[base_vector_id * max_in_degree + neighbor_idx] =
//         -sum;
//       }

//       // NOTE(shiwen): in case of dim % 32 != 0
//       // TODO(shiwen): need this primitive?
//       __syncwarp();
//     }
//   }
// }

template <typename id_type = uint32_t, typename data_type = float,
          uint32_t max_in_degree = 64, uint32_t dim = 128>
struct compute_sort_warp_state {
  data_type base_data[dim];
  data_type neighbor_data[dim];
  data_type distances[max_in_degree];
  id_type neighbor_ids[max_in_degree];
};

// FIXME(shiwen): too much __syncwarp()!!!
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t max_in_degree, uint32_t tomb = 0xFFFFFFFF, uint32_t dim,
          uint32_t shared_memory_size, typename data_type = float,
          typename id_type = uint32_t>
__global__ void compute_and_sort_ip_distance_kernel(
    data_type const* __restrict__ base_data, id_type* __restrict__ graph,
    data_type* __restrict__ neighbor_distance) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp =
      sizeof(compute_sort_warp_state<id_type, data_type, max_in_degree, dim>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  extern __shared__
      compute_sort_warp_state<id_type, data_type, max_in_degree, dim>
          warp_states[];

  uint32_t const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
  uint32_t const local_warp_id = threadIdx.x / lane_width;
  uint32_t const lane_id = threadIdx.x % lane_width;

  // Shared memory layout per warp
  data_type* base_vector_sdata = warp_states[local_warp_id].base_data;
  data_type* neighbor_vector_sdata = warp_states[local_warp_id].neighbor_data;
  data_type* distance_sdata = warp_states[local_warp_id].distances;
  id_type* neighbor_id_sdata = warp_states[local_warp_id].neighbor_ids;

  for (uint32_t base_vector_id = global_warp_id; base_vector_id < base_num;
       base_vector_id += global_warp_num) {
    // Load base vector
    for (uint32_t i = lane_id; i < dim; i += lane_width) {
      assert(i < dim);
      base_vector_sdata[i] = base_data[base_vector_id * dim + i];
    }
    // __syncwarp();

    uint32_t min_invalid_neighbor_idx = max_in_degree;
    // Process neighbors and collect distances
    for (uint32_t neighbor_idx = 0; neighbor_idx < max_in_degree;
         ++neighbor_idx) {
      // __syncwarp();
      assert(neighbor_idx < max_in_degree);
      assert(base_vector_id < base_num);

      id_type const neighbor_id =
          graph[base_vector_id * max_in_degree + neighbor_idx];
      // __syncwarp();

      if (neighbor_id == tomb) {
        min_invalid_neighbor_idx = neighbor_idx;
        // NOTE(shiwen): set all distance of tomb id to FLT_MAX
        if (lane_id == 0) {
          for (uint32_t neighbor_idx = min_invalid_neighbor_idx;
               neighbor_idx < max_in_degree; neighbor_idx++) {
            assert(neighbor_idx < max_in_degree);
            distance_sdata[neighbor_idx] = FLT_MAX;
          }
        }
        // __syncwarp();

        break;
      }

      // Load neighbor vector
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        assert(neighbor_id < base_num);

        neighbor_vector_sdata[i] = base_data[neighbor_id * dim + i];
      }
      // __syncwarp();

      data_type sum = 0;
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        sum += base_vector_sdata[i] * neighbor_vector_sdata[i];
      }

      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      }
      // __syncwarp();

      if (lane_id == 0) {
        assert(neighbor_idx < max_in_degree);
        distance_sdata[neighbor_idx] = -sum;
        neighbor_id_sdata[neighbor_idx] = neighbor_id;
      }
      __syncwarp();
    }

    // __syncwarp();

    // FIXME(shiwen): check the 3rd template.
    warp_sort<data_type, id_type, max_in_degree, lane_width>(
        distance_sdata, neighbor_id_sdata, true);

    // __syncwarp();

    for (uint32_t i = lane_id; i < max_in_degree; i += lane_width) {
      assert(i < max_in_degree);
      if (i < min_invalid_neighbor_idx) {
        neighbor_distance[base_vector_id * max_in_degree + i] =
            distance_sdata[i];
        graph[base_vector_id * max_in_degree + i] = neighbor_id_sdata[i];
      } else if (i >= min_invalid_neighbor_idx) {
        neighbor_distance[base_vector_id * max_in_degree + i] = FLT_MAX;
        graph[base_vector_id * max_in_degree + i] = tomb;
      }
    }
    __syncwarp();
  }
}

// FIXME(shiwen): too much __syncwarp()!!!
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t max_in_degree, uint32_t tomb = 0xFFFFFFFF, uint32_t dim,
          uint32_t shared_memory_size, typename data_type = float,
          typename id_type = uint32_t>
__global__ void sort_neighbor_kernel(data_type const* __restrict__ base_data,
                                     id_type* __restrict__ graph) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp =
      sizeof(compute_sort_warp_state<id_type, data_type, max_in_degree, dim>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  extern __shared__
      compute_sort_warp_state<id_type, data_type, max_in_degree, dim>
          shared_warp_states[];

  uint32_t const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
  uint32_t const local_warp_id = threadIdx.x / lane_width;
  uint32_t const lane_id = threadIdx.x % lane_width;

  // Shared memory layout per warp
  data_type* base_vector_sdata = shared_warp_states[local_warp_id].base_data;
  data_type* neighbor_vector_sdata =
      shared_warp_states[local_warp_id].neighbor_data;
  data_type* distance_sdata = shared_warp_states[local_warp_id].distances;
  id_type* neighbor_id_sdata = shared_warp_states[local_warp_id].neighbor_ids;

  for (uint32_t base_vector_id = global_warp_id; base_vector_id < base_num;
       base_vector_id += global_warp_num) {
    // Load base vector
    for (uint32_t i = lane_id; i < dim; i += lane_width) {
      assert(i < dim);
      base_vector_sdata[i] = base_data[base_vector_id * dim + i];
    }
    // __syncwarp();

    uint32_t min_invalid_neighbor_idx = max_in_degree;
    // Process neighbors and collect distances
    for (uint32_t neighbor_idx = 0; neighbor_idx < max_in_degree;
         ++neighbor_idx) {
      // __syncwarp();
      assert(neighbor_idx < max_in_degree);
      assert(base_vector_id < base_num);

      id_type const neighbor_id =
          graph[base_vector_id * max_in_degree + neighbor_idx];
      // __syncwarp();

      if (neighbor_id == tomb) {
        min_invalid_neighbor_idx = neighbor_idx;
        // NOTE(shiwen): set all distance of tomb id to FLT_MAX
        if (lane_id == 0) {
          for (uint32_t neighbor_idx = min_invalid_neighbor_idx;
               neighbor_idx < max_in_degree; neighbor_idx++) {
            assert(neighbor_idx < max_in_degree);
            distance_sdata[neighbor_idx] = FLT_MAX;
          }
        }
        // __syncwarp();

        break;
      }

      // Load neighbor vector
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        assert(neighbor_id < base_num);

        neighbor_vector_sdata[i] = base_data[neighbor_id * dim + i];
      }
      // __syncwarp();

      data_type sum = 0;
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        sum += base_vector_sdata[i] * neighbor_vector_sdata[i];
      }

      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      }
      // __syncwarp();

      if (lane_id == 0) {
        assert(neighbor_idx < max_in_degree);
        distance_sdata[neighbor_idx] = -sum;
        neighbor_id_sdata[neighbor_idx] = neighbor_id;
      }
      __syncwarp();
    }

    // __syncwarp();

    // FIXME(shiwen): check the 3rd template.
    warp_sort<data_type, id_type, max_in_degree, lane_width>(
        distance_sdata, neighbor_id_sdata, true);

    // __syncwarp();

    for (uint32_t i = lane_id; i < max_in_degree; i += lane_width) {
      assert(i < max_in_degree);
      if (i < min_invalid_neighbor_idx) {
        graph[base_vector_id * max_in_degree + i] = neighbor_id_sdata[i];
      } else if (i >= min_invalid_neighbor_idx) {
        graph[base_vector_id * max_in_degree + i] = tomb;
      }
    }
    __syncwarp();
  }
}

// FIXME(shiwen): too much __syncwarp()!!!
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t max_in_degree, uint32_t tomb = 0xFFFFFFFF, uint32_t dim,
          uint32_t shared_memory_size, typename data_type = float,
          typename id_type = uint32_t>
__global__ void reverse_compute_and_sort_ip_distance_kernel(
    data_type const* __restrict__ base_data, id_type* __restrict__ graph,
    data_type* __restrict__ neighbor_distance) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp =
      sizeof(compute_sort_warp_state<id_type, data_type, max_in_degree, dim>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  extern __shared__
      compute_sort_warp_state<id_type, data_type, max_in_degree, dim>
          warp_states[];

  uint32_t const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
  uint32_t const local_warp_id = threadIdx.x / lane_width;
  uint32_t const lane_id = threadIdx.x % lane_width;

  // Shared memory layout per warp
  data_type* base_vector_sdata = warp_states[local_warp_id].base_data;
  data_type* neighbor_vector_sdata = warp_states[local_warp_id].neighbor_data;
  data_type* distance_sdata = warp_states[local_warp_id].distances;
  id_type* neighbor_id_sdata = warp_states[local_warp_id].neighbor_ids;

  for (uint32_t base_vector_id = global_warp_id; base_vector_id < base_num;
       base_vector_id += global_warp_num) {
    auto valid_num = graph[base_vector_id * max_in_degree] - 1;
    assert(valid_num < max_in_degree);
    if (valid_num == 0) {
      continue;
    }
    // Load base vector
    for (uint32_t i = lane_id; i < dim; i += lane_width) {
      assert(i < dim);
      base_vector_sdata[i] = base_data[base_vector_id * dim + i];
    }
    // __syncwarp();

    // Process neighbors and collect distances
    neighbor_vector_sdata[0] = FLT_MAX;
    for (uint32_t neighbor_idx = 1; neighbor_idx < valid_num + 1;
         ++neighbor_idx) {
      // __syncwarp();
      assert(neighbor_idx < max_in_degree);
      assert(base_vector_id < base_num);

      id_type const neighbor_id =
          graph[base_vector_id * max_in_degree + neighbor_idx];

      assert(neighbor_id < base_num);

      // Load neighbor vector
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        assert(neighbor_id < base_num);

        neighbor_vector_sdata[i] = base_data[neighbor_id * dim + i];
      }
      // __syncwarp();

      data_type sum = 0;
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        sum += base_vector_sdata[i] * neighbor_vector_sdata[i];
      }

      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      }
      // __syncwarp();

      if (lane_id == 0) {
        assert(neighbor_idx < max_in_degree);
        distance_sdata[neighbor_idx] = -sum;
        neighbor_id_sdata[neighbor_idx] = neighbor_id;
      }
      __syncwarp();
    }

    for (auto i = valid_num + 1; i < max_in_degree; i++) {
      neighbor_id_sdata[i] = FLT_MAX;
    }

    // __syncwarp();

    // FIXME(shiwen): check the 3rd template.
    warp_sort<data_type, id_type, max_in_degree, lane_width>(
        distance_sdata, neighbor_id_sdata, true);

    neighbor_id_sdata[max_in_degree - 1] = valid_num + 1;

    // __syncwarp();

    for (uint32_t i = lane_id; i < valid_num; i += lane_width) {
      neighbor_distance[base_vector_id * max_in_degree + i] = distance_sdata[i];
      graph[base_vector_id * max_in_degree + i] = neighbor_id_sdata[i];
    }
    if (lane_id == 0) {
      graph[base_vector_id * max_in_degree + max_in_degree - 1] = valid_num + 1;
    }
    __syncwarp();
  }
}

// FIXME(shiwen): too much __syncwarp()!!!
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t reverse_graph_degree, uint32_t tomb = 0xFFFFFFFF,
          uint32_t dim, uint32_t shared_memory_size, typename data_type = float,
          typename id_type = uint32_t>
__global__ void reverse_sort_kernel(data_type const* __restrict__ base_data,
                                    id_type* __restrict__ reverse_graph) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp = sizeof(
      compute_sort_warp_state<id_type, data_type, reverse_graph_degree, dim>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  extern __shared__
      compute_sort_warp_state<id_type, data_type, reverse_graph_degree, dim>
          sort_warp_states[];

  uint32_t const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
  uint32_t const local_warp_id = threadIdx.x / lane_width;
  uint32_t const lane_id = threadIdx.x % lane_width;

  // Shared memory layout per warp
  data_type* base_vector_sdata = sort_warp_states[local_warp_id].base_data;
  data_type* neighbor_vector_sdata =
      sort_warp_states[local_warp_id].neighbor_data;
  data_type* distance_sdata = sort_warp_states[local_warp_id].distances;
  id_type* neighbor_id_sdata = sort_warp_states[local_warp_id].neighbor_ids;

  for (uint32_t base_vector_id = global_warp_id; base_vector_id < base_num;
       base_vector_id += global_warp_num) {
    auto valid_num = reverse_graph[base_vector_id * reverse_graph_degree];
    // assert(valid_num < rever_graph_degree);
    if (valid_num == 0) {
      if (lane_id == 0) {
        reverse_graph[base_vector_id * reverse_graph_degree +
                      reverse_graph_degree - 1] = valid_num;
      }
      continue;
    }
    // Load base vector
    for (uint32_t i = lane_id; i < dim; i += lane_width) {
      assert(i < dim);
      base_vector_sdata[i] = base_data[base_vector_id * dim + i];
    }
    // __syncwarp();

    // Process neighbors and collect distances
    distance_sdata[0] = FLT_MAX;
    for (uint32_t neighbor_idx = 1; neighbor_idx < valid_num + 1;
         ++neighbor_idx) {
      // __syncwarp();
      // assert(neighbor_idx < max_in_degree);
      assert(base_vector_id < base_num);

      id_type const neighbor_id =
          reverse_graph[base_vector_id * reverse_graph_degree + neighbor_idx];

      assert(neighbor_id < base_num);

      // Load neighbor vector
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        assert(neighbor_id < base_num);

        neighbor_vector_sdata[i] = base_data[neighbor_id * dim + i];
      }
      // __syncwarp();

      data_type sum = 0;
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        sum += base_vector_sdata[i] * neighbor_vector_sdata[i];
      }

      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      }
      // __syncwarp();

      if (lane_id == 0) {
        // assert(neighbor_idx < max_in_degree);
        distance_sdata[neighbor_idx] = -sum;
        neighbor_id_sdata[neighbor_idx] = neighbor_id;
      }
      __syncwarp();
    }

    for (auto i = valid_num + 1; i < reverse_graph_degree; i++) {
      distance_sdata[i] = FLT_MAX;
    }

    // __syncwarp();

    // FIXME(shiwen): check the 3rd template.
    warp_sort<data_type, id_type, reverse_graph_degree, lane_width>(
        distance_sdata, neighbor_id_sdata, true);

    // __syncwarp();

    for (uint32_t i = lane_id; i < valid_num; i += lane_width) {
      reverse_graph[base_vector_id * reverse_graph_degree + i] =
          neighbor_id_sdata[i];
    }
    if (lane_id == 0) {
      reverse_graph[base_vector_id * reverse_graph_degree +
                    reverse_graph_degree - 1] = valid_num;
    }
    __syncwarp();
  }
}

template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t graph_max_in_degree, uint32_t pruned_graph_max_in_degree,
          uint32_t tomb = 0xFFFFFFFF, uint32_t dim, bool is_strict = true,
          typename data_type = float, typename id_type = uint32_t>
__global__ void rng_prune_kernel(
    data_type const* __restrict__ base_data, id_type const* __restrict__ graph,
    data_type const* __restrict__ neighbor_distance,
    id_type* __restrict__ pruned_graph) {
  constexpr auto stride = block_size * grid_size;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;

  for (auto base_id = thread_idx; base_id < base_num; base_id += stride) {
    // insert base id of the first neighbor first.
    auto neighbor_idx = 0;
    auto pruned_graph_neighbor_idx = 0;
    assert(neighbor_idx < graph_max_in_degree);
    auto neighbor_base_id = graph[base_id * graph_max_in_degree + neighbor_idx];
    if (neighbor_base_id == tomb) {
      continue;
    }
    assert(pruned_graph_neighbor_idx < pruned_graph_max_in_degree);
    pruned_graph[base_id * pruned_graph_max_in_degree +
                 pruned_graph_neighbor_idx] = neighbor_base_id;
    neighbor_idx++;
    auto explore_flag = true;
    for (; (explore_flag && neighbor_idx < graph_max_in_degree);
         neighbor_idx++) {
      assert(neighbor_idx < graph_max_in_degree);
      neighbor_base_id = graph[base_id * graph_max_in_degree + neighbor_idx];
      if (neighbor_base_id == tomb) {
        break;
      }
      auto compare_idx = 0;
      for (; compare_idx <= pruned_graph_neighbor_idx; compare_idx++) {
        assert(compare_idx < pruned_graph_max_in_degree);
        auto compare_base_id =
            pruned_graph[base_id * pruned_graph_max_in_degree + compare_idx];
        assert(compare_base_id != tomb);
        // TODO(shiwen): FP16?
        data_type distance = 0;
        for (auto i = 0; i < dim; i++) {
          assert(i < dim);
          auto x_value = base_data[neighbor_base_id * dim + i];
          // TODO(shiwen): use shared memory.
          assert(i < dim);
          auto y_value = base_data[compare_base_id * dim + i];
          distance += x_value * y_value;
        }
        distance = -distance;
        assert(neighbor_idx < graph_max_in_degree);
        if (distance <
            neighbor_distance[base_id * graph_max_in_degree + neighbor_idx]) {
          break;
        }
      }
      // pass all the distance tests.
      if (compare_idx > pruned_graph_neighbor_idx) {
        pruned_graph_neighbor_idx++;
        // NOTE(shiwen):
        assert(pruned_graph_neighbor_idx < pruned_graph_max_in_degree);
        assert(base_id != neighbor_base_id);
        assert(pruned_graph_neighbor_idx < pruned_graph_max_in_degree);
        pruned_graph[base_id * pruned_graph_max_in_degree +
                     pruned_graph_neighbor_idx] = neighbor_base_id;
        if (pruned_graph_neighbor_idx == pruned_graph_max_in_degree - 1) {
          explore_flag = false;
          break;
        }
      }
    }
    // TODO(shiwen): slight RNG prune?
  }
}

// from_node_id -> to_node_id
template <typename id_type = uint32_t, uint32_t reverse_graph_degree,
          uint32_t reverse_graph_max_neighbor>
__device__ __forceinline__ void add_to_reverse_graph_thread_level(
    id_type const& from_node_id, id_type const& to_node_id,
    id_type* reverse_graph) {
  constexpr uint32_t start_idx = 1;
  auto prev_edge_num =
      atomicAdd(&reverse_graph[from_node_id * reverse_graph_degree], 1);
  auto insert_idx = start_idx + prev_edge_num;
  // naive cut off
  if (insert_idx < reverse_graph_max_neighbor) {
    reverse_graph[from_node_id * reverse_graph_degree + insert_idx] =
        to_node_id;
  } else {
    reverse_graph[from_node_id * reverse_graph_degree] =
        reverse_graph_max_neighbor;
  }
}

template <typename data_type = float, typename id_type = uint32_t, uint32_t dim,
          uint32_t base_num>
__device__ __forceinline__ data_type
distance_between(id_type const& x_id, id_type const& y_id,
                 data_type const* __restrict__ base_data) {
  data_type sum = 0;
  for (auto i = 0; i < dim; i++) {
    auto x_value = base_data[x_id * dim + i];
    auto y_value = base_data[y_id * dim + i];
    sum += x_value * y_value;
  }
  return -sum;
}

// NOTE(shiwen): init value: 1. the first element of each row of reverse graph
// is set to 0(init number of reverse edge number). 2.the first element of each
// row of prune graph is set to 0(init number of prune number)
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t origin_graph_degree, uint32_t prune_graph_degree,
          uint32_t reverse_graph_degree, uint32_t tomb = 0xFFFFFFFF,
          uint32_t dim, bool is_strict = true, typename data_type = float,
          typename id_type = uint32_t>
__global__ void rng_prune_and_add_reverse_kernel(
    data_type const* __restrict__ base_data, id_type const* __restrict__ graph,
    id_type* __restrict__ reverse_graph, id_type* __restrict__ pruned_graph) {
  constexpr auto reverse_graph_max_neighbor =
      reverse_graph_degree - prune_graph_degree;
  constexpr auto stride = block_size * grid_size;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;
  // choose this strategy: if we add one edge to the prune graph, we add the
  // reverse edge to the reverse graph immediately.
  for (auto base_id = thread_idx; base_id < base_num; base_id += stride) {
    // insert base id of the first neighbor first.
    auto neighbor_idx = 0;
    assert(neighbor_idx < origin_graph_degree);
    auto neighbor_base_id = graph[base_id * origin_graph_degree + neighbor_idx];
    if (neighbor_base_id == tomb) {
      continue;
    }
    // start at the second place because the first place of each row is set to
    // num of neighbor.
    auto pruned_graph_neighbor_idx = 1;
    assert(neighbor_base_id < base_num);
    assert(pruned_graph_neighbor_idx < reverse_graph_max_neighbor);
    pruned_graph[base_id * prune_graph_degree + pruned_graph_neighbor_idx] =
        neighbor_base_id;
    add_to_reverse_graph_thread_level<id_type, reverse_graph_degree,
                                      reverse_graph_max_neighbor>(
        neighbor_base_id, base_id, reverse_graph);
    neighbor_idx++;
    auto explore_flag = true;
    for (; (explore_flag && neighbor_idx < origin_graph_degree);
         neighbor_idx++) {
      assert(neighbor_idx < origin_graph_degree);
      neighbor_base_id = graph[base_id * origin_graph_degree + neighbor_idx];
      if (neighbor_base_id == tomb) {
        break;
      }
      auto compare_idx = 1;
      for (; compare_idx <= pruned_graph_neighbor_idx; compare_idx++) {
        assert(compare_idx < reverse_graph_max_neighbor);
        auto compare_base_id =
            pruned_graph[base_id * prune_graph_degree + compare_idx];
        assert(compare_base_id != tomb);
        data_type compare_distance =
            distance_between<float, uint32_t, dim, base_num>(
                neighbor_base_id, compare_base_id, base_data);
        data_type neighbor_distance =
            distance_between<float, uint32_t, dim, base_num>(
                neighbor_base_id, base_id, base_data);
        assert(neighbor_idx < origin_graph_degree);
        if (compare_distance < neighbor_distance) {
          break;
        }
      }
      // pass all the distance tests.
      if (compare_idx > pruned_graph_neighbor_idx) {
        pruned_graph_neighbor_idx++;
        // NOTE(shiwen):
        assert(pruned_graph_neighbor_idx < prune_graph_degree);
        assert(base_id != neighbor_base_id);
        pruned_graph[base_id * prune_graph_degree + pruned_graph_neighbor_idx] =
            neighbor_base_id;
        add_to_reverse_graph_thread_level<id_type, reverse_graph_degree,
                                          reverse_graph_max_neighbor>(
            neighbor_base_id, base_id, reverse_graph);
        if (pruned_graph_neighbor_idx == prune_graph_degree - 1) {
          explore_flag = false;
          break;
        }
      }
    }
    // set neighbor number.
    pruned_graph[base_id * prune_graph_degree] = pruned_graph_neighbor_idx;
    // TODO(shiwen): slight RNG prune?
  }
}

template <typename data_type = uint32_t>
__device__ __forceinline__ void swap(data_type& a, data_type& b) {
  data_type temp = a;
  a = b;
  b = temp;
}

template <typename id_type = uint32_t, typename data_type = float,
          uint32_t number>
__device__ __forceinline__ void bubble_sort(data_type K[], id_type V[]) {
  id_type i, j;
  for (i = 0; i < number - 1; i++) {
    for (j = 0; j < number - 1 - i; j++) {
      if (K[j] > K[j + 1]) {
        swap(K[j], K[j + 1]);
        swap(V[j], V[j + 1]);
      }
    }
  }
}

// NOTE(shiwen): init value: 1. the first element of each row of reverse graph
// is set to 0(init number of reverse edge number). 2.the first element of
// each row of prune graph is set to 0(init number of prune number)
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t origin_graph_degree, uint32_t prune_graph_degree,
          uint32_t reverse_graph_degree, uint32_t tomb = 0xFFFFFFFF,
          uint32_t dim, bool is_strict = true, typename data_type = float,
          typename id_type = uint32_t>
__global__ void __launch_bounds__(block_size, 8)
    fusion_sort_rng_prune_and_add_reverse_kernel(
        data_type const* __restrict__ base_data, id_type* __restrict__ graph,
        id_type* __restrict__ reverse_graph,
        id_type* __restrict__ pruned_graph) {
  constexpr auto reverse_graph_max_neighbor =
      reverse_graph_degree - prune_graph_degree;
  constexpr auto stride = block_size * grid_size;

  data_type distance_local_memory[origin_graph_degree];

  auto thread_idx = threadIdx.x + block_size * blockIdx.x;
  // choose this strategy: if we add one edge to the prune graph, we add the
  // reverse edge to the reverse graph immediately.
  for (auto base_id = thread_idx; base_id < base_num; base_id += stride) {
    // sort the neighbor list by distance
    for (auto neighbor_idx = 0; neighbor_idx < origin_graph_degree;
         neighbor_idx++) {
      auto neighbor_id = graph[base_id * origin_graph_degree + neighbor_idx];
      if (neighbor_id == tomb) {
        distance_local_memory[neighbor_idx] = FLT_MAX;
        continue;
      }
      assert(neighbor_id < base_num);
      distance_local_memory[neighbor_idx] =
          distance_between<float, uint32_t, dim, base_num>(base_id, neighbor_id,
                                                           base_data);
    }
    bubble_sort<id_type, data_type, origin_graph_degree>(
        &distance_local_memory[0], graph + base_id * origin_graph_degree);

    // insert base id of the first neighbor first.
    auto neighbor_idx = 0;
    assert(neighbor_idx < origin_graph_degree);
    auto neighbor_base_id = graph[base_id * origin_graph_degree + neighbor_idx];
    if (neighbor_base_id == tomb) {
      continue;
    }
    // start at the second place because the first place of each row is set to
    // num of neighbor.
    auto pruned_graph_neighbor_idx = 1;
    assert(neighbor_base_id < base_num);
    assert(pruned_graph_neighbor_idx < reverse_graph_max_neighbor);
    pruned_graph[base_id * prune_graph_degree + pruned_graph_neighbor_idx] =
        neighbor_base_id;
    add_to_reverse_graph_thread_level<id_type, reverse_graph_degree,
                                      reverse_graph_max_neighbor>(
        neighbor_base_id, base_id, reverse_graph);
    neighbor_idx++;
    auto explore_flag = true;
    for (; (explore_flag && neighbor_idx < origin_graph_degree);
         neighbor_idx++) {
      assert(neighbor_idx < origin_graph_degree);
      neighbor_base_id = graph[base_id * origin_graph_degree + neighbor_idx];
      if (neighbor_base_id == tomb) {
        break;
      }
      auto compare_idx = 1;
      for (; compare_idx <= pruned_graph_neighbor_idx; compare_idx++) {
        assert(compare_idx < reverse_graph_max_neighbor);
        auto compare_base_id =
            pruned_graph[base_id * prune_graph_degree + compare_idx];
        assert(compare_base_id != tomb);
        data_type compare_distance =
            distance_between<float, uint32_t, dim, base_num>(
                neighbor_base_id, compare_base_id, base_data);
        data_type neighbor_distance =
            distance_between<float, uint32_t, dim, base_num>(
                neighbor_base_id, base_id, base_data);
        assert(neighbor_idx < origin_graph_degree);
        if (compare_distance < neighbor_distance) {
          break;
        }
      }
      // pass all the distance tests.
      if (compare_idx > pruned_graph_neighbor_idx) {
        pruned_graph_neighbor_idx++;
        // NOTE(shiwen):
        assert(pruned_graph_neighbor_idx < prune_graph_degree);
        assert(base_id != neighbor_base_id);
        pruned_graph[base_id * prune_graph_degree + pruned_graph_neighbor_idx] =
            neighbor_base_id;
        add_to_reverse_graph_thread_level<id_type, reverse_graph_degree,
                                          reverse_graph_max_neighbor>(
            neighbor_base_id, base_id, reverse_graph);
        if (pruned_graph_neighbor_idx == prune_graph_degree - 1) {
          explore_flag = false;
          break;
        }
      }
    }
    // set neighbor number.
    pruned_graph[base_id * prune_graph_degree] = pruned_graph_neighbor_idx;
    // TODO(shiwen): slight RNG prune?
  }
}

template <typename id_type, uint32_t reverse_edge_num, uint32_t pruned_edge_num>
struct pr_neighbor_list {
  // NOTE(shiwen): reverse_num should be atomic fetch.
  id_type reverse_num;
  id_type reverse_list[reverse_edge_num];
  // 👆 cache line?
  id_type pruned_num;
  id_type pruned_list[pruned_edge_num];
  // 👆 cache line?
};

template <typename id_type, uint32_t reverse_edge_num, uint32_t pruned_edge_num>
__device__ __forceinline__ void add_reverse(
    id_type const& from_id, id_type const& to_id,
    pr_neighbor_list<id_type, reverse_edge_num,
                     pruned_edge_num>* __restrict__ pr_lists) {
  constexpr uint32_t pr_neighbor_list_size =
      sizeof(pr_neighbor_list<id_type, reverse_edge_num, pruned_edge_num>);
  // NOTE(shiwen): cache line allignment.
  static_assert(pr_neighbor_list_size % 128 == 0);
  auto prev_edge_num = atomicAdd(&(pr_lists[from_id].reverse_num), 1);
  auto insert_idx = prev_edge_num;
  // naive cut off
  if (insert_idx < reverse_edge_num) {
    pr_lists[from_id].reverse_list[insert_idx] = to_id;
  } else {
    // FIXME(shiwen): maybe useless ... ...
    pr_lists[from_id].reverse_num = reverse_edge_num;
  }
}

// NOTE(shiwen): must call set_prune_number later.
template <typename id_type, uint32_t reverse_edge_num, uint32_t pruned_edge_num>
__device__ __forceinline__ void add_pruned(
    id_type const& id, id_type const& idx,
    pr_neighbor_list<id_type, reverse_edge_num,
                     pruned_edge_num>* __restrict__ pr_list) {
  assert(idx < pruned_edge_num);
  pr_list->pruned_list[idx] = id;
}

template <typename id_type, uint32_t reverse_edge_num, uint32_t pruned_edge_num>
__device__ __forceinline__ void set_prune_number(
    id_type const& prune_number,
    pr_neighbor_list<id_type, reverse_edge_num,
                     pruned_edge_num>* __restrict__ pr_list) {
  assert(prune_number < pruned_edge_num);
  pr_list->pruned_num = prune_number;
}

// NOTE(shiwen): maybe bugs if struct is alligned...
template <typename id_type, uint32_t reset_idx0, uint32_t reset_idx1>
__device__ __forceinline__ void reset_pr_list(id_type* __restrict__ pr_list) {
  pr_list[reset_idx0] = 0;
  pr_list[reset_idx1] = 0;
}

// NOTE(shiwen): init value: 1. the first element of each row of reverse
// graph is set to 0(init number of reverse edge number).
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t origin_graph_degree, uint32_t pruned_edge_num,
          uint32_t reverse_edge_num, uint32_t tomb = 0xFFFFFFFF, uint32_t dim,
          bool is_strict = true, typename data_type = float,
          typename id_type = uint32_t>
// in fusion_sort_rng_prune_and_add_reverse_kernel, pruned graph is just one
// temp graph. we can use local memory(in
// fusion_sort_rng_prune_and_add_reverse_kernel, prune_graph size is
// base_num(10,000,000) x pruned_graph_degree. but in this v1 version, only
// allocate thread_num(144 x 256) x pruned_graph_degree local memory)
__global__ void /*__launch_bounds__(block_size, 8)*/
init_pr_lists(pr_neighbor_list<id_type, reverse_edge_num,
                               pruned_edge_num>* __restrict__ global_pr_lists) {
  constexpr auto stride = block_size * grid_size;

  auto thread_idx = threadIdx.x + block_size * blockIdx.x;
  // choose this strategy: if we add one edge to the prune graph, we add the
  // reverse edge to the reverse graph immediately.
  for (auto base_id = thread_idx; base_id < base_num; base_id += stride) {
    pr_neighbor_list<id_type, reverse_edge_num, pruned_edge_num>* pr_list =
        global_pr_lists + base_id;
    reset_pr_list<uint32_t, 0, 1 + reverse_edge_num>((id_type*)pr_list);
  }
}

// NOTE(shiwen): init value: 1. the first element of each row of reverse
// graph is set to 0(init number of reverse edge number).
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t origin_graph_degree, uint32_t pruned_edge_num,
          uint32_t reverse_edge_num, uint32_t tomb = 0xFFFFFFFF, uint32_t dim,
          bool is_strict = true, typename data_type = float,
          typename id_type = uint32_t>
// in fusion_sort_rng_prune_and_add_reverse_kernel, pruned graph is just one
// temp graph. we can use local memory(in
// fusion_sort_rng_prune_and_add_reverse_kernel, prune_graph size is
// base_num(10,000,000) x pruned_graph_degree. but in this v1 version, only
// allocate thread_num(144 x 256) x pruned_graph_degree local memory)
__global__ void /*__launch_bounds__(block_size, 8)*/
fusion_prune_reverse_kernel_v0(
    data_type const* __restrict__ base_data, id_type* __restrict__ graph,
    pr_neighbor_list<id_type, reverse_edge_num,
                     pruned_edge_num>* __restrict__ global_pr_lists) {
  constexpr auto stride = block_size * grid_size;

  data_type distance_local_memory[origin_graph_degree];

  auto thread_idx = threadIdx.x + block_size * blockIdx.x;
  // choose this strategy: if we add one edge to the prune graph, we add the
  // reverse edge to the reverse graph immediately.
  for (auto base_id = thread_idx; base_id < base_num; base_id += stride) {
    pr_neighbor_list<id_type, reverse_edge_num, pruned_edge_num>* pr_list =
        global_pr_lists + base_id;
    // sort the neighbor list by distance
    for (auto neighbor_idx = 0; neighbor_idx < origin_graph_degree;
         neighbor_idx++) {
      id_type neighbor_id = graph[base_id * origin_graph_degree + neighbor_idx];
      if (neighbor_id == tomb || neighbor_id == base_id ||
          neighbor_id == 2147483647) {
        distance_local_memory[neighbor_idx] = FLT_MAX;
        continue;
      }
      assert(neighbor_id != tomb);
      // assert(neighbor_id < base_num);
      distance_local_memory[neighbor_idx] =
          distance_between<float, uint32_t, dim, base_num>(base_id, neighbor_id,
                                                           base_data);
    }
    bubble_sort<id_type, data_type, origin_graph_degree>(
        &distance_local_memory[0], graph + base_id * origin_graph_degree);

    // insert base id of the first neighbor first.
    auto neighbor_idx = 0;
    assert(neighbor_idx < origin_graph_degree);
    auto neighbor_base_id = graph[base_id * origin_graph_degree + neighbor_idx];
    if (neighbor_base_id == tomb || neighbor_base_id == 2147483647) {
      continue;
    }
    // start at the second place because the first place of each row is set to
    // num of neighbor.
    auto pruned_graph_neighbor_idx = 0;
    assert(neighbor_base_id < base_num);
    // pr_list->pruned_list[pruned_graph_neighbor_idx] = neighbor_base_id;
    add_pruned<id_type, reverse_edge_num, pruned_edge_num>(
        neighbor_base_id, pruned_graph_neighbor_idx, pr_list);
    add_reverse<id_type, reverse_edge_num, pruned_edge_num>(
        neighbor_base_id, base_id, global_pr_lists);
    neighbor_idx++;
    auto explore_flag = true;
    for (; (explore_flag && neighbor_idx < origin_graph_degree);
         neighbor_idx++) {
      assert(neighbor_idx < origin_graph_degree);
      neighbor_base_id = graph[base_id * origin_graph_degree + neighbor_idx];
      if (neighbor_base_id == tomb || neighbor_base_id == 2147483647) {
        break;
      }
      auto compare_idx = 0;
      for (; compare_idx <= pruned_graph_neighbor_idx; compare_idx++) {
        auto compare_base_id = pr_list->pruned_list[compare_idx];
        assert(compare_base_id != tomb);
        assert(neighbor_base_id != tomb);
        assert(neighbor_base_id < base_num);
        assert(compare_base_id < base_num);
        data_type compare_distance =
            distance_between<float, uint32_t, dim, base_num>(
                neighbor_base_id, compare_base_id, base_data);
        // data_type neighbor_distance =
        //     distance_between<float, uint32_t, dim, base_num>(
        //         neighbor_base_id, base_id, base_data);
        data_type neighbor_distance = distance_local_memory[neighbor_idx];
        assert(neighbor_idx < origin_graph_degree);
        if (compare_distance < neighbor_distance) {
          break;
        }
      }
      // pass all the distance tests.
      if (compare_idx > pruned_graph_neighbor_idx) {
        pruned_graph_neighbor_idx++;
        // NOTE(shiwen):
        // assert(pruned_graph_neighbor_idx < prune_graph_degree);
        assert(base_id != neighbor_base_id);
        // pr_list->pruned_list[pruned_graph_neighbor_idx] = neighbor_base_id;
        add_pruned<id_type, reverse_edge_num, pruned_edge_num>(
            neighbor_base_id, pruned_graph_neighbor_idx, pr_list);
        add_reverse<id_type, reverse_edge_num, pruned_edge_num>(
            neighbor_base_id, base_id, global_pr_lists);
        if (pruned_graph_neighbor_idx == pruned_edge_num - 1) {
          explore_flag = false;
          break;
        }
      }
    }
    // TODO(shiwen): slight RNG prune?
    // 实现 slight RNG prune
    if (!is_strict && pruned_graph_neighbor_idx < pruned_edge_num - 1) {
      // 我们已经添加了 pruned_graph_neighbor_idx + 1 个邻居
      // 需要补充到 pruned_edge_num 个
      uint32_t start_neighbor_idx = neighbor_idx;  // 从当前位置继续
      
      // 继续遍历剩余的邻居，直到满足pruned_edge_num或没有更多邻居
      for (neighbor_idx = start_neighbor_idx; 
           neighbor_idx < origin_graph_degree && 
           pruned_graph_neighbor_idx < pruned_edge_num - 1; 
           neighbor_idx++) {
        
        neighbor_base_id = graph[base_id * origin_graph_degree + neighbor_idx];
        
        // 跳过无效的邻居
        if (neighbor_base_id == tomb || 
            neighbor_base_id == base_id || 
            neighbor_base_id == 2147483647) {
          continue;
        }
        
        // 检查这个邻居是否已经在pruned_list中
        bool already_added = false;
        for (uint32_t i = 0; i <= pruned_graph_neighbor_idx; i++) {
          if (pr_list->pruned_list[i] == neighbor_base_id) {
            already_added = true;
            break;
          }
        }
        
        // 如果这个邻居还没有添加过，添加它
        if (!already_added) {
          pruned_graph_neighbor_idx++;
          add_pruned<id_type, reverse_edge_num, pruned_edge_num>(
              neighbor_base_id, pruned_graph_neighbor_idx, pr_list);
          add_reverse<id_type, reverse_edge_num, pruned_edge_num>(
              neighbor_base_id, base_id, global_pr_lists);
        }
      }
    }

    set_prune_number<id_type, reverse_edge_num, pruned_edge_num>(
        pruned_graph_neighbor_idx + 1, pr_list);
  }
}

// NOTE(shiwen): set the first element of each row of prune graph to 0(init
// number of prune number)
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t prune_graph_degree, uint32_t reverse_graph_degree,
          uint32_t tomb = 0xFFFFFFFF, uint32_t dim, bool is_strict = true,
          typename data_type = float, typename id_type = uint32_t>
__global__ void merge_prune_graph_to_reverse_graph_kernel(
    data_type const* __restrict__ base_data,
    id_type* __restrict__ reverse_graph, id_type* __restrict__ pruned_graph) {
  constexpr auto stride = block_size * grid_size;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;
  // choose this strategy: if we add one edge to the prune graph, we add the
  // reverse edge to the reverse graph immediately.
  for (auto base_id = thread_idx; base_id < base_num; base_id += stride) {
    auto reverse_graph_neighbor_num =
        reverse_graph[base_id * reverse_graph_degree];
    auto prune_graph_neighbor_num = pruned_graph[base_id * prune_graph_degree];
    // NOTE(shiwen): for subsequent operations.
    reverse_graph[base_id * reverse_graph_degree] =
        reverse_graph_neighbor_num + prune_graph_neighbor_num;
    assert(reverse_graph_neighbor_num + prune_graph_neighbor_num <
           reverse_graph_degree);
    // NOTE(shiwen): for prune graph reuse.
    pruned_graph[base_id * prune_graph_degree] = 0;
    for (auto merge_idx = 0; merge_idx < prune_graph_neighbor_num;
         merge_idx++) {
      reverse_graph[base_id * reverse_graph_degree +
                    reverse_graph_neighbor_num + 1 + merge_idx] =
          pruned_graph[base_id * prune_graph_degree + 1 + merge_idx];
    }
  }
}

// NOTE(shiwen): set the first element of each row of prune graph to 0(init
// number of reverse number)
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t merge_graph_degree, uint32_t final_graph_degree,
          uint32_t tomb = 0xFFFFFFFF, uint32_t dim, bool is_strict = true,
          typename data_type = float, typename id_type = uint32_t>
__global__ void prune_merge_graph(
    data_type const* __restrict__ base_data, id_type* __restrict__ merge_graph,
    data_type const* __restrict__ neighbor_distance,
    id_type* __restrict__ final_graph) {
  constexpr auto stride = block_size * grid_size;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;

  for (auto base_id = thread_idx; base_id < base_num; base_id += stride) {
    auto neighbor_num =
        merge_graph[base_id * merge_graph_degree + merge_graph_degree - 1] - 1;
    if (neighbor_num == 0) {
      // merge_graph[base_id * merge_graph_degree] = 1;
      continue;
    }
    // insert base id of the first neighbor first.
    auto neighbor_idx = 1;
    auto neighbor_base_id =
        merge_graph[base_id * merge_graph_degree + neighbor_idx];
    // start at 0, final graph.
    auto final_graph_neighbor_idx = 0;
    assert(neighbor_base_id < base_num);
    assert(final_graph_neighbor_idx < final_graph_degree);
    final_graph[base_id * final_graph_degree + final_graph_neighbor_idx] =
        neighbor_base_id;
    neighbor_idx++;
    auto explore_flag = true;
    // FIXME(shiwen): this condition...
    for (; (explore_flag && neighbor_idx < neighbor_num + 1); neighbor_idx++) {
      assert(neighbor_idx < merge_graph_degree);
      neighbor_base_id =
          merge_graph[base_id * merge_graph_degree + neighbor_idx];
      // in this setting, neighbor_base_id should not be tomb.
      assert(neighbor_base_id < base_num);
      auto compare_idx = 0;
      for (; compare_idx <= final_graph_neighbor_idx; compare_idx++) {
        assert(compare_idx < final_graph_degree);
        auto compare_base_id =
            final_graph[base_id * final_graph_degree + compare_idx];
        assert(compare_base_id != tomb);
        data_type compare_distance =
            distance_between<float, uint32_t, dim, base_num>(
                neighbor_base_id, compare_base_id, base_data);
        data_type the_neighbor_distance =
            neighbor_distance[base_id * merge_graph_degree + neighbor_idx];
        if (compare_distance < neighbor_distance) {
          break;
        }
      }
      // pass all the distance tests.
      if (compare_idx > final_graph_neighbor_idx) {
        final_graph_neighbor_idx++;
        // NOTE(shiwen):
        assert(final_graph_neighbor_idx < final_graph_degree);
        assert(base_id != neighbor_base_id);
        final_graph[base_id * final_graph_degree + final_graph_neighbor_idx] =
            neighbor_base_id;
        if (final_graph_neighbor_idx == final_graph_degree - 1) {
          explore_flag = false;
          break;
        }
      }
    }
    // for  reuse of reverse graph.
    merge_graph[base_id * merge_graph_degree] = 1;
    // TODO(shiwen): slight RNG prune?
  }
}

// NOTE(shiwen): set the first element of each row of prune graph to 0(init
// number of reverse number)
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t merge_graph_degree, uint32_t final_graph_degree,
          uint32_t tomb = 0xFFFFFFFF, uint32_t dim, bool is_strict = true,
          typename data_type = float, typename id_type = uint32_t>
__global__ void prune_merge_graph_without_distance(
    data_type const* __restrict__ base_data, id_type* __restrict__ merge_graph,
    id_type* __restrict__ final_graph) {
  constexpr auto stride = block_size * grid_size;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;

  for (auto base_id = thread_idx; base_id < base_num; base_id += stride) {
    auto neighbor_num =
        merge_graph[base_id * merge_graph_degree + merge_graph_degree - 1];
    if (neighbor_num == 0) {
      // merge_graph[base_id * merge_graph_degree] = 1;
      for (auto i = 0; i < final_graph_degree; i++) {
        final_graph[base_id * final_graph_degree + i] = tomb;
      }
      continue;
    }
    // insert base id of the first neighbor first.
    auto neighbor_idx = 0;
    auto neighbor_base_id =
        merge_graph[base_id * merge_graph_degree + neighbor_idx];
    // start at 0, final graph.
    auto final_graph_neighbor_idx = 0;
    assert(neighbor_base_id < base_num);
    assert(final_graph_neighbor_idx < final_graph_degree);
    final_graph[base_id * final_graph_degree + final_graph_neighbor_idx] =
        neighbor_base_id;
    neighbor_idx++;
    auto explore_flag = true;
    // FIXME(shiwen): this condition...
    for (; (explore_flag && neighbor_idx < neighbor_num); neighbor_idx++) {
      assert(neighbor_idx < merge_graph_degree);
      neighbor_base_id =
          merge_graph[base_id * merge_graph_degree + neighbor_idx];
      // in this setting, neighbor_base_id should not be tomb.
      assert(neighbor_base_id < base_num);
      auto compare_idx = 0;
      for (; compare_idx <= final_graph_neighbor_idx; compare_idx++) {
        assert(compare_idx < final_graph_degree);
        auto compare_base_id =
            final_graph[base_id * final_graph_degree + compare_idx];
        assert(compare_base_id != tomb);
        data_type compare_distance =
            distance_between<float, uint32_t, dim, base_num>(
                neighbor_base_id, compare_base_id, base_data);
        data_type neighbor_distance =
            distance_between<float, uint32_t, dim, base_num>(
                neighbor_base_id, base_id, base_data);
        if (compare_distance < neighbor_distance) {
          break;
        }
      }
      // pass all the distance tests.
      if (compare_idx > final_graph_neighbor_idx) {
        final_graph_neighbor_idx++;
        // NOTE(shiwen):
        assert(final_graph_neighbor_idx < final_graph_degree);
        assert(base_id != neighbor_base_id);
        final_graph[base_id * final_graph_degree + final_graph_neighbor_idx] =
            neighbor_base_id;
        if (final_graph_neighbor_idx == final_graph_degree - 1) {
          explore_flag = false;
          break;
        }
      }
    }
    for (auto i = final_graph_neighbor_idx + 1; i < final_graph_degree; i++) {
      final_graph[base_id * final_graph_degree + i] = tomb;
    }
    // for  reuse of reverse graph.
    merge_graph[base_id * merge_graph_degree] = 0;
    // TODO(shiwen): slight RNG prune?
  }
}

template <typename id_type, uint32_t reverse_edge_num, uint32_t pruned_edge_num>
struct pr_sort_list {
  id_type list[2 + reverse_edge_num + pruned_edge_num];
};

// merge, sort.
template <typename id_type, typename data_type, uint32_t reverse_edge_num,
          uint32_t pruned_edge_num, uint32_t dim, uint32_t base_num>
__device__ __forceinline__ void merge_and_sort(
    pr_sort_list<id_type, reverse_edge_num, pruned_edge_num>* pr_list,
    id_type const& reverse_real_number, id_type const& prune_real_number,
    data_type* distance_local_memory, id_type const& base_id,
    data_type const* base_data) {
  distance_local_memory[0] = FLT_MAX;
  for (auto i = 1; i < 1 + reverse_real_number; i++) {
    distance_local_memory[i] = distance_between<float, uint32_t, dim, base_num>(
        base_id, pr_list->list[i], base_data);
  }
  for (auto i = 1 + reverse_real_number; i < 2 + reverse_edge_num; i++) {
    distance_local_memory[i] = FLT_MAX;
  }
  for (auto i = 2 + reverse_edge_num;
       i < 2 + reverse_edge_num + prune_real_number; i++) {
    distance_local_memory[i] = distance_between<float, uint32_t, dim, base_num>(
        base_id, pr_list->list[i], base_data);
  }
  for (auto i = 2 + reverse_edge_num + prune_real_number;
       i < 2 + reverse_edge_num + pruned_edge_num; i++) {
    distance_local_memory[i] = FLT_MAX;
  }
  bubble_sort<id_type, data_type, 2 + reverse_edge_num + pruned_edge_num>(
      distance_local_memory, (id_type*)(pr_list->list));
}

// NOTE(shiwen): reset the pr_lists after use...
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t pruned_edge_num, uint32_t reverse_edge_num,
          uint32_t final_graph_degree, uint32_t tomb = 0xFFFFFFFF, uint32_t dim,
          bool is_strict = true, typename data_type = float,
          typename id_type = uint32_t>
__global__ void fusion_merge_sort_prune_kernel(
    data_type const* __restrict__ base_data,
    pr_neighbor_list<id_type, reverse_edge_num,
                     pruned_edge_num>* __restrict__ global_pr_lists,
    id_type* __restrict__ final_graph) {
  constexpr auto stride = block_size * grid_size;
  using pr_neighbor_list_t =
      pr_neighbor_list<id_type, reverse_edge_num, pruned_edge_num>;
  using pr_sort_list_t =
      pr_sort_list<id_type, reverse_edge_num, pruned_edge_num>;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;

  data_type distance_local_memory[2 + reverse_edge_num + pruned_edge_num];

  for (auto base_id = thread_idx; base_id < base_num; base_id += stride) {
    pr_neighbor_list_t* pr_list = global_pr_lists + base_id;
    auto reverse_real_num = pr_list->reverse_num;
    auto prune_real_num = pr_list->pruned_num;
    // 👆 not in one cache line.
    auto neighbor_num = reverse_real_num + prune_real_num;
    if (neighbor_num == 0) {
      // merge_graph[base_id * merge_graph_degree] = 1;
      for (auto i = 0; i < final_graph_degree; i++) {
        final_graph[base_id * final_graph_degree + i] = tomb;
      }
      continue;
    }
    // force cast..
    pr_sort_list_t* pr_sort_list = reinterpret_cast<pr_sort_list_t*>(pr_list);
    merge_and_sort<id_type, data_type, reverse_edge_num, pruned_edge_num, dim,
                   base_num>(pr_sort_list, reverse_real_num, prune_real_num,
                             &distance_local_memory[0], base_id, base_data);

    // prune:
    auto neighbor_idx = 0;
    auto final_graph_neighbor_idx = 0;

    auto neighbor_base_id = pr_sort_list->list[neighbor_idx];
    assert(neighbor_base_id < base_num);
    assert(final_graph_neighbor_idx < final_graph_degree);
    final_graph[base_id * final_graph_degree + final_graph_neighbor_idx] =
        neighbor_base_id;
    neighbor_idx++;
    auto explore_flag = true;
    // FIXME(shiwen): this condition...
    for (; (explore_flag && neighbor_idx < neighbor_num); neighbor_idx++) {
      // assert(neighbor_idx < merge_graph_degree);
      neighbor_base_id = pr_sort_list->list[neighbor_idx];
      // in this setting, neighbor_base_id should not be tomb.
      assert(neighbor_base_id < base_num);
      auto compare_idx = 0;
      for (; compare_idx <= final_graph_neighbor_idx; compare_idx++) {
        assert(compare_idx < final_graph_degree);
        auto compare_base_id =
            final_graph[base_id * final_graph_degree + compare_idx];
        assert(compare_base_id != tomb);
        data_type compare_distance =
            distance_between<float, uint32_t, dim, base_num>(
                neighbor_base_id, compare_base_id, base_data);
        // data_type neighbor_distance =
        //     distance_between<float, uint32_t, dim, base_num>(
        //         neighbor_base_id, base_id, base_data);
        data_type neighbor_distance = distance_local_memory[neighbor_idx];
        if (compare_distance < neighbor_distance) {
          break;
        }
      }
      // pass all the distance tests.
      if (compare_idx > final_graph_neighbor_idx) {
        final_graph_neighbor_idx++;
        // NOTE(shiwen):
        assert(final_graph_neighbor_idx < final_graph_degree);
        assert(base_id != neighbor_base_id);
        final_graph[base_id * final_graph_degree + final_graph_neighbor_idx] =
            neighbor_base_id;
        if (final_graph_neighbor_idx == final_graph_degree - 1) {
          explore_flag = false;
          break;
        }
      }
    }
    for (auto i = final_graph_neighbor_idx + 1; i < final_graph_degree; i++) {
      final_graph[base_id * final_graph_degree + i] = tomb;
    }
    // TODO(shiwen): slight RNG prune?
        // 实现 slight RNG prune
    if (!is_strict && final_graph_neighbor_idx < final_graph_degree - 1) {
      // 遍历合并排序后的邻居列表，找出那些尚未添加但有效的邻居
      // 创建一个标记数组，标记已经添加到final_graph的邻居
      bool already_added[reverse_edge_num + pruned_edge_num];
      for (uint32_t i = 0; i < neighbor_num; i++) {
        already_added[i] = false;
      }
      
      // 标记已经添加的邻居
      for (uint32_t i = 0; i <= final_graph_neighbor_idx; i++) {
        auto added_id = final_graph[base_id * final_graph_degree + i];
        for (uint32_t j = 0; j < neighbor_num; j++) {
          if (pr_sort_list->list[j] == added_id) {
            already_added[j] = true;
            break;
          }
        }
      }
      
      // 按照排序顺序添加尚未添加的邻居，直到达到final_graph_degree
      for (uint32_t i = 0; i < neighbor_num && final_graph_neighbor_idx < final_graph_degree - 1; i++) {
        if (!already_added[i]) {
          neighbor_base_id = pr_sort_list->list[i];
          
          // 确保不添加无效的邻居或自身
          if (neighbor_base_id != tomb && 
              neighbor_base_id != base_id && 
              neighbor_base_id < base_num) {
            
            final_graph_neighbor_idx++;
            final_graph[base_id * final_graph_degree + final_graph_neighbor_idx] = 
                neighbor_base_id;
          }
        }
      }
    }
    
    // 将剩余位置填充为tomb值
    for (auto i = final_graph_neighbor_idx + 1; i < final_graph_degree; i++) {
      final_graph[base_id * final_graph_degree + i] = tomb;
    }
  }
}
}  // namespace Gpu
}  // namespace Gbuilder
