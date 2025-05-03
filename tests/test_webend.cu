#include "Gbuilder.cuh"
#include "Gsearcher.cuh"
#include "spdlog/spdlog.h"
#include "utils.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <immintrin.h>
#include <omp.h>
#include <sys/types.h>
#include <x86intrin.h>

namespace Gbuilder {
namespace Gpu {

float compare(float const* a, float const* b, uint32_t size) {
  __m512 msum0 = _mm512_setzero_ps();

  while (size >= 16) {
    __m512 mx = _mm512_loadu_ps(a);
    __m512 my = _mm512_loadu_ps(b);
    a += 16;
    b += 16;
    msum0 = _mm512_fmadd_ps(mx, my, msum0);  // fma: mx * my + msum0
    size -= 16;
  }

  __m256 msum1 =
      _mm512_extractf32x8_ps(msum0, 1) + _mm512_extractf32x8_ps(msum0, 0);

  if (size >= 8) {
    __m256 mx = _mm256_loadu_ps(a);
    __m256 my = _mm256_loadu_ps(b);
    a += 8;
    b += 8;
    msum1 = _mm256_fmadd_ps(mx, my, msum1);
    size -= 8;
  }

  __m128 msum2 =
      _mm256_extractf128_ps(msum1, 1) + _mm256_extractf128_ps(msum1, 0);

  if (size >= 4) {
    __m128 mx = _mm_loadu_ps(a);
    __m128 my = _mm_loadu_ps(b);
    a += 4;
    b += 4;
    msum2 = _mm_fmadd_ps(mx, my, msum2);
    size -= 4;
  }

  if (size > 0) {
    __m128i mask = _mm_set_epi32(size > 2 ? -1 : 0, size > 1 ? -1 : 0,
                                 size > 0 ? -1 : 0, 0);
    __m128 mx = _mm_maskload_ps(a, mask);
    __m128 my = _mm_maskload_ps(b, mask);
    msum2 = _mm_fmadd_ps(mx, my, msum2);
  }

  msum2 = _mm_hadd_ps(msum2, msum2);
  msum2 = _mm_hadd_ps(msum2, msum2);
  return -1.0f * _mm_cvtss_f32(msum2);
}

struct SimpleNeighbor {
  uint32_t id;
  float distance;
  SimpleNeighbor() = default;
  SimpleNeighbor(uint32_t id, float distance) : id{id}, distance{distance} {}
  inline bool operator<(SimpleNeighbor const& other) const {
    return distance < other.distance ||
           (distance == other.distance && id < other.id);
  }
  inline bool operator>(SimpleNeighbor const& other) const {
    return distance > other.distance ||
           (distance == other.distance && id > other.id);
  }
  friend void swap(SimpleNeighbor& a, SimpleNeighbor& b) {
    std::swap(a.id, b.id);
    std::swap(a.distance, b.distance);
  }
};

void RNGPrune(uint32_t M, std::vector<SimpleNeighbor>& full_set,
              uint32_t target_id, std::vector<uint32_t>& pruned_list,
              float const* data, bool is_strict, uint32_t num_base,
              uint32_t const dimension) {
  uint32_t M_ctr = M;
  uint32_t start = 0;

  while (start < full_set.size() && full_set[start].id == target_id) start++;
  if (start == full_set.size()) {
    return;
  }

  std::vector<uint32_t> result;
  result.reserve(M_ctr);
  result.emplace_back(full_set[start].id);

  while (result.size() < M_ctr && (++start) < full_set.size()) {
    auto& p = full_set[start];
    bool occlude = false;
    for (size_t i = 0; i < result.size(); ++i) {
      if (p.id == result[i]) {
        occlude = true;
        break;
      }
      float djk = compare(data + dimension * p.id, data + dimension * result[i],
                          dimension);

      if (djk < p.distance) {
        occlude = true;
        break;
      }
    }
    if (!occlude) {
      if (p.id != target_id &&
          std::find(result.begin(), result.end(), p.id) == result.end()) {
        result.emplace_back(p.id);
      }
    }
  }

  start = 0;  // double check
  while (start < full_set.size() && full_set[start].id == target_id) start++;
  while (result.size() < M_ctr && (++start) < full_set.size()) {
    auto& p = full_set[start];
    bool occlude = false;
    for (size_t i = 0; i < result.size(); ++i) {
      if (p.id == result[i]) {
        occlude = true;
        break;
      }
      float djk = compare(data + dimension * p.id, data + dimension * result[i],
                          dimension);
      if (djk < p.distance) {
        occlude = true;
        break;
      }
    }
    if (!occlude) {
      if (p.id != target_id &&
          std::find(result.begin(), result.end(), p.id) == result.end()) {
        result.emplace_back(p.id);
      }
    }
  }

  if (!is_strict) {
    for (size_t i = 1; i < full_set.size() && result.size() < M_ctr; ++i) {
      if (std::find(result.begin(), result.end(), full_set[i].id) ==
              result.end() &&
          full_set[i].id < num_base) {
        if (full_set[i].id != target_id) {
          result.emplace_back(full_set[i].id);
        }
      }
    }
  }

  pruned_list = result;
}

std::vector<std::vector<uint32_t>> MatchNN(
    uint32_t num_base, uint32_t num_query, uint32_t max_degree, uint32_t N_ctr,
    uint32_t M_nn, uint32_t const* query_knn, uint32_t& ep, float const* data,
    uint32_t const dimension, int thread_limit) {
  int original_threads = omp_get_max_threads();
  omp_set_num_threads(thread_limit);

  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<uint32_t> match(num_query, num_base + 1);
  std::vector<uint32_t> frequency(num_base, 0);
  std::vector<bool> vis(num_base, false);

  for (uint32_t it_nq = 0; it_nq < num_query; ++it_nq) {
    uint32_t base = num_base + 1;
    bool ifmatch = false;
    for (uint32_t j = 0; j < N_ctr; j++) {
      uint32_t nn = query_knn[it_nq * N_ctr + j];
      if (nn >= num_base || ifmatch) break;
      ++frequency[nn];
      if (vis[nn]) {
        continue;
      } else {
        vis[nn] = true;
        ifmatch = true;
        base = nn;
      }
    }
    match[it_nq] = base;
  }

  auto after_match = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> match_duration = after_match - start_time;
  SPDLOG_INFO("MatchNN: Initial matching completed in {:.2f} seconds",
              match_duration.count());

  std::vector<uint32_t> order(num_base);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](uint32_t i, uint32_t j) {
    return frequency[i] > frequency[j];
  });
  assert(order[0] < num_base);
  ep = order[0];

  uint32_t const MAX_DEGREE = 128 - M_nn;
  std::vector<std::vector<uint32_t>> match_graph(num_base);
  std::vector<std::vector<uint32_t>> tmp_graph(
      num_base, std::vector<uint32_t>(MAX_DEGREE, UINT32_MAX));
  std::vector<std::atomic<uint32_t>> degrees(num_base);
  std::vector<bool> is_full(num_base);

  //   auto before_parallel = std::chrono::high_resolution_clock::now();

  // #pragma omp parallel for schedule(dynamic, 200)
  //   for (uint32_t it_nq = 0; it_nq < num_query; ++it_nq) {
  //     uint32_t cur_main_id = match[it_nq];
  //     if (cur_main_id >= num_base) {
  //       continue;
  //     }
  //     std::set<uint32_t> vis;
  //     std::vector<SimpleNeighbor> full_set;
  //     vis.insert(cur_main_id);
  //     for (uint32_t j = 0; j < N_ctr; j++) {
  //       uint32_t base_id = query_knn[it_nq * N_ctr + j];
  //       if (base_id >= num_base) break;
  //       if (vis.find(base_id) != vis.end()) continue;
  //       vis.insert(base_id);
  //       float distance = compare(data + dimension * base_id,
  //                                data + dimension * cur_main_id, dimension);
  //       full_set.emplace_back(SimpleNeighbor(base_id, distance));
  //     }
  //     std::sort(full_set.begin(), full_set.end());
  //     std::vector<uint32_t> pruned_list;
  //     RNGPrune(M_nn, full_set, cur_main_id, pruned_list, data, false,
  //     num_base,
  //              dimension);
  //     for (uint32_t des_node : pruned_list) {
  //       if (is_full[des_node]) continue;
  //       uint32_t cur_degree =
  //           degrees[des_node].fetch_add(1, std::memory_order_relaxed);
  //       if (cur_degree < MAX_DEGREE) {
  //         tmp_graph[des_node][cur_degree] = cur_main_id;
  //       } else if (cur_degree == MAX_DEGREE) {
  //         is_full[des_node] = true;
  //       }
  //     }
  //     match_graph[cur_main_id] = pruned_list;
  //   }

  //   auto after_parallel = std::chrono::high_resolution_clock::now();
  //   std::chrono::duration<double> parallel_duration =
  //       after_parallel - before_parallel;
  //   SPDLOG_INFO("MatchNN: Initial graph building completed in {:.2f}
  //   seconds",
  //               parallel_duration.count());

  //   auto before_final_prune = std::chrono::high_resolution_clock::now();

  // #pragma omp parallel for schedule(dynamic, 200)
  //   for (uint32_t it_nb = 0; it_nb < num_base; ++it_nb) {
  //     std::vector<uint32_t> const& vec1 = match_graph[it_nb];
  //     std::vector<uint32_t> const& vec2 = tmp_graph[it_nb];

  //     uint32_t actual_size = degrees[it_nb].load(std::memory_order_relaxed);
  //     actual_size = std::min(actual_size, MAX_DEGREE);
  //     std::unordered_set<uint32_t> mergedSet;
  //     mergedSet.reserve(vec1.size() + actual_size);
  //     mergedSet.insert(vec1.begin(), vec1.end());
  //     mergedSet.insert(vec2.begin(), vec2.begin() + actual_size);

  //     match_graph[it_nb] =
  //         std::vector<uint32_t>(mergedSet.begin(), mergedSet.end());
  //     if (match_graph[it_nb].size() > M_nn) {
  //       std::vector<SimpleNeighbor> full_set;
  //       for (uint32_t& base_id : match_graph[it_nb]) {
  //         float distance = compare(data + dimension * base_id,
  //                                  data + dimension * it_nb, dimension);

  //         full_set.emplace_back(SimpleNeighbor(base_id, distance));
  //       }
  //       std::sort(full_set.begin(), full_set.end());
  //       std::vector<uint32_t> pruned_list;
  //       RNGPrune(M_nn, full_set, it_nb, pruned_list, data, true, num_base,
  //                dimension);
  //       match_graph[it_nb] = std::move(pruned_list);
  //     }
  //   }

  //   auto end_time = std::chrono::high_resolution_clock::now();
  //   std::chrono::duration<double> final_duration = end_time -
  //   before_final_prune; SPDLOG_INFO("MatchNN: Final pruning completed in
  //   {:.2f} seconds",
  //               final_duration.count());

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_duration = end_time - start_time;
  SPDLOG_INFO("MatchNN: Total execution time: {:.2f} seconds",
              total_duration.count());

  omp_set_num_threads(original_threads);

  return match_graph;
}

void statDegree(uint32_t num_base,
                std::vector<std::vector<uint32_t>>& fusionNN_graph) {
  size_t total_edges = 0;
  size_t min_degree = std::numeric_limits<size_t>::max();
  size_t max_degrees = 0;
  double avg_degree = 0.0;
  for (auto const& neighbors : fusionNN_graph) {
    total_edges += neighbors.size();
    min_degree = std::min(min_degree, neighbors.size());
    max_degrees = std::max(max_degrees, neighbors.size());
  }
  avg_degree = static_cast<double>(total_edges) / num_base;
  SPDLOG_INFO(
      "MatchNN: Graph statistics - Edges: {}, Avg degree: {:.2f}, Min degree: "
      "{}, Max degree: {}",
      total_edges, avg_degree, min_degree, max_degrees);
}

void statDegree(uint32_t num_base, uint32_t degree,
                std::vector<uint32_t>& fusionNN_graph) {
  size_t total_edges = 0;
  size_t min_degree = std::numeric_limits<size_t>::max();
  size_t max_degrees = 0;
  double avg_degree = 0.0;
  for (uint32_t i = 0; i < num_base; i++) {
    size_t size = 0;
    for (uint32_t j = 0; j < degree; j++) {
      if (fusionNN_graph[i * degree + j] < num_base) {
        size++;
      }
    }
    total_edges += size;
    min_degree = std::min(min_degree, size);
    max_degrees = std::max(max_degrees, size);
  }
  avg_degree = static_cast<double>(total_edges) / num_base;
  SPDLOG_INFO(
      "MatchNN: Graph statistics - Edges: {}, Avg degree: {:.2f}, Min degree: "
      "{}, Max degree: {}",
      total_edges, avg_degree, min_degree, max_degrees);
}

std::vector<std::vector<uint32_t>> MatchSup(
    uint32_t num_base, uint32_t num_query, uint32_t topn, uint32_t N_ctr,
    uint32_t M_nn, uint32_t const* query_knn, float const* data,
    uint32_t const dimension, int thread_limit) {
  int original_threads = omp_get_max_threads();
  omp_set_num_threads(thread_limit);

  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<uint32_t> match(num_query, num_base + 1);
  std::vector<bool> vis(num_base, false);

  for (uint32_t it_nq = 0; it_nq < num_query; ++it_nq) {
    uint32_t base = num_base + 1;
    bool ifmatch = false;
    for (uint32_t j = topn; j < topn + 1; j++) {
      uint32_t nn = query_knn[it_nq * N_ctr + j];
      if (nn >= num_base || ifmatch) break;
      if (vis[nn]) {
        continue;
      } else {
        vis[nn] = true;
        ifmatch = true;
        base = nn;
      }
    }
    match[it_nq] = base;
  }

  auto after_match = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> match_duration = after_match - start_time;
  SPDLOG_INFO("MatchNN: Initial matching completed in {:.2f} seconds",
              match_duration.count());

  uint32_t const MAX_DEGREE = 128 - M_nn;
  std::vector<std::vector<uint32_t>> match_graph(num_base);
  std::vector<std::vector<uint32_t>> tmp_graph(
      num_base, std::vector<uint32_t>(MAX_DEGREE, UINT32_MAX));
  std::vector<std::atomic<uint32_t>> degrees(num_base);
  std::vector<bool> is_full(num_base);

  auto before_parallel = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 200)
  for (uint32_t it_nq = 0; it_nq < num_query; ++it_nq) {
    uint32_t cur_main_id = match[it_nq];
    if (cur_main_id >= num_base) {
      continue;
    }
    std::set<uint32_t> vis;
    std::vector<SimpleNeighbor> full_set;
    vis.insert(cur_main_id);
    for (uint32_t j = 0; j < N_ctr; j++) {
      uint32_t base_id = query_knn[it_nq * N_ctr + j];
      if (base_id >= num_base) break;
      if (vis.find(base_id) != vis.end()) continue;
      vis.insert(base_id);
      float distance = compare(data + dimension * base_id,
                               data + dimension * cur_main_id, dimension);
      full_set.emplace_back(SimpleNeighbor(base_id, distance));
    }
    std::sort(full_set.begin(), full_set.end());
    std::vector<uint32_t> pruned_list;
    RNGPrune(M_nn, full_set, cur_main_id, pruned_list, data, false, num_base,
             dimension);
    for (uint32_t des_node : pruned_list) {
      if (is_full[des_node]) continue;
      uint32_t cur_degree =
          degrees[des_node].fetch_add(1, std::memory_order_relaxed);
      if (cur_degree < MAX_DEGREE) {
        tmp_graph[des_node][cur_degree] = cur_main_id;
      } else if (cur_degree == MAX_DEGREE) {
        is_full[des_node] = true;
      }
    }
    match_graph[cur_main_id] = pruned_list;
  }

  auto after_parallel = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> parallel_duration =
      after_parallel - before_parallel;
  SPDLOG_INFO("MatchNN: Initial graph building completed in {:.2f} seconds",
              parallel_duration.count());

  auto before_final_prune = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 200)
  for (uint32_t it_nb = 0; it_nb < num_base; ++it_nb) {
    std::vector<uint32_t> const& vec1 = match_graph[it_nb];
    std::vector<uint32_t> const& vec2 = tmp_graph[it_nb];
    uint32_t actual_size = degrees[it_nb].load(std::memory_order_relaxed);
    actual_size = std::min(actual_size, MAX_DEGREE);
    std::unordered_set<uint32_t> mergedSet;
    mergedSet.reserve(vec1.size() + actual_size);
    mergedSet.insert(vec1.begin(), vec1.end());
    mergedSet.insert(vec2.begin(), vec2.begin() + actual_size);
    match_graph[it_nb] =
        std::vector<uint32_t>(mergedSet.begin(), mergedSet.end());
    if (match_graph[it_nb].size() > M_nn) {
      std::vector<SimpleNeighbor> full_set;
      for (uint32_t& base_id : match_graph[it_nb]) {
        float distance = compare(data + dimension * base_id,
                                 data + dimension * it_nb, dimension);
        full_set.emplace_back(SimpleNeighbor(base_id, distance));
      }
      std::sort(full_set.begin(), full_set.end());
      std::vector<uint32_t> pruned_list;
      RNGPrune(M_nn, full_set, it_nb, pruned_list, data, true, num_base,
               dimension);
      match_graph[it_nb] = std::move(pruned_list);
    }
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> final_duration = end_time - before_final_prune;
  std::chrono::duration<double> total_duration = end_time - start_time;

  SPDLOG_INFO("Match {}: Final pruning completed in {:.2f} seconds", topn,
              final_duration.count());
  SPDLOG_INFO("Match {}: Total execution time: {:.2f} seconds", topn,
              total_duration.count());

  omp_set_num_threads(original_threads);

  return match_graph;
}

std::vector<std::vector<uint32_t>> FusionNN(
    uint32_t num_base, uint32_t M_nn, float const* data,
    std::vector<std::vector<uint32_t>>& topNN_graph,
    std::vector<std::vector<std::vector<uint32_t>>>& supply_graphs,
    uint32_t const dimension, int thread_limit) {
  int original_threads = omp_get_max_threads();
  omp_set_num_threads(thread_limit);
  std::vector<std::vector<uint32_t>> fusionNN_graph(num_base);
  auto start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 100)
  for (uint32_t it_nb = 0; it_nb < num_base; ++it_nb) {
    std::vector<uint32_t> final_set;
    std::vector<SimpleNeighbor> full_set;
    std::set<uint32_t> vis;
    full_set.reserve(M_nn * 3);
    for (uint32_t& base_id : topNN_graph[it_nb]) {
      if (vis.find(base_id) != vis.end() || base_id == it_nb ||
          base_id >= num_base)
        continue;
      vis.insert(base_id);
      float distance = compare(data + dimension * it_nb,
                               data + dimension * base_id, dimension);
      full_set.push_back(SimpleNeighbor(base_id, distance));
    }
    for (auto& graph : supply_graphs) {
      for (uint32_t& base_id : graph[it_nb]) {
        if (vis.find(base_id) != vis.end() || base_id == it_nb ||
            base_id >= num_base)
          continue;
        vis.insert(base_id);
        float distance = compare(data + dimension * it_nb,
                                 data + dimension * base_id, dimension);
        full_set.push_back(SimpleNeighbor(base_id, distance));
      }
    }
    std::sort(full_set.begin(), full_set.end());
    std::vector<uint32_t> pruned_list;
    RNGPrune(M_nn * 2, full_set, it_nb, pruned_list, data, true, num_base,
             dimension);

    fusionNN_graph[it_nb] = pruned_list;
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_duration = end_time - start_time;
  SPDLOG_INFO("FusionNN: Completed in {:.2f} seconds", total_duration.count());
  omp_set_num_threads(original_threads);
  return fusionNN_graph;
}

std::vector<std::vector<uint32_t>> FusionFinal(
    uint32_t num_base, uint32_t M_supply, uint32_t M_link, uint32_t M_final,
    float const* data, std::vector<uint32_t>& supply_graph_,
    std::vector<std::vector<uint32_t>>& bipartite_graph_,
    std::vector<uint32_t>& link_graph_, uint32_t const dimension,
    int thread_limit) {
  int original_threads = omp_get_max_threads();
  omp_set_num_threads(thread_limit);

  auto start_time = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<uint32_t>> final_graph_(num_base);
#pragma omp parallel for schedule(dynamic, 100)
  for (uint32_t it_nb = 0; it_nb < num_base; ++it_nb) {
    std::vector<uint32_t> final_set;
    std::vector<SimpleNeighbor> full_set;
    std::set<uint32_t> vis;
    final_set.reserve(M_final);
    final_graph_[it_nb].reserve(M_final);
    full_set.reserve(M_final + 80);
    for (uint32_t& base_id : bipartite_graph_[it_nb]) {
      if (vis.find(base_id) != vis.end() || base_id == it_nb ||
          base_id >= num_base)
        continue;
      vis.insert(base_id);
      float distance = compare(data + dimension * it_nb,
                               data + dimension * base_id, dimension);
      full_set.push_back(SimpleNeighbor(base_id, distance));
      final_graph_[it_nb].push_back(base_id);
    }
    for (uint32_t j = 0; j < M_link; j++) {
      uint32_t base_id = link_graph_[it_nb * M_link + j];
      if (base_id >= num_base) continue;
      if (vis.find(base_id) != vis.end()) continue;
      vis.insert(base_id);
      float distance = compare(data + dimension * it_nb,
                               data + dimension * base_id, dimension);
      full_set.push_back(SimpleNeighbor(base_id, distance));
    }
    for (uint32_t j = 0; j < M_supply; j++) {
      uint32_t base_id = supply_graph_[it_nb * M_supply + j];
      if (base_id >= num_base) continue;
      if (vis.find(base_id) != vis.end() || base_id == it_nb ||
          base_id >= num_base)
        continue;
      vis.insert(base_id);
      float distance = compare(data + dimension * it_nb,
                               data + dimension * base_id, dimension);
      full_set.push_back(SimpleNeighbor(base_id, distance));
    }
    std::sort(full_set.begin(), full_set.end());
    std::vector<uint32_t> pruned_list;
    RNGPrune(M_final, full_set, it_nb, pruned_list, data, true, num_base,
             dimension);
    std::unordered_set<uint32_t> final_set_lookup(final_graph_[it_nb].begin(),
                                                  final_graph_[it_nb].end());
    std::vector<uint32_t> ok_insert;
    ok_insert.reserve(M_final);
    size_t const remaining_slots = M_final - final_graph_[it_nb].size();
    for (uint32_t candidate : pruned_list) {
      if (ok_insert.size() >= remaining_slots) break;
      if (final_set_lookup.find(candidate) == final_set_lookup.end()) {
        ok_insert.push_back(candidate);
      }
    }
    final_graph_[it_nb].insert(final_graph_[it_nb].end(), ok_insert.begin(),
                               ok_insert.end());
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_duration = end_time - start_time;
  SPDLOG_INFO("FusionFinal: Completed in {:.2f} seconds",
              total_duration.count());
  omp_set_num_threads(original_threads);

  return final_graph_;
}

void SaveGraph(std::vector<std::vector<uint32_t>> const& graph,
               std::string const& file_path, uint32_t ep, uint32_t base_num) {
  auto start_time = std::chrono::high_resolution_clock::now();
  std::ofstream out(file_path, std::ios::binary | std::ios::out);
  if (!out.is_open()) {
    throw std::runtime_error("cannot open file");
  }
  out.write((char*)&ep, sizeof(uint32_t));
  out.write((char*)&base_num, sizeof(uint32_t));
  for (uint32_t i = 0; i < base_num; ++i) {
    uint32_t nbr_size = graph[i].size();
    out.write((char*)&nbr_size, sizeof(uint32_t));
    out.write((char*)graph[i].data(), sizeof(uint32_t) * nbr_size);
  }
  out.close();
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_duration = end_time - start_time;
  SPDLOG_INFO("SaveGraph: Completed in {:.2f} seconds", total_duration.count());
}

TEST(GpuConstructionTime, TestEnd2EndCPUGPU_Webvid_2dot5M_512) {
  // 获取环境变量来控制供应图轮数和保存路径
  int match_rounds = 10;  // 默认值：6轮匹配
  std::string save_path_base = "../../indices/webvid-2.5M/webvid-gpu-std-r";

  char const* rounds_env = std::getenv("MATCH_ROUNDS");
  if (rounds_env) {
    try {
      match_rounds = std::stoi(rounds_env);
      SPDLOG_INFO("Using MATCH_ROUNDS from environment: {}", match_rounds);
    } catch (std::exception& e) {
      SPDLOG_WARN("Failed to parse MATCH_ROUNDS env var, using default: {}",
                  match_rounds);
    }
  }

  char const* save_path_env = std::getenv("SAVE_PATH_BASE");
  if (save_path_env) {
    save_path_base = save_path_env;
    SPDLOG_INFO("Using SAVE_PATH_BASE from environment: {}", save_path_base);
  }

  std::string save_path = save_path_base + std::to_string(match_rounds);
  SPDLOG_INFO("Graph will be saved to: {}", save_path);

  auto test_start_time = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Starting End-to-End CPU-GPU Webvid 2.5M test");

  cudaDeviceReset();
  auto cuda_reset_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cuda_reset_duration =
      cuda_reset_time - test_start_time;
  SPDLOG_INFO("CUDA device reset completed in {:.2f} seconds",
              cuda_reset_duration.count());

  // 配置参数
  constexpr uint32_t gt_degree = 128;
  constexpr uint32_t match_degree = 128;
  constexpr uint32_t base_num = 2505000;
  constexpr uint32_t dim = 512;
  constexpr uint32_t tomb = 0XFFFFFFFF;
  constexpr uint32_t reverse_edge_num = 111;
  constexpr uint32_t pruned_edge_num = 15;
  constexpr uint32_t top1_projection_degree = 16;
  constexpr uint32_t num_element_pr_list =
      reverse_edge_num + pruned_edge_num + 2;
  uint32_t ep = 0;  // ep变量

  // search相关参数
  constexpr uint32_t query_num = base_num;

  constexpr uint32_t first_round_search_grid_size = 144;
  constexpr uint32_t first_round_search_block_size = 512;
  constexpr uint32_t Km = 32;
  constexpr uint32_t Kp = 2;
  constexpr uint32_t Kd = 16;
  constexpr uint32_t topk = Km + Kp * Kd;
  constexpr uint32_t reset_iter = 15;
  constexpr uint32_t hashtable_size = 1 << 12;
  constexpr uint32_t shared_memory_size =
      (first_round_search_block_size / 32) *
      sizeof(
          search_warp_state_store_base_data<uint32_t, float, dim, Km, Kp, Kd>);

  constexpr uint32_t global_warp_num =
      first_round_search_grid_size * first_round_search_block_size / 32;

  constexpr uint32_t first_round_pruned_edge_num = 55;
  constexpr uint32_t first_round_reverse_edge_num = 71;

  constexpr uint32_t second_round_pruned_edge_num = 55;
  constexpr uint32_t second_round_reverse_edge_num = 71;

  constexpr uint32_t second_search_query_num = base_num;

  constexpr uint32_t second_round_search_grid_size = 144;
  constexpr uint32_t second_round_search_block_size = 512;
  constexpr uint32_t second_Km = 32;
  constexpr uint32_t second_Kp = 2;
  constexpr uint32_t second_Kd = 16;
  constexpr uint32_t second_topk = Km + Kp * Kd;
  constexpr uint32_t second_reset_iter = 15;
  constexpr uint32_t second_hashtable_size = 1 << 12;
  constexpr uint32_t second_shared_memory_size =
      (second_round_search_block_size / 32) *
      sizeof(search_warp_state_store_base_data<uint32_t, float, dim, second_Km,
                                               second_Kp, second_Kd>);

  constexpr uint32_t second_global_warp_num =
      second_round_search_grid_size * second_round_search_block_size / 32;

  constexpr uint32_t final_degree = 55;

  auto param_setup_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> param_setup_duration =
      param_setup_time - cuda_reset_time;
  SPDLOG_INFO("Parameter setup completed in {:.2f} seconds",
              param_setup_duration.count());

  auto basedata_file_name = "./data/webvid-2505000-512/web_2505000_512.fbin";
  auto gt_file = "./data/webvid-2505000-512/gt_2505000_128.ibin";

  // CPU线程数限制
  int max_threads = omp_get_max_threads();
  int cpu_thread_limit = std::min(max_threads, 64);  // 限制最大线程数为64
  SPDLOG_INFO("Using {} CPU threads", cpu_thread_limit);

  // 读取数据
  auto data_load_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Starting data loading from files");

  auto h_gt_data = std::vector<uint32_t>(2500000 * gt_degree);
  auto h_base_data = std::vector<float>(base_num * dim);

  read_vec_from_file(h_gt_data, gt_file);
  auto gt_load_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gt_load_duration =
      gt_load_time - data_load_start;
  SPDLOG_INFO("GT data loaded in {:.2f} seconds", gt_load_duration.count());

  read_vec_from_file(h_base_data, basedata_file_name);
  auto base_load_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> base_load_duration =
      base_load_time - gt_load_time;
  SPDLOG_INFO("Base data loaded in {:.2f} seconds", base_load_duration.count());

  auto data_load_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> data_load_duration =
      data_load_end - data_load_start;
  SPDLOG_INFO("Data loading completed in {:.2f} seconds",
              data_load_duration.count());

  // 预先声明需要的变量
  auto h_projection = std::vector<uint32_t>(base_num * top1_projection_degree);
  auto h_second_round_search_merge =
      std::vector<uint32_t>(base_num * final_degree);
  std::vector<std::vector<uint32_t>> fusionNN_graph;
  std::vector<std::vector<uint32_t>> final_graph;
  std::promise<void> cpu_task_completed;
  std::future<void> cpu_future = cpu_task_completed.get_future();

  // 分配GPU内存
  auto gpu_alloc_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Starting GPU memory allocation");

  float* d_base_data = nullptr;
  uint32_t* d_space_128_xx = nullptr;
  uint32_t* d_space_128_yy = nullptr;
  uint32_t* d_top1_projection = nullptr;
  HashTable<uint32_t, 1 << 12>* d_hashtables = nullptr;

  cudaMalloc(&d_base_data, base_num * dim * sizeof(float));
  cudaMalloc(&d_space_128_xx, base_num * gt_degree * sizeof(uint32_t));
  cudaMalloc(
      &d_space_128_yy,
      base_num *
          sizeof(
              pr_neighbor_list<uint32_t, reverse_edge_num, pruned_edge_num>));
  cudaMalloc(&d_top1_projection,
             base_num * top1_projection_degree * sizeof(uint32_t));
  cudaMalloc(&d_hashtables,
             global_warp_num * hashtable_size * sizeof(uint32_t));

  auto gpu_alloc_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gpu_alloc_duration =
      gpu_alloc_end - gpu_alloc_start;
  SPDLOG_INFO("GPU memory allocation completed in {:.2f} seconds",
              gpu_alloc_duration.count());

  // 创建CUDA流和事件
  auto stream_create_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Creating CUDA streams and events");

  cudaStream_t compute_stream, copy_stream;
  cudaEvent_t data_ready, gpu_phase2_done;

  cudaStreamCreate(&compute_stream);
  cudaStreamCreate(&copy_stream);
  cudaEventCreate(&data_ready);
  cudaEventCreate(&gpu_phase2_done);

  // 总体时间测量事件
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 为每个主要内核创建计时事件
  int const NUM_KERNELS = 12;
  cudaEvent_t kernel_starts[NUM_KERNELS], kernel_ends[NUM_KERNELS];
  char const* kernel_names[NUM_KERNELS] = {
      "match_top1_kernel_v0",
      "init_pr_lists_phase1",
      "fusion_prune_reverse_kernel_v0_phase1",
      "fusion_merge_sort_prune_kernel_phase1",
      "link_process_v0_store_base_data_round1",
      "init_pr_lists_round1",
      "fusion_prune_reverse_kernel_v0_round1",
      "fusion_merge_sort_prune_kernel_round1",
      "link_process_v0_store_base_data_round2",
      "init_pr_lists_round2",
      "fusion_prune_reverse_kernel_v0_round2",
      "fusion_merge_sort_prune_kernel_round2"};

  for (int i = 0; i < NUM_KERNELS; i++) {
    cudaEventCreate(&kernel_starts[i]);
    cudaEventCreate(&kernel_ends[i]);
  }

  auto stream_create_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> stream_create_duration =
      stream_create_end - stream_create_start;
  SPDLOG_INFO("CUDA streams and events created in {:.2f}seconds",
              stream_create_duration.count());

  // ===== 开始计时：从这里开始真正的图构建 =====
  auto graph_build_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Starting graph construction");
  cudaEventRecord(start, compute_stream);

  // 1. 异步数据传输到GPU (在copy_stream上)
  auto data_transfer_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Starting data transfer to GPU");

  cudaMemcpyAsync(d_base_data, h_base_data.data(),
                  base_num * dim * sizeof(float), cudaMemcpyHostToDevice,
                  copy_stream);

  cudaMemcpyAsync(d_space_128_xx, h_gt_data.data(),
                  2500000 * gt_degree * sizeof(uint32_t),
                  cudaMemcpyHostToDevice, copy_stream);

  auto h_init_match = std::vector<uint32_t>(base_num * match_degree, tomb);

  cudaMemcpyAsync(d_space_128_yy, h_init_match.data(),
                  base_num * match_degree * sizeof(uint32_t),
                  cudaMemcpyHostToDevice, copy_stream);

  // 标记数据传输完成
  cudaEventRecord(data_ready, copy_stream);

  auto data_transfer_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> data_transfer_duration =
      data_transfer_end - data_transfer_start;
  SPDLOG_INFO("Data transfer to GPU initiated in {:.2f} seconds",
              data_transfer_duration.count());

  // 3. CPU启动单独线程执行MatchNN（与GPU并行）
  auto cpu_thread_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO(
      "Starting CPU thread for MatchNN (will run in parallel with all GPU "
      "work)");

  // 创建线程来执行CPU部分的工作，不阻塞主线程
  std::thread cpu_worker([&]() {
    auto match_start = std::chrono::high_resolution_clock::now();
    SPDLOG_INFO("CPU thread: Starting MatchNN computation");

    // 执行MatchNN计算
    std::vector<std::vector<uint32_t>> topnn_projection_graph =
        MatchNN(base_num, 2500000, match_degree, gt_degree, 40,
                const_cast<uint32_t*>(h_gt_data.data()), ep,
                const_cast<float*>(h_base_data.data()), dim, cpu_thread_limit);

    auto matchnn_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> matchnn_duration = matchnn_end - match_start;
    SPDLOG_INFO(
        "CPU thread: MatchNN base computation completed in {:.2f} seconds",
        matchnn_duration.count());

    std::vector<std::vector<std::vector<uint32_t>>> supply_graphs;
    auto match_sup_start = std::chrono::high_resolution_clock::now();
    SPDLOG_INFO(
        "CPU thread: Starting Match Supplementary computations (rounds: {})",
        match_rounds);

    // 添加动态轮数的补充图
    for (int i = 1; i < match_rounds + 1; i++) {
      auto round_start = std::chrono::high_resolution_clock::now();
      std::vector<std::vector<uint32_t>> proj_graph = MatchSup(
          base_num, 2500000, i, gt_degree, 40,
          const_cast<uint32_t*>(h_gt_data.data()),
          const_cast<float*>(h_base_data.data()), dim, cpu_thread_limit);
      auto round_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> round_duration = round_end - round_start;
      SPDLOG_INFO("CPU thread: Top{} projection completed in {:.2f} seconds", i,
                  round_duration.count());

      supply_graphs.push_back(proj_graph);
    }

    auto match_sup_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> match_sup_duration =
        match_sup_end - match_sup_start;
    SPDLOG_INFO(
        "CPU thread: All Match Supplementary computations ({} rounds) "
        "completed in {:.2f} seconds",
        match_rounds, match_sup_duration.count());

    auto fusion_nn_start = std::chrono::high_resolution_clock::now();
    SPDLOG_INFO("CPU thread: Starting FusionNN");

    fusionNN_graph =
        FusionNN(base_num, 40, const_cast<float*>(h_base_data.data()),
                 topnn_projection_graph, supply_graphs, dim, cpu_thread_limit);

    auto fusion_nn_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fusion_nn_duration =
        fusion_nn_end - fusion_nn_start;
    SPDLOG_INFO("CPU thread: FusionNN completed in {:.2f}seconds",
                fusion_nn_duration.count());

    auto match_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> match_duration = match_end - match_start;
    SPDLOG_INFO("CPU thread: All CPU computations completed in {:.2f}seconds",
                match_duration.count());

    auto stat_start = std::chrono::high_resolution_clock::now();
    statDegree(base_num, fusionNN_graph);
    auto stat_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> stat_duration = stat_end - stat_start;
    SPDLOG_INFO("CPU thread: Graph statistics calculated in {:.2f} seconds",
                stat_duration.count());

    // 通知主线程CPU工作已完成
    cpu_task_completed.set_value();
  });

  // 2. GPU计算第一阶段 - top1 projection
  auto gpu_phase1_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Starting GPU Phase 1 - top1 projection");

  cudaStreamWaitEvent(compute_stream, data_ready, 0);

  // GPU Kernels - 第一阶段 (top1 projection)
  constexpr uint32_t match_grid_size = 144;
  constexpr uint32_t match_block_size = 512;

  // KERNEL 0: match_top1_kernel_v0
  cudaEventRecord(kernel_starts[0], compute_stream);
  match_top1_kernel_v0<match_grid_size, match_block_size, 2500000, gt_degree,
                       match_degree, 127, tomb, float, uint32_t>
      <<<match_grid_size, match_block_size, 0, compute_stream>>>(
          d_space_128_xx, d_space_128_yy);
  cudaEventRecord(kernel_ends[0], compute_stream);

  constexpr uint32_t grid_size = 144;
  constexpr uint32_t block_size = 256;

  // KERNEL 1: init_pr_lists_phase1
  cudaEventRecord(kernel_starts[1], compute_stream);
  init_pr_lists<grid_size, block_size, base_num, match_degree, pruned_edge_num,
                reverse_edge_num, tomb, dim, true, float, uint32_t>
      <<<grid_size, block_size, 0, compute_stream>>>(
          (pr_neighbor_list<uint32_t, reverse_edge_num, pruned_edge_num>*)
              d_space_128_xx);
  cudaEventRecord(kernel_ends[1], compute_stream);

  // KERNEL 2: fusion_prune_reverse_kernel_v0_phase1
  cudaEventRecord(kernel_starts[2], compute_stream);
  fusion_prune_reverse_kernel_v0<grid_size, block_size, base_num, match_degree,
                                 pruned_edge_num, reverse_edge_num, tomb, dim,
                                 false, float, uint32_t>
      <<<grid_size, block_size, 0, compute_stream>>>(
          d_base_data, d_space_128_yy,
          (pr_neighbor_list<uint32_t, reverse_edge_num, pruned_edge_num>*)
              d_space_128_xx);
  cudaEventRecord(kernel_ends[2], compute_stream);

  constexpr uint32_t projection_grid_size = 144;
  constexpr uint32_t projection_block_size = 256;

  // KERNEL 3: fusion_merge_sort_prune_kernel_phase1
  cudaEventRecord(kernel_starts[3], compute_stream);
  fusion_merge_sort_prune_kernel<projection_grid_size, projection_block_size,
                                 base_num, pruned_edge_num, reverse_edge_num,
                                 top1_projection_degree, tomb, dim, false,
                                 float, uint32_t>
      <<<projection_grid_size, projection_block_size, 0, compute_stream>>>(
          d_base_data,
          (pr_neighbor_list<uint32_t, reverse_edge_num, pruned_edge_num>*)
              d_space_128_xx,
          d_top1_projection);
  cudaEventRecord(kernel_ends[3], compute_stream);

  // 创建计算完成事件等待copy_stream
  cudaEvent_t phase1_compute_done;
  cudaEventCreate(&phase1_compute_done);
  cudaEventRecord(phase1_compute_done, compute_stream);

  // 确保数据拷贝前计算已完成
  cudaStreamWaitEvent(copy_stream, phase1_compute_done, 0);

  auto phase1_copy_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Copying top1 projection data back to host");

  // 在计算完成后异步拷贝投影数据供最终融合使用
  cudaMemcpyAsync(h_projection.data(), d_top1_projection,
                  base_num * top1_projection_degree * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost, copy_stream);

  cudaEventDestroy(phase1_compute_done);

  auto gpu_phase1_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gpu_phase1_duration =
      gpu_phase1_end - gpu_phase1_start;
  SPDLOG_INFO("GPU Phase 1 launched in {:.2f} seconds",
              gpu_phase1_duration.count());

  // 4. GPU继续执行第二阶段 - 两轮link
  auto gpu_phase2_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Starting GPU Phase 2 - Two rounds of link");

  // KERNEL 4: link_process_v0_store_base_data_round1
  cudaEventRecord(kernel_starts[4], compute_stream);
  link_process_v0_store_base_data<
      first_round_search_grid_size, first_round_search_block_size, base_num,
      query_num, dim, top1_projection_degree, shared_memory_size, Km, Kp, Kd,
      topk, 0XFFFFFFFF, hashtable_size, reset_iter, uint32_t, float>
      <<<first_round_search_grid_size, first_round_search_block_size,
         shared_memory_size, compute_stream>>>(
          d_base_data, d_hashtables, d_top1_projection, d_space_128_xx);
  cudaEventRecord(kernel_ends[4], compute_stream);

  // KERNEL 5: init_pr_lists_round1
  cudaEventRecord(kernel_starts[5], compute_stream);
  init_pr_lists<grid_size, block_size, base_num, match_degree,
                first_round_pruned_edge_num, first_round_reverse_edge_num, tomb,
                dim, true, float, uint32_t>
      <<<grid_size, block_size, 0, compute_stream>>>(
          (pr_neighbor_list<uint32_t, first_round_reverse_edge_num,
                            first_round_pruned_edge_num>*)d_space_128_yy);
  cudaEventRecord(kernel_ends[5], compute_stream);

  // KERNEL 6: fusion_prune_reverse_kernel_v0_round1
  cudaEventRecord(kernel_starts[6], compute_stream);
  fusion_prune_reverse_kernel_v0<
      grid_size, block_size, base_num, topk, first_round_pruned_edge_num,
      first_round_reverse_edge_num, tomb, dim, true, float, uint32_t>
      <<<grid_size, block_size, 0, compute_stream>>>(
          d_base_data, d_space_128_xx,
          (pr_neighbor_list<uint32_t, first_round_reverse_edge_num,
                            first_round_pruned_edge_num>*)d_space_128_yy);
  cudaEventRecord(kernel_ends[6], compute_stream);

  // KERNEL 7: fusion_merge_sort_prune_kernel_round1
  cudaEventRecord(kernel_starts[7], compute_stream);
  fusion_merge_sort_prune_kernel<grid_size, block_size, base_num,
                                 first_round_pruned_edge_num,
                                 first_round_reverse_edge_num, final_degree,
                                 tomb, dim, true, float, uint32_t>
      <<<projection_grid_size, projection_block_size, 0, compute_stream>>>(
          d_base_data,
          (pr_neighbor_list<uint32_t, first_round_reverse_edge_num,
                            first_round_pruned_edge_num>*)d_space_128_yy,
          d_space_128_xx);
  cudaEventRecord(kernel_ends[7], compute_stream);

  // KERNEL 8: link_process_v0_store_base_data_round2
  cudaEventRecord(kernel_starts[8], compute_stream);
  link_process_v0_store_base_data<
      second_round_search_grid_size, second_round_search_block_size, base_num,
      query_num, dim, final_degree, second_shared_memory_size, second_Km,
      second_Kp, second_Kd, second_topk, 0XFFFFFFFF, hashtable_size, reset_iter,
      uint32_t, float>
      <<<first_round_search_grid_size, first_round_search_block_size,
         shared_memory_size, compute_stream>>>(
          d_base_data, d_hashtables, d_space_128_xx, (uint32_t*)d_space_128_yy);
  cudaEventRecord(kernel_ends[8], compute_stream);

  // KERNEL 9: init_pr_lists_round2
  cudaEventRecord(kernel_starts[9], compute_stream);
  init_pr_lists<grid_size, block_size, base_num, match_degree,
                second_round_pruned_edge_num, second_round_reverse_edge_num,
                tomb, dim, true, float, uint32_t>
      <<<grid_size, block_size, 0, compute_stream>>>(
          (pr_neighbor_list<uint32_t, second_round_reverse_edge_num,
                            second_round_pruned_edge_num>*)d_space_128_xx);
  cudaEventRecord(kernel_ends[9], compute_stream);

  // KERNEL 10: fusion_prune_reverse_kernel_v0_round2
  cudaEventRecord(kernel_starts[10], compute_stream);
  fusion_prune_reverse_kernel_v0<grid_size, block_size, base_num, second_topk,
                                 second_round_pruned_edge_num,
                                 second_round_reverse_edge_num, tomb, dim, true,
                                 float, uint32_t>
      <<<grid_size, block_size, 0, compute_stream>>>(
          d_base_data, (uint32_t*)d_space_128_yy,
          (pr_neighbor_list<uint32_t, second_round_reverse_edge_num,
                            second_round_pruned_edge_num>*)d_space_128_xx);
  cudaEventRecord(kernel_ends[10], compute_stream);

  // KERNEL 11: fusion_merge_sort_prune_kernel_round2
  cudaEventRecord(kernel_starts[11], compute_stream);
  fusion_merge_sort_prune_kernel<grid_size, block_size, base_num,
                                 second_round_pruned_edge_num,
                                 second_round_reverse_edge_num, final_degree,
                                 tomb, dim, true, float, uint32_t>
      <<<projection_grid_size, projection_block_size, 0, compute_stream>>>(
          d_base_data,
          (pr_neighbor_list<uint32_t, second_round_reverse_edge_num,
                            second_round_pruned_edge_num>*)d_space_128_xx,
          (uint32_t*)d_space_128_yy);
  cudaEventRecord(kernel_ends[11], compute_stream);

  // 第二阶段完成，准备数据传输
  cudaEvent_t compute_done;
  cudaEventCreate(&compute_done);
  cudaEventRecord(compute_done, compute_stream);

  // 确保数据拷贝前计算已完成
  cudaStreamWaitEvent(copy_stream, compute_done, 0);

  auto phase2_copy_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Copying second round results back to host");

  // 5. 异步拷贝最终结果给CPU使用
  cudaMemcpyAsync(h_second_round_search_merge.data(), (uint32_t*)d_space_128_yy,
                  base_num * final_degree * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost, copy_stream);

  // 标记GPU第二阶段完成 - 这个事件用于CPU同步
  cudaEventRecord(gpu_phase2_done, copy_stream);

  cudaEventDestroy(compute_done);

  auto gpu_phase2_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gpu_phase2_duration =
      gpu_phase2_end - gpu_phase2_start;
  SPDLOG_INFO("GPU Phase 2 launched in {:.2f} seconds",
              gpu_phase2_duration.count());

  // 6. 等待GPU和CPU任务都完成
  auto sync_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Waiting for GPU and CPU tasks to complete");

  // 使用future的wait_for来检查CPU任务是否完成
  bool gpu_done = false;
  bool cpu_done = false;

  while (!gpu_done || !cpu_done) {
    if (!gpu_done && cudaEventQuery(gpu_phase2_done) == cudaSuccess) {
      gpu_done = true;
      auto gpu_wait_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> gpu_wait_duration =
          gpu_wait_end - sync_start;
      SPDLOG_INFO("GPU tasks completed in {:.2f} seconds",
                  gpu_wait_duration.count());
    }

    if (!cpu_done && cpu_future.wait_for(std::chrono::microseconds(100)) ==
                         std::future_status::ready) {
      cpu_done = true;
      auto cpu_wait_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> cpu_wait_duration =
          cpu_wait_end - sync_start;
      SPDLOG_INFO("CPU tasks completed in {:.2f} seconds",
                  cpu_wait_duration.count());
    }

    // 短暂睡眠以避免忙等待
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // 确保CPU线程已经完成
  if (cpu_worker.joinable()) {
    cpu_worker.join();
    auto cpu_thread_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_thread_duration =
        cpu_thread_end - cpu_thread_start;
    SPDLOG_INFO("CPU worker thread joined after {:.2f}seconds",
                cpu_thread_duration.count());
  }

  auto sync_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> sync_duration = sync_end - sync_start;
  SPDLOG_INFO("Synchronization completed in {:.2f} seconds",
              sync_duration.count());

  // 现在读取所有内核的执行时间
  SPDLOG_INFO("Individual GPU kernel execution times:");
  float kernel_times[NUM_KERNELS];
  for (int i = 0; i < NUM_KERNELS; i++) {
    cudaEventElapsedTime(&kernel_times[i], kernel_starts[i], kernel_ends[i]);
    SPDLOG_INFO("  - {}: {:.3f} seconds", kernel_names[i],
                kernel_times[i] / 1000);
  }

  // 7. 执行最终的融合阶段
  auto fusion_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Starting final fusion phase");

  final_graph = FusionFinal(base_num, top1_projection_degree, final_degree, 70,
                            const_cast<float*>(h_base_data.data()),
                            h_projection, fusionNN_graph,
                            h_second_round_search_merge, dim, cpu_thread_limit);

  auto fusion_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fusion_duration = fusion_end - fusion_start;
  SPDLOG_INFO("FusionFinal completed in {:.2f} seconds",
              fusion_duration.count());

  // 停止GPU计时
  cudaEventRecord(stop, compute_stream);
  cudaEventSynchronize(stop);

  // 计算建图总时间
  auto graph_build_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> graph_build_duration =
      graph_build_end - graph_build_start;
  SPDLOG_INFO("Total graph building time: {:.2f} seconds",
              graph_build_duration.count());

  float gpu_milliseconds = 0;
  cudaEventElapsedTime(&gpu_milliseconds, start, stop);
  SPDLOG_INFO("GPU kernel execution time: {:.3f} seconds",
              gpu_milliseconds / 1000);

  // 8. 保存最终图（不计入建图时间）
  auto save_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Starting graph saving to {}", save_path);

  SaveGraph(final_graph, save_path, ep, base_num);

  auto save_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> save_duration = save_end - save_start;
  SPDLOG_INFO("Graph saving completed in {:.2f} seconds",
              save_duration.count());

  auto stat_final_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Calculating final graph statistics");

  statDegree(base_num, final_graph);
  statDegree(base_num, top1_projection_degree, h_projection);
  statDegree(base_num, final_degree, h_second_round_search_merge);

  auto stat_final_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> stat_final_duration =
      stat_final_end - stat_final_start;
  SPDLOG_INFO("Final graph statistics calculated in {:.2f}seconds",
              stat_final_duration.count());

  // 清理资源
  auto cleanup_start = std::chrono::high_resolution_clock::now();
  SPDLOG_INFO("Starting resource cleanup");

  // 清理内核计时事件
  for (int i = 0; i < NUM_KERNELS; i++) {
    cudaEventDestroy(kernel_starts[i]);
    cudaEventDestroy(kernel_ends[i]);
  }

  cudaStreamDestroy(compute_stream);
  cudaStreamDestroy(copy_stream);
  cudaEventDestroy(data_ready);
  cudaEventDestroy(gpu_phase2_done);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_base_data);
  cudaFree(d_space_128_xx);
  cudaFree(d_space_128_yy);
  cudaFree(d_top1_projection);
  cudaFree(d_hashtables);

  auto cleanup_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cleanup_duration = cleanup_end - cleanup_start;
  SPDLOG_INFO("Resource cleanup completed in {:.2f} seconds",
              cleanup_duration.count());

  auto test_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> test_duration = test_end_time - test_start_time;
  SPDLOG_INFO("Total test execution time: {:.2f} seconds",
              test_duration.count());
  SPDLOG_INFO("Pure graph construction time: {:.2f} seconds",
              graph_build_duration.count());
}

}  // namespace Gpu
}  // namespace Gbuilder
