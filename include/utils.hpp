#pragma once
#include "spdlog/spdlog.h"
#include <filesystem>
#include <fstream>
#include <vector>

#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(0);                                                 \
    }                                                          \
  }

template <typename vec_type>
bool read_vec_from_file(std::vector<vec_type>& vec, char const* file_path) {
  if (std::filesystem::exists(file_path)) {
    std::ifstream file(file_path, std::ios::binary);
    if (file.is_open()) {
      file.read(reinterpret_cast<char*>(vec.data()),
                vec.size() * sizeof(vec_type));
      file.close();
      SPDLOG_INFO("loaded from file: {}", file_path);
      return true;
    } else {
      SPDLOG_ERROR("Failed to open file: {}", file_path);
      return false;
    }
  }
  SPDLOG_ERROR("file is not exist: {}", file_path);
  return false;
}

template <typename vec_type>
void dump_vec2_file(std::vector<vec_type> const& vec, char const* file_name) {
  std::ofstream outFile(file_name, std::ios::binary | std::ios::out);
  if (!outFile) {
    SPDLOG_ERROR("fail to open or create the file {}", file_name);
    exit(-1);
  }

  outFile.write(reinterpret_cast<char const*>(vec.data()),
                vec.size() * sizeof(vec_type));
  if (!outFile) {
    SPDLOG_ERROR("failed to write to the file {}", file_name);
    exit(-1);
  }

  outFile.close();
  SPDLOG_INFO("successfully dumped vector to file {}", file_name);
}