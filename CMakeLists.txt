cmake_minimum_required(VERSION 3.10)
project(GGidxbuild LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_ARCHITECTURES 8.6)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug, Release, RelWithDebInfo, MinSizeRel)." FORCE)
  message(STATUS "Defaulting CMAKE_BUILD_TYPE to Release.")
endif()

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP support not found!")
endif()
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")

    # set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -Xcompiler=-O0,-march=native,-fopenmp,-Wall,-Wno-unknown-pragmas --generate-line-info --ptxas-options=-v,-gencode=arch=compute_86,code=sm_86 -Xcompiler=-fPIC")
    # set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -g -Xcompiler=-O0,-g,-march=native,-fopenmp,-Wall,-Wno-unknown-pragmas --generate-line-info --ptxas-options=-v,-gencode=arch=compute_86,code=sm_86 -Xcompiler=-fPIC")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -Xcompiler=-O0,-march=native,-fopenmp,-Wall,-Wno-unknown-pragmas --generate-line-info --ptxas-options=-v,-O0, -gencode=arch=compute_86,code=sm_86 -Xcompiler=-fPIC")

endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -Xcompiler=-O3,-march=native,-fopenmp,-DNDEBUG,-Wall,-Wno-unknown-pragmas --generate-line-info --ptxas-options=-v,-O3, -gencode=arch=compute_86,code=sm_86 -Xcompiler=-fPIC")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

include(FetchContent)

FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.15.1
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)

add_subdirectory(src)
add_subdirectory(tests)
