set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
add_library(${PROJECT_NAME}CUDA SHARED ${CUDA_SOURCES})
set_target_properties(${PROJECT_NAME}CUDA PROPERTIES CUDA_ARCHITECTURES "80;86")

file(GLOB_RECURSE CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(${PROJECT_NAME} SHARED ${CPP_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}CUDA spdlog::spdlog)
