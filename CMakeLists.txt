cmake_minimum_required(VERSION 3.20)
project(learn_FlashAttention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(NvidiaCutlass REQUIRED)
find_package(CCCL REQUIRED)
find_package(Torch REQUIRED)

add_library(FlashAttentionUtility utility/torch_utils.cpp)

target_include_directories(FlashAttentionUtility
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(FlashAttentionUtility
    PUBLIC
        nvidia::cutlass::cutlass
        CCCL::CCCL           
        ${TORCH_LIBRARIES}
)

add_subdirectory(FlashAttentionV2)