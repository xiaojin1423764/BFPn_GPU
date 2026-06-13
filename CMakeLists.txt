# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(BFPnSolver VERSION 1.0.0 LANGUAGES CXX CUDA)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA包
find_package(CUDAToolkit REQUIRED)

# 查找其他依赖
find_package(OpenMP REQUIRED)

# 编译选项
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")

# Release模式优化
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# 调试模式
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 源文件
set(CUDA_SOURCES
    src/main.cu
    src/bfpm_solver.cu
    src/energy_solver.cu
    src/collision_integral.cu
    src/spatial_transport.cu
    src/angle_diffusion.cu
)

# 创建可执行文件
add_executable(bfp_solver ${CUDA_SOURCES})

# 链接CUDA库
target_link_libraries(bfp_solver
    CUDA::cudart
    CUDA::cufft
    CUDA::cublas
    CUDA::cusparse
    CUDA::curand
    OpenMP::OpenMP_CXX
)

# 设置目标属性
set_target_properties(bfp_solver PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 安装规则
install(TARGETS bfp_solver
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# 复制数据文件
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

# 测试
enable_testing()
add_test(NAME basic_test COMMAND bfp_solver --test)
