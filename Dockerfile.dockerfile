# Dockerfile
# BFPn GPU Solver - Multi-stage build

# Stage 1: Build environment
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /src

# 复制源代码
COPY . /src/

# 创建构建目录并编译
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=70 \
        && \
    make -j$(nproc)

# Stage 2: Runtime environment
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-numpy \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

# 安装Python包
RUN pip3 install --no-cache-dir numpy matplotlib scipy

# 复制可执行文件
COPY --from=builder /src/build/bin/bfp_solver /usr/local/bin/
COPY --from=builder /src/scripts /usr/local/share/bfpn/scripts

# 创建数据目录
RUN mkdir -p /data

# 设置环境变量
ENV CUDA_VISIBLE_DEVICES=0
ENV BFPN_DATA_DIR=/data

# 工作目录
WORKDIR /workspace

# 默认命令
CMD ["bfp_solver", "--help"]

# 入口点脚本
COPY docker/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["entrypoint.sh"]
