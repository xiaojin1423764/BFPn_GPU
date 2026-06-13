# docker/entrypoint.sh
#!/bin/bash

# Docker entrypoint for BFPn solver

set -e

# 检查GPU可用性
if ! nvidia-smi > /dev/null 2>&1; then
    echo "Warning: NVIDIA GPU not detected or nvidia-smi not available"
fi

# 生成测试数据（如果不存在）
if [ ! -f "/data/data_cross.txt" ] || [ ! -f "/data/cross_total.txt" ]; then
    echo "Generating test data..."
    python3 /usr/local/share/bfpn/scripts/generate_test_data.py \
        -o /data \
        -n ${NG:-500} \
        -l ${LMAX:-259.0}
fi

# 执行命令
if [ "$1" = "solver" ]; then
    shift
    exec bfp_solver --data /data "$@"
elif [ "$1" = "generate-data" ]; then
    shift
    exec python3 /usr/local/share/bfpn/scripts/generate_test_data.py -o /data "$@"
elif [ "$1" = "plot" ]; then
    shift
    exec python3 /usr/local/share/bfpn/scripts/plot_results.py "$@"
elif [ "$1" = "benchmark" ]; then
    shift
    exec python3 /usr/local/share/bfpn/scripts/benchmark.py -e /usr/local/bin/bfpn_solver "$@"
else
    exec "$@"
fi
