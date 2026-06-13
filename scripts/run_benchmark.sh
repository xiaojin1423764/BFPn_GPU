# scripts/run_benchmark.sh
#!/bin/bash

# BFPn GPU Solver Benchmark Script

set -e

echo "BFPn GPU Solver Benchmark"
echo "========================"

# 检查可执行文件
if [ ! -f "../build/bin/bfp_solver" ]; then
    echo "Building project..."
    mkdir -p ../build
    cd ../build
    cmake ..
    make -j$(nproc)
    cd ../scripts
fi

# 运行不同规模的测试
echo ""
echo "Test 1: Small grid (NY=10, NZ=10, NG=100)"
../build/bin/bfp_solver --data ../data --time 10.0

echo ""
echo "Test 2: Medium grid (NY=20, NZ=20, NG=250)"
# 修改参数后运行

echo ""
echo "Test 3: Full grid (NY=10, NZ=10, NG=500)"
../build/bin/bfp_solver --data ../data --time 40.0

echo ""
echo "Benchmark complete!"
