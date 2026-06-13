# scripts/benchmark.py
#!/usr/bin/env python3
"""
BFPn Solver Performance Benchmark
"""

import subprocess
import time
import json
import argparse
import os
import re

def run_benchmark(executable, nx, ny, nz, ng, nmu, nom, t_final, gpu_id=0):
    """运行单次基准测试"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [
        executable,
        '--nx', str(nx),
        '--ny', str(ny),
        '--nz', str(nz),
        '--ng', str(ng),
        '--nmu', str(nmu),
        '--nom', str(nom),
        '--time', str(t_final)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    start = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    elapsed = time.time() - start
    
    # 解析输出
    output = result.stdout + result.stderr
    
    # 提取性能数据
    gpu_time = None
    if 'GPU time' in output:
        match = re.search(r'GPU time:\s*([\d.]+)\s*ms', output)
        if match:
            gpu_time = float(match.group(1))
    
    return {
        'nx': nx, 'ny': ny, 'nz': nz, 'ng': ng, 'nmu': nmu, 'nom': nom,
        't_final': t_final,
        'wall_time': elapsed,
        'gpu_time': gpu_time,
        'success': result.returncode == 0,
        'output': output
    }

def scaling_test(executable, output_file):
    """规模扩展测试"""
    results = []
    
    # 基础配置
    base_configs = [
        {'nx': 1000, 'ny': 10, 'nz': 10, 'ng': 100, 'nmu': 11, 'nom': 11},
        {'nx': 2000, 'ny': 10, 'nz': 10, 'ng': 250, 'nmu': 11, 'nom': 11},
        {'nx': 4000, 'ny': 10, 'nz': 10, 'ng': 500, 'nmu': 11, 'nom': 11},
        {'nx': 4000, 'ny': 20, 'nz': 20, 'ng': 500, 'nmu': 21, 'nom': 21},
    ]
    
    for config in base_configs:
        print(f"\n{'='*50}")
        print(f"Testing config: {config}")
        
        result = run_benchmark(executable, t_final=10.0, **config)
        results.append(result)
        
        print(f"Wall time: {result['wall_time']:.2f}s")
        if result['gpu_time']:
            print(f"GPU time: {result['gpu_time']:.2f}ms")
    
    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # 生成报告
    generate_report(results, output_file.replace('.json', '.txt'))

def generate_report(results, output_file):
    """生成文本报告"""
    with open(output_file, 'w') as f:
        f.write("BFPn GPU Solver Benchmark Report\n")
        f.write("="*60 + "\n\n")
        
        for r in results:
            f.write(f"Configuration:\n")
            f.write(f"  Grid: {r['nx']}x{r['ny']}x{r['nz']}\n")
            f.write(f"  Energy: {r['ng']}, Angle: {r['nmu']}x{r['nom']}\n")
            f.write(f"  Wall time: {r['wall_time']:.3f}s\n")
            if r['gpu_time']:
                f.write(f"  GPU time: {r['gpu_time']:.3f}ms\n")
            f.write(f"  Status: {'SUCCESS' if r['success'] else 'FAILED'}\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description='Benchmark BFPn solver')
    parser.add_argument('-e', '--executable', type=str, default='./build/bin/bfp_solver',
                       help='Path to solver executable')
    parser.add_argument('-o', '--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--scaling', action='store_true', help='Run scaling tests')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.executable):
        print(f"Error: Executable {args.executable} not found")
        print("Please build the project first: mkdir build && cd build && cmake .. && make")
        return 1
    
    if args.scaling:
        scaling_test(args.executable, args.output)
    else:
        # 单次测试
        result = run_benchmark(args.executable, 4000, 10, 10, 500, 11, 11, 40.0, args.gpu)
        print(f"\nWall time: {result['wall_time']:.2f}s")
        print(f"Success: {result['success']}")
    
    return 0

if __name__ == '__main__':
    exit(main())
