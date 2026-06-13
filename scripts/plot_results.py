# scripts/plot_results.py
#!/usr/bin/env python3
"""
BFPn Solver Results Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import os

def load_dose_data(filename):
    """加载剂量数据"""
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]  # time, dose

def plot_dose_evolution(time, dose, output_file=None):
    """绘制剂量随时间演化"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 线性尺度
    axes[0].plot(time, dose, 'b-', linewidth=2)
    axes[0].set_xlabel('Depth (cm)')
    axes[0].set_ylabel('Dose (relative)')
    axes[0].set_title('Depth-Dose Distribution (Linear)')
    axes[0].grid(True)
    
    # 对数尺度
    axes[1].semilogy(time, dose, 'r-', linewidth=2)
    axes[1].set_xlabel('Depth (cm)')
    axes[1].set_ylabel('Dose (relative, log)')
    axes[1].set_title('Depth-Dose Distribution (Log)')
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    else:
        plt.show()

def plot_comparison(sim_file, ref_file, output_file=None):
    """对比模拟结果和参考解"""
    t_sim, dose_sim = load_dose_data(sim_file)
    t_ref, dose_ref = load_dose_data(ref_file)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t_ref, dose_ref, 'k--', linewidth=2, label='Reference (Analytical)')
    ax.plot(t_sim, dose_sim, 'b-', linewidth=1.5, label='BFPn Simulation')
    ax.set_xlabel('Depth (cm)')
    ax.set_ylabel('Relative Dose')
    ax.set_title('Proton Depth-Dose Distribution Comparison')
    ax.legend()
    ax.grid(True)
    
    # 计算误差
    # 插值到相同网格
    dose_ref_interp = np.interp(t_sim, t_ref, dose_ref)
    rel_error = np.abs(dose_sim - dose_ref_interp) / (dose_ref_interp + 1e-10)
    
    ax2 = ax.twinx()
    ax2.semilogy(t_sim, rel_error * 100, 'r:', alpha=0.5, label='Relative Error (%)')
    ax2.set_ylabel('Relative Error (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_file}")
    else:
        plt.show()
    
    print(f"Max relative error: {np.max(rel_error)*100:.2f}%")
    print(f"Mean relative error: {np.mean(rel_error)*100:.2f}%")

def create_3d_visualization(data_dir, output_file=None):
    """创建3D剂量分布可视化 (如果有多维数据)"""
    # 这里可以扩展读取2D/3D剂量分布
    pass

def main():
    parser = argparse.ArgumentParser(description='Plot BFPn solver results')
    parser.add_argument('-i', '--input', type=str, default='dose_output.txt',
                       help='Input dose file')
    parser.add_argument('-r', '--ref', type=str, default=None,
                       help='Reference solution file for comparison')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output plot file')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                       help='Output format')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1
    
    if args.ref:
        if not os.path.exists(args.ref):
            print(f"Error: Reference file {args.ref} not found")
            return 1
        plot_comparison(args.input, args.ref, args.output)
    else:
        time, dose = load_dose_data(args.input)
        plot_dose_evolution(time, dose, args.output)
    
    return 0

if __name__ == '__main__':
    exit(main())
