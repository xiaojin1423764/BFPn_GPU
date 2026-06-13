# scripts/generate_test_data.py
#!/usr/bin/env python3
"""
BFPn Solver Test Data Generator

生成质子治疗模拟所需的截面数据文件
"""

import numpy as np
import argparse
import os

# 物理常数
ME = 9.10956e-31      # kg
MP = 1.673e-27        # kg
ALPHA = 1/137.0
HBARC = 197.3         # MeV·fm
C_LIGHT = 2.998e8     # m/s

def generate_cross_sections(Ng, Lg, output_dir):
    """
    生成截面数据文件
    
    Parameters:
    -----------
    Ng : int
        能量组数
    Lg : float
        最大能量 (MeV)
    output_dir : str
        输出目录
    """
    dg = Lg / Ng
    en = np.arange(Ng) * dg + dg/2 + 1.0  # 能量中点
    
    # 初始化数组
    mu = np.zeros((Ng, 3))
    sigma_c = np.zeros(Ng)
    
    print(f"Generating cross sections for {Ng} energy groups...")
    print(f"Energy range: 1.0 - {Lg+1.0} MeV")
    
    for i, E in enumerate(en):
        # 质子速度 (非相对论近似，高能时需要修正)
        if E < 938.3:
            v = np.sqrt(2 * E * 1.6e-13 / MP)
        else:
            # 相对论
            gamma = E / 938.3 + 1
            v = C_LIGHT * np.sqrt(1 - 1/gamma**2)
        
        beta = v / C_LIGHT
        
        # 计算mu参数 (用于角度分布)
        # 基于多次散射理论
        Zt = 3  # 锂靶
        
        # mu(:,0): 角度分布形状参数
        chi_c2 = 0.157 * Zt**(2/3) / E**2  # 特征散射角平方
        mu[i, 0] = 1.5 + 2.0 * np.log(E / 10.0)  # 经验公式
        
        # mu(:,1): 角度分布宽度参数
        mu[i, 1] = 15.0 + 5.0 * np.sqrt(E / 100.0)
        
        # mu(:,2): 能量沉积特征长度
        # 与阻止本领相关
        I = 78e-6  # 平均激发能 (MeV)
        beta_2 = beta**2
        F_beta = np.log(1.02e6 * beta_2 / (1 - beta_2)) - beta_2 - 4.31
        S_col = 0.170 / beta_2 * F_beta + 0.02  # 碰撞阻止本领
        
        # 特征长度
        lambda_val = E / S_col if S_col > 0 else 1000.0
        mu[i, 2] = lambda_val * dg  # 无量纲化
        
        # 总截面 (简化模型)
        # 基于Bethe-Bloch公式和核截面
        sigma_nuclear = 2 * np.pi * HBARC**2 / ALPHA**2 / 4 * Zt**2 / E**2
        sigma_nuclear *= np.log(1 + (200/E)**2) / 1000  # mb to appropriate units
        
        sigma_c[i] = sigma_nuclear * 0.9  # 应用修正因子
        
        # 特定能量范围的修正 (模拟原始MATLAB代码)
        if 200 <= i < 250:
            aw = np.arange(0.5, 22, 0.5)
            idx = i - 200
            if idx < len(aw):
                sigma_c[i] -= 0.0002 * aw[idx]
    
    # 确保正值
    sigma_c = np.maximum(sigma_c, 1e-10)
    mu[:, 2] = np.maximum(mu[:, 2], 1e-10)
    
    # 保存文件
    np.savetxt(os.path.join(output_dir, 'data_cross.txt'), mu, fmt='%.10e')
    np.savetxt(os.path.join(output_dir, 'cross_total.txt'), sigma_c, fmt='%.10e')
    
    print(f"Files saved to {output_dir}")
    print(f"  - data_cross.txt: shape {mu.shape}")
    print(f"  - cross_total.txt: shape {sigma_c.shape}")
    
    # 生成可视化
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # mu参数
        axes[0, 0].semilogy(en, mu[:, 0], 'b-', label='mu1 (shape)')
        axes[0, 0].semilogy(en, mu[:, 1], 'r-', label='mu2 (width)')
        axes[0, 0].semilogy(en, mu[:, 2], 'g-', label='mu3 (length)')
        axes[0, 0].set_xlabel('Energy (MeV)')
        axes[0, 0].set_ylabel('mu parameters')
        axes[0, 0].legend()
        axes[0, 0].set_title('Cross Section Model Parameters')
        axes[0, 0].grid(True)
        
        # 总截面
        axes[0, 1].semilogy(en, sigma_c, 'b-')
        axes[0, 1].set_xlabel('Energy (MeV)')
        axes[0, 1].set_ylabel('Total Cross Section')
        axes[0, 1].set_title('Total Cross Section vs Energy')
        axes[0, 1].grid(True)
        
        # 阻止本领估算
        S_s = np.zeros(Ng + 1)
        for i in range(Ng + 1):
            E_edge = i * dg + 1.0
            if E_edge < 938.3:
                beta_2 = E_edge * (E_edge + 2*938.3) / (E_edge + 938.3)**2
            else:
                beta_2 = 1 - 1/(E_edge/938.3 + 1)**2
            
            if beta_2 > 0 and beta_2 < 1:
                F_beta = np.log(1.02e6 * beta_2 / (1 - beta_2)) - beta_2 - 4.31
                S_s[i] = 0.170 / beta_2 * F_beta + 0.02
        
        axes[1, 0].semilogy(np.arange(Ng+1) * dg + 1.0, S_s, 'r-')
        axes[1, 0].set_xlabel('Energy (MeV)')
        axes[1, 0].set_ylabel('Stopping Power (MeV/cm)')
        axes[1, 0].set_title('Collision Stopping Power')
        axes[1, 0].grid(True)
        
        # 能量分布示例
        sig_E = 1.0
        E0 = 230.0
        f_E = 1/np.sqrt(2*np.pi)/sig_E * np.exp(-((en - E0)/sig_E/np.sqrt(2))**2)
        axes[1, 1].plot(en, f_E, 'g-')
        axes[1, 1].set_xlabel('Energy (MeV)')
        axes[1, 1].set_ylabel('Probability Density')
        axes[1, 1].set_title('Initial Energy Distribution (Gaussian)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_sections.png'), dpi=150)
        print(f"Visualization saved to {os.path.join(output_dir, 'cross_sections.png')}")
    except ImportError:
        print("matplotlib not available, skipping visualization")

def generate_reference_solution(Nx, Ny, Nz, Ng, output_dir):
    """
    生成参考解用于验证
    """
    print("Generating reference solution (analytical approximation)...")
    
    # 简化的一维深度剂量分布
    # Bragg峰近似
    depth = np.linspace(0, 40, Nx+1)
    
    # 简化Bragg曲线
    range_m = 30.0  # 近似射程
    sigma_range = 1.0
    
    # 高斯展宽的Bragg峰
    dose = np.exp(-((depth - range_m)/sigma_range)**2 / 2)
    dose += 0.1 * np.exp(-depth/10)  # 入口剂量
    
    np.savetxt(os.path.join(output_dir, 'reference_dose.txt'), 
               np.column_stack([depth, dose]), fmt='%.6e')
    
    print(f"Reference solution saved to {os.path.join(output_dir, 'reference_dose.txt')}")

def validate_data_files(data_dir):
    """
    验证数据文件格式
    """
    print("\nValidating data files...")
    
    cross_file = os.path.join(data_dir, 'data_cross.txt')
    sigma_file = os.path.join(data_dir, 'cross_total.txt')
    
    errors = []
    
    if not os.path.exists(cross_file):
        errors.append(f"Missing file: {cross_file}")
    else:
        mu = np.loadtxt(cross_file)
        if mu.ndim != 2 or mu.shape[1] != 3:
            errors.append(f"data_cross.txt has wrong shape: {mu.shape}, expected (N, 3)")
        print(f"  data_cross.txt: OK, shape {mu.shape}")
    
    if not os.path.exists(sigma_file):
        errors.append(f"Missing file: {sigma_file}")
    else:
        sigma = np.loadtxt(sigma_file)
        if sigma.ndim != 1:
            errors.append(f"cross_total.txt has wrong dimensions: {sigma.ndim}")
        print(f"  cross_total.txt: OK, shape {sigma.shape}")
    
    if errors:
        print("\nErrors found:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("  All files valid!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Generate test data for BFPn solver')
    parser.add_argument('-n', '--ng', type=int, default=500, help='Number of energy groups')
    parser.add_argument('-l', '--lmax', type=float, default=259.0, help='Maximum energy (MeV)')
    parser.add_argument('-o', '--output', type=str, default='./data', help='Output directory')
    parser.add_argument('--nx', type=int, default=4000, help='Spatial grid points (for ref sol)')
    parser.add_argument('--ny', type=int, default=10, help='Y grid points')
    parser.add_argument('--nz', type=int, default=10, help='Z grid points')
    parser.add_argument('--validate', action='store_true', help='Validate existing files')
    parser.add_argument('--ref', action='store_true', help='Generate reference solution')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    if args.validate:
        validate_data_files(args.output)
        return
    
    # 生成截面数据
    generate_cross_sections(args.ng, args.lmax, args.output)
    
    # 生成参考解
    if args.ref:
        generate_reference_solution(args.nx, args.ny, args.nz, args.ng, args.output)
    
    # 验证
    validate_data_files(args.output)
    
    print(f"\nDone! Data files ready in {args.output}")

if __name__ == '__main__':
    main()
