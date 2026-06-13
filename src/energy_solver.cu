// src/energy_solver.cu
#include <cublas_v2.h>
#include "energy_solver.cuh"
#include "../include/config.hpp"
#include <algorithm>

// 基于总截面的简化能量衰减核
__global__ void energyAttenuationKernel(
    double* F, double* f_F,
    const double* sigma_c,
    int Ng, int nyz, int NmuNom,
    double dt
) {
    const long long n_per_E = static_cast<long long>(nyz) * NmuNom;
    const long long total   = n_per_E * Ng;
    
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    
    int ek = static_cast<int>(gid / n_per_E);
    double sigma = sigma_c[ek];
    
    double Fold = F[gid];
    double decay = exp(-sigma * dt);
    double Fnew = Fold * decay;
    
    F[gid] = Fnew;
    if (f_F) {
        f_F[gid] = Fold - Fnew;  // 视为在该步沉积的能量（或通量损失）
    }
}

// PCR求解核函数（当前未在简化版本中使用，保留以便后续扩展）
template<int BLOCK_SIZE>
__global__ void pcrKernel(
    double* x,
    const double* a_in, const double* b_in, 
    const double* c_in, const double* d_in,
    size_t pitch, int n_eqs, int n_systems
) {
    __shared__ double s_a[BLOCK_SIZE];
    __shared__ double s_b[BLOCK_SIZE];
    __shared__ double s_c[BLOCK_SIZE];
    __shared__ double s_d[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (bid >= n_systems) return;
    
    // 计算pitched索引
    char* a_ptr = (char*)a_in + bid * pitch;
    char* b_ptr = (char*)b_in + bid * pitch;
    char* c_ptr = (char*)c_in + bid * pitch;
    char* d_ptr = (char*)d_in + bid * pitch;
    char* x_ptr = (char*)x + bid * pitch;
    
    double* a_row = (double*)(a_ptr);
    double* b_row = (double*)(b_ptr);
    double* c_row = (double*)(c_ptr);
    double* d_row = (double*)(d_ptr);
    double* x_row = (double*)(x_ptr);
    
    // 加载到共享内存
    if (tid < n_eqs) {
        s_a[tid] = (tid > 0) ? a_row[tid] : 0.0;
        s_b[tid] = b_row[tid];
        s_c[tid] = (tid < n_eqs - 1) ? c_row[tid] : 0.0;
        s_d[tid] = d_row[tid];
    }
    __syncthreads();
    
    // PCR迭代
    int step = 1;
    while (step < n_eqs) {
        double new_a = 0.0, new_b = 0.0, new_c = 0.0, new_d = 0.0;
        
        if (tid < n_eqs) {
            int left = tid - step;
            int right = tid + step;
            
            // 左侧约简
            if (left >= 0) {
                double r = 1.0 / (s_b[left] - s_a[left] * 
                    ((left - step >= 0) ? s_c[left - step] : 0.0));
                double alpha = -s_a[tid] * r;
                new_a = alpha * s_a[left];
                new_d = alpha * s_d[left];
            }
            
            // 右侧约简
            if (right < n_eqs) {
                double r = 1.0 / (s_b[right] - s_c[right] * 
                    ((right + step < n_eqs) ? s_a[right + step] : 0.0));
                double beta = -s_c[tid] * r;
                new_c = beta * s_c[right];
                new_d += beta * s_d[right];
            }
            
            new_b = s_b[tid];
            if (left >= 0) new_b += (-s_a[tid]) * s_c[left] * 
                1.0 / (s_b[left] - s_a[left] * ((left-step>=0)?s_c[left-step]:0.0));
            if (right < n_eqs) new_b += (-s_c[tid]) * s_a[right] * 
                1.0 / (s_b[right] - s_c[right] * ((right+step<n_eqs)?s_a[right+step]:0.0));
            
            new_d += s_d[tid];
        }
        
        __syncthreads();
        
        if (tid < n_eqs) {
            s_a[tid] = new_a;
            s_b[tid] = new_b;
            s_c[tid] = new_c;
            s_d[tid] = new_d;
        }
        
        __syncthreads();
        step *= 2;
    }
    
    // 写回结果
    if (tid < n_eqs) {
        x_row[tid] = s_d[tid] / s_b[tid];
    }
}

EnergyDepositionSolver::EnergyDepositionSolver(
    int max_batch_size, int n_eqs, cublasHandle_t handle)
    : max_batch(max_batch_size), n_equations(n_eqs), cublas_handle(handle) {
    
    // 分配pitched内存
    d_a.allocate(n_eqs, max_batch);
    d_b.allocate(n_eqs, max_batch);
    d_c.allocate(n_eqs, max_batch);
    d_d.allocate(n_eqs, max_batch);
    d_x.allocate(n_eqs, max_batch);
    
    d_rhs_buffer.allocate(max_batch * n_eqs);
}

void EnergyDepositionSolver::solve(
    double* F, double* f_F,
    const double* F_half, const double* f_F_half,
    const double* S_s, const double* sigma_c, const double* T_c,
    double dt, double dg,
    int nyz, int NmuNom,
    bool is_secondary, const double* source_term,
    cudaStream_t stream) {
    
    // 简化版本：忽略能群之间的三对角耦合，仅基于 sigma_c 对每个能群做指数衰减，
    // 并把本步损失记入 f_F，保证数值演化与剂量计算有意义。
    int Ng = n_equations;
    long long total = static_cast<long long>(Ng) * nyz * NmuNom;
    
    int block_size = 256;
    int grid_size  = static_cast<int>((total + block_size - 1) / block_size);
    
    energyAttenuationKernel<<<grid_size, block_size, 0, stream>>>(
        F, f_F,
        sigma_c,
        Ng, nyz, NmuNom,
        dt
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void EnergyDepositionSolver::solveBatch(
    double* F, double* f_F,
    const double* F_half, const double* f_F_half,
    const double* S_s, const double* sigma_c, const double* T_c,
    double dt, double dg,
    int nyz, int NmuNom, int batch_start, int batch_size,
    bool is_secondary, const double* source_term,
    cudaStream_t stream) {
    
    // 构建三对角系统 (使用kernel并行构建)
    // 这里简化，实际需要一个专门的build kernel
    
    // 选择PCR block size
    if (n_equations <= 256) {
        pcrKernel<256><<<batch_size, 256, 0, stream>>>(
            d_x.data(),
            (double*)((char*)d_a.data()),  // 需要正确处理pitch
            (double*)((char*)d_b.data()),
            (double*)((char*)d_c.data()),
            (double*)((char*)d_d.data()),
            d_a.getPitch(), n_equations, batch_size
        );
    } else if (n_equations <= 512) {
        pcrKernel<512><<<batch_size, 512, 0, stream>>>(
            d_x.data(),
            (double*)((char*)d_a.data()),
            (double*)((char*)d_b.data()),
            (double*)((char*)d_c.data()),
            (double*)((char*)d_d.data()),
            d_a.getPitch(), n_equations, batch_size
        );
    }
    
    // 复制结果回f_F
    // 更新F
}
