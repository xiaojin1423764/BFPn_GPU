// src/energy_solver.cu
#include <cublas_v2.h>
#include "energy_solver.cuh"
#include "../include/config.hpp"
#include <algorithm>

__global__ void energyDgCnKernel(
    const double* F_old,
    const double* f_old,
    double* F_new,
    double* f_new,
    const double* S_s,
    const double* sigma_c,
    const double* T_c,
    int Ng,
    int nyz,
    int NmuNom,
    double dt,
    double dg,
    double density,
    bool is_secondary,
    const double* source_term,
    const double* source_term_f,
    bool use_legacy,
    bool enable_straggling
) {
    const long long n_per_E = static_cast<long long>(nyz) * NmuNom;
    long long lane = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lane >= n_per_E) return;

    const double dg2 = max(dg * dg, Numerics::EPSILON);

    auto load = [&](const double* arr, int ek) -> double {
        if (!arr || ek < 0 || ek >= Ng) return 0.0;
        return arr[static_cast<long long>(ek) * n_per_E + lane];
    };

    double next_F = 0.0;
    double next_f = 0.0;

    for (int ek = Ng - 1; ek >= 0; --ek) {
        long long idx = static_cast<long long>(ek) * n_per_E + lane;
        double Fg = F_old[idx];
        double fg = f_old ? f_old[idx] : 0.0;
        double sig = max(sigma_c[ek], 0.0);
        double S_lo = max(S_s[ek], 0.0);
        double S_hi = max(S_s[ek + 1], 0.0);
        double source = (is_secondary && source_term) ? source_term[idx] : 0.0;
        double source2 = (is_secondary && source_term_f) ? source_term_f[idx] : 0.0;
        double T_g = (enable_straggling && T_c) ? max(T_c[ek], 0.0) : 0.0;
        double T_hi = (enable_straggling && T_c && ek + 1 < Ng) ? max(T_c[ek + 1], 0.0) : 0.0;
        double T_lo = (enable_straggling && T_c && ek > 0) ? max(T_c[ek - 1], 0.0) : 0.0;

        if (use_legacy) {
            double denom = 1.0 / dt + density * S_lo / dg + sig;
            double F_val = (max(Fg, 0.0) / dt + density * S_hi / dg * next_F + source)
                         / max(denom, Numerics::EPSILON);
            double denom_f = 1.0 / dt + density * S_lo / dg + sig;
            double f_val = (fg / dt + density * S_hi / dg * next_f)
                         / max(denom_f, Numerics::EPSILON);
            if (T_g > 0.0) {
                double Fhi = load(F_old, ek + 1);
                double Flo = load(F_old, ek - 1);
                double lap = max(Fhi, 0.0) - 2.0 * max(Fg, 0.0) + max(Flo, 0.0);
                F_val += 0.5 * dt * T_g / dg2 * lap;
            }
            F_new[idx] = max(F_val, 0.0);
            f_new[idx] = max(f_val, 0.0);
            next_F = F_new[idx];
            next_f = f_new[idx];
            continue;
        }

        const double p1_hi = load(F_old, ek + 1);
        const double p2_hi = load(f_old, ek + 1);
        const double p1_lo = load(F_old, ek - 1);
        const double up = p1_hi - p2_hi;
        const double cur = Fg - fg;

        const double den_F = max(1.0 / dt + 0.5 * density * S_lo / dg + 0.5 * sig,
                                 Numerics::EPSILON);
        const double inv_F = 1.0 / den_F;
        const double A = (1.5 * density * S_hi / dg) * inv_F;
        const double F_from_f = (0.5 * density * S_lo / dg) * inv_F;
        const double F_from_hi = (0.5 * density * S_hi / dg) * inv_F;
        const double strag_rhs = 0.5 * density * T_hi / dg2 * p1_hi
                               + 0.5 * density * T_lo / dg2 * p1_lo
                               - density * T_g / dg2 * Fg;
        const double rhs4 = inv_F * (0.5 * density * S_hi / dg * up
                                    -0.5 * density * S_lo / dg * cur
                                    -0.5 * sig * Fg
                                    + Fg / dt
                                    + strag_rhs
                                    + source);

        const double rhs1e = fg / dt - 0.5 * sig * fg;
        const double rhs2e = 1.5 * density * S_hi / dg * up
                           + 1.5 * density * S_lo / dg * cur
                           - 1.5 * density * (S_lo + S_hi) / dg * Fg
                           - 0.5 * density * (S_hi - S_lo) / dg * fg;
        const double rhs3e = A * (0.5 * density * S_hi / dg * up
                                 -0.5 * density * S_lo / dg * cur
                                 -0.5 * sig * Fg
                                 + Fg / dt
                                 + strag_rhs
                                 + source);

        const double high2 = 1.5 * density * S_hi / dg;
        const double hi_delta = next_F - next_f;
        const double rhs = rhs1e + rhs2e - rhs3e
                         + source2
                         + (high2 - A * 0.5 * density * S_hi / dg) * hi_delta;
        const double coe = 1.0 / dt + 0.5 * sig
                         + 1.5 * density * S_lo / dg
                         + 0.5 * density * (S_hi - S_lo) / dg
                         + A * 0.5 * density * S_lo / dg;

        double f_val = rhs / max(coe, Numerics::EPSILON);
        double F_val = F_from_f * f_val + rhs4 + F_from_hi * hi_delta;

        F_new[idx] = F_val;
        f_new[idx] = f_val;

        next_F = F_new[idx];
        next_f = f_new[idx];
    }
}

__global__ void secondarySourceFromPrimaryKernel(
    const double* F_primary,
    const double* f_primary,
    double* source_F,
    double* source_f,
    const double* ker_e1,
    const double* ker_e2,
    const double* ker_v,
    int Ng,
    int nyz,
    int NmuNom,
    double dg
) {
    long long total = static_cast<long long>(Ng) * nyz * NmuNom;
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    long long n_per_E = static_cast<long long>(nyz) * NmuNom;
    int ek = static_cast<int>(gid / n_per_E);
    long long rem = gid - static_cast<long long>(ek) * n_per_E;
    int cell = static_cast<int>(rem / NmuNom);
    int out_ang = static_cast<int>(rem - static_cast<long long>(cell) * NmuNom);

    double q1 = 0.0;
    double q2 = 0.0;
    for (int src_e = 0; src_e < Ng; ++src_e) {
        const double e1 = ker_e1[static_cast<long long>(ek) * Ng + src_e];
        const double e2 = ker_e2[static_cast<long long>(ek) * Ng + src_e];
        if (e1 == 0.0 && e2 == 0.0) continue;

        const double* kv = ker_v + static_cast<long long>(src_e) * NmuNom * NmuNom;
        for (int in_ang = 0; in_ang < NmuNom; ++in_ang) {
            const double w = kv[static_cast<long long>(in_ang) * NmuNom + out_ang];
            if (w == 0.0) continue;

            const long long src_idx = (static_cast<long long>(src_e) * nyz + cell) * NmuNom + in_ang;
            const double p1 = max(F_primary[src_idx], 0.0);
            const double p2 = f_primary ? f_primary[src_idx] : 0.0;
            q1 += p1 * e1 * w;
            q2 += p2 * e2 * w;
        }
    }

    source_F[gid] = dg * (q1 + q2 / 3.0);
    source_f[gid] = 0.0;
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
    double dt, double dg, double density,
    int nyz, int NmuNom,
    bool is_secondary, const double* source_term,
    const double* source_term_f,
    bool use_legacy,
    bool enable_straggling,
    cudaStream_t stream) {
    
    int Ng = n_equations;
    long long n_per_E = static_cast<long long>(nyz) * NmuNom;
    long long total = static_cast<long long>(Ng) * n_per_E;

    if (d_F_old.getSize() < static_cast<size_t>(total)) {
        d_F_old.allocate(static_cast<size_t>(total));
        d_f_old.allocate(static_cast<size_t>(total));
        d_F_new.allocate(static_cast<size_t>(total));
        d_f_new.allocate(static_cast<size_t>(total));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_F_old.data(), F,
                               static_cast<size_t>(total) * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_f_old.data(), f_F,
                               static_cast<size_t>(total) * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));

    int block_size = 256;
    int grid_size  = static_cast<int>((n_per_E + block_size - 1) / block_size);

    energyDgCnKernel<<<grid_size, block_size, 0, stream>>>(
        d_F_old.data(), d_f_old.data(),
        d_F_new.data(), d_f_new.data(),
        S_s, sigma_c, T_c,
        Ng, nyz, NmuNom,
        dt, dg, density,
        is_secondary, source_term, source_term_f, use_legacy, enable_straggling
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(F, d_F_new.data(),
                               static_cast<size_t>(total) * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(f_F, d_f_new.data(),
                               static_cast<size_t>(total) * sizeof(double),
                               cudaMemcpyDeviceToDevice, stream));
}

void EnergyDepositionSolver::solveSecondaryFromPrimary(
    double* F_secondary, double* f_secondary,
    const double* F_primary, const double* f_primary,
    const double* ker_e1, const double* ker_e2, const double* ker_v,
    const double* S_s, const double* sigma_c, const double* T_c,
    double dt, double dg, double density,
    int nyz, int NmuNom,
    bool use_legacy,
    bool enable_straggling,
    cudaStream_t stream) {

    int Ng = n_equations;
    long long n_per_E = static_cast<long long>(nyz) * NmuNom;
    long long total = static_cast<long long>(Ng) * n_per_E;

    if (d_rhs_buffer.getSize() < static_cast<size_t>(total)) {
        d_rhs_buffer.allocate(static_cast<size_t>(total));
    }
    if (d_collision_f_temp.getSize() < static_cast<size_t>(total)) {
        d_collision_f_temp.allocate(static_cast<size_t>(total));
    }

    int block_size = 256;
    int total_grid = static_cast<int>((total + block_size - 1) / block_size);
    secondarySourceFromPrimaryKernel<<<total_grid, block_size, 0, stream>>>(
        F_primary,
        f_primary,
        d_rhs_buffer.data(),
        d_collision_f_temp.data(),
        ker_e1,
        ker_e2,
        ker_v,
        Ng,
        nyz,
        NmuNom,
        dg
    );
    CUDA_CHECK(cudaGetLastError());

    solve(F_secondary, f_secondary,
          F_secondary, f_secondary,
          S_s, sigma_c, T_c,
          dt, dg, density,
          nyz, NmuNom,
          true, d_rhs_buffer.data(),
          nullptr,
          use_legacy,
          enable_straggling,
          stream);
}

void EnergyDepositionSolver::solveBatch(
    double* F, double* f_F,
    const double* F_half, const double* f_F_half,
    const double* S_s, const double* sigma_c, const double* T_c,
    double dt, double dg, double density,
    int nyz, int NmuNom, int batch_start, int batch_size,
    bool is_secondary, const double* source_term,
    const double* source_term_f,
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
