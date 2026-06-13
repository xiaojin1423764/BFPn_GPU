// src/spatial_transport.cu
#include "spatial_transport.cuh"
#include "../include/config.hpp"
#include <vector>

SpatialTransportSolver::SpatialTransportSolver(int ny, int nz, double dy_, double dz_) 
    : Ny(ny), Nz(nz), nyz((ny+1)*(nz+1)), dy(dy_), dz(dz_) {
    
    CUFFT_CHECK(cufftPlan2d(&plan_forward, Ny+1, Nz+1, CUFFT_D2Z));
    CUFFT_CHECK(cufftPlan2d(&plan_inverse, Ny+1, Nz+1, CUFFT_Z2D));
    
    d_fft_buffer.allocate(nyz);
    d_transport_factor.allocate(nyz);
    d_ky.allocate(nyz);
    d_kz.allocate(nyz);
}

SpatialTransportSolver::~SpatialTransportSolver() {
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
}

// 显式一阶迎风空间传输核
__global__ void spatialTransportKernel(
    const double* F_in,
    double* F_out,
    const double* Omend,
    int Ny, int Nz,
    int Ng, int NmuNom,
    double dy, double dz,
    double dt
) {
    int Ny1 = Ny + 1;
    int Nz1 = Nz + 1;
    long long nyz = static_cast<long long>(Ny1) * Nz1;
    
    long long total = static_cast<long long>(Ng) * nyz * NmuNom;
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    
    long long s_total = gid / NmuNom;
    int ang = static_cast<int>(gid % NmuNom);
    int ek  = static_cast<int>(s_total / nyz);
    long long s = s_total % nyz;
    
    int j = static_cast<int>(s / Ny1);
    int i = static_cast<int>(s % Ny1);
    
    long long base = (static_cast<long long>(ek) * nyz + s) * NmuNom + ang;
    
    // 方向余弦（这里假定 Omend 存的是 (omega_y, omega_z)）
    double omega_y = Omend[2 * ang + 0];
    double omega_z = Omend[2 * ang + 1];
    
    double F_center = F_in[base];
    
    // y 方向迎风差分
    long long s_im1 = (i > 0)        ? (j * Ny1 + (i - 1)) : s;
    long long s_ip1 = (i < Ny1 - 1)  ? (j * Ny1 + (i + 1)) : s;
    long long idx_im1 = (static_cast<long long>(ek) * nyz + s_im1) * NmuNom + ang;
    long long idx_ip1 = (static_cast<long long>(ek) * nyz + s_ip1) * NmuNom + ang;
    
    double dFdy = 0.0;
    if (omega_y > 0.0) {
        dFdy = (F_center - F_in[idx_im1]) / dy;
    } else if (omega_y < 0.0) {
        dFdy = (F_in[idx_ip1] - F_center) / dy;
    }
    
    // z 方向迎风差分
    long long s_jm1 = (j > 0)        ? ((j - 1) * Ny1 + i) : s;
    long long s_jp1 = (j < Nz1 - 1)  ? ((j + 1) * Ny1 + i) : s;
    long long idx_jm1 = (static_cast<long long>(ek) * nyz + s_jm1) * NmuNom + ang;
    long long idx_jp1 = (static_cast<long long>(ek) * nyz + s_jp1) * NmuNom + ang;
    
    double dFdz = 0.0;
    if (omega_z > 0.0) {
        dFdz = (F_center - F_in[idx_jm1]) / dz;
    } else if (omega_z < 0.0) {
        dFdz = (F_in[idx_jp1] - F_center) / dz;
    }
    
    double F_new = F_center - dt * (omega_y * dFdy + omega_z * dFdz);
    F_out[base] = F_new;
}

void SpatialTransportSolver::initializeWaveNumbers() {
    std::vector<double> h_ky(nyz), h_kz(nyz);
    
    // 计算波数 (FFT shift后的顺序)
    for (int j = 0; j <= Nz; j++) {
        for (int i = 0; i <= Ny; i++) {
            int idx = j * (Ny+1) + i;
            
            double ky = 2.0 * Physics::PI / (Ny+1) * (i - (Ny+1)/2);
            double kz = 2.0 * Physics::PI / (Nz+1) * (j - (Nz+1)/2);
            
            // FFT shift
            int i_shift = (i + (Ny+1)/2) % (Ny+1);
            int j_shift = (j + (Nz+1)/2) % (Nz+1);
            int idx_shift = j_shift * (Ny+1) + i_shift;
            
            h_ky[idx_shift] = ky;
            h_kz[idx_shift] = kz;
        }
    }
    
    d_ky.copyFromHost(h_ky.data(), nyz);
    d_kz.copyFromHost(h_kz.data(), nyz);
}

void SpatialTransportSolver::initializeTransportFactor(
    const double* Omend, double dt, int NmuNom) {
    
    // 对每个角度方向计算传输因子
    // (2/dt - i*ky*omega1 - i*kz*omega2) / (2/dt + i*ky*omega1 + i*kz*omega2)
    
    // 这里简化处理，实际需要为每个角度计算
}

void SpatialTransportSolver::solve(double* F, const double* Omend,
                                    int Ng, int NmuNom, double dt,
                                    cudaStream_t stream) {
    // 显式一阶迎风格式的简化实现：不使用 FFT，而是在物理空间直接对
    // y、z 方向做平流更新。为了避免读写冲突，使用临时数组存放 F_new。
    long long Ny1 = Ny + 1;
    long long Nz1 = Nz + 1;
    long long nyz_ll = Ny1 * Nz1;
    long long total = static_cast<long long>(Ng) * nyz_ll * NmuNom;
    
    if (d_F_tmp.getSize() < static_cast<size_t>(total)) {
        d_F_tmp.allocate(static_cast<size_t>(total));
    }
    
    int block_size = 256;
    int grid_size  = static_cast<int>((total + block_size - 1) / block_size);
    
    spatialTransportKernel<<<grid_size, block_size, 0, stream>>>(
        F,                          // F_in
        d_F_tmp.data(),             // F_out
        Omend,
        Ny, Nz,
        Ng, NmuNom,
        dy, dz,
        dt
    );
    CUDA_CHECK(cudaGetLastError());
    
    // 将更新结果写回 F
    CUDA_CHECK(cudaMemcpyAsync(
        F,
        d_F_tmp.data(),
        static_cast<size_t>(total) * sizeof(double),
        cudaMemcpyDeviceToDevice,
        stream
    ));
}
