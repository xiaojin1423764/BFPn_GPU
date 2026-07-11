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

__device__ __forceinline__ double muscl_paper_slope(double left, double center, double right) {
    double forward = right - center;
    if (fabs(forward) < Numerics::EPSILON) return 0.0;
    double theta = (center - left) / forward;
    double eta = max(0.0, max(min(2.0 * theta, 1.0), min(theta, 2.0)));
    return forward * eta;
}

__device__ __forceinline__ double spatial_flux_divergence_value(
    const double* F_in,
    const double* Omend,
    int ek,
    int ang,
    int i,
    int j,
    int Ny,
    int Nz,
    int NmuNom,
    double dy,
    double dz
) {
    int Ny1 = Ny + 1;
    long long nyz = static_cast<long long>(Ny1) * (Nz + 1);
    long long s = static_cast<long long>(j) * Ny1 + i;
    long long base = (static_cast<long long>(ek) * nyz + s) * NmuNom + ang;

    double omega_y = Omend[2 * ang + 0];
    double omega_z = Omend[2 * ang + 1];

    double c = F_in[base];

    const int im2 = max(0, i - 2);
    const int im1 = max(0, i - 1);
    const int ip1 = min(Ny, i + 1);
    const int ip2 = min(Ny, i + 2);
    const long long plane_base = static_cast<long long>(ek) * nyz * NmuNom + ang;
    const long long y_row = static_cast<long long>(j) * Ny1;
    double ym2 = F_in[plane_base + (y_row + im2) * NmuNom];
    double ym1 = F_in[plane_base + (y_row + im1) * NmuNom];
    double yp1 = F_in[plane_base + (y_row + ip1) * NmuNom];
    double yp2 = F_in[plane_base + (y_row + ip2) * NmuNom];
    double slope_y_m1 = muscl_paper_slope(ym2, ym1, c);
    double slope_y_0 = muscl_paper_slope(ym1, c, yp1);
    double slope_y_p1 = muscl_paper_slope(c, yp1, yp2);

    double left_flux_y;
    double right_flux_y;
    if (omega_y >= 0.0) {
        left_flux_y = omega_y * (ym1 + 0.5 * slope_y_m1);
        right_flux_y = omega_y * (c + 0.5 * slope_y_0);
    } else {
        left_flux_y = omega_y * (c - 0.5 * slope_y_0);
        right_flux_y = omega_y * (yp1 - 0.5 * slope_y_p1);
    }

    const int jm2 = max(0, j - 2);
    const int jm1 = max(0, j - 1);
    const int jp1 = min(Nz, j + 1);
    const int jp2 = min(Nz, j + 2);
    double zm2 = F_in[plane_base + (static_cast<long long>(jm2) * Ny1 + i) * NmuNom];
    double zm1 = F_in[plane_base + (static_cast<long long>(jm1) * Ny1 + i) * NmuNom];
    double zp1 = F_in[plane_base + (static_cast<long long>(jp1) * Ny1 + i) * NmuNom];
    double zp2 = F_in[plane_base + (static_cast<long long>(jp2) * Ny1 + i) * NmuNom];
    double slope_z_m1 = muscl_paper_slope(zm2, zm1, c);
    double slope_z_0 = muscl_paper_slope(zm1, c, zp1);
    double slope_z_p1 = muscl_paper_slope(c, zp1, zp2);

    double left_flux_z;
    double right_flux_z;
    if (omega_z >= 0.0) {
        left_flux_z = omega_z * (zm1 + 0.5 * slope_z_m1);
        right_flux_z = omega_z * (c + 0.5 * slope_z_0);
    } else {
        left_flux_z = omega_z * (c - 0.5 * slope_z_0);
        right_flux_z = omega_z * (zp1 - 0.5 * slope_z_p1);
    }

    return (right_flux_y - left_flux_y) / dy +
           (right_flux_z - left_flux_z) / dz;
}

__global__ void spatialFluxDivergencePlaneKernel(
    const double* F_in,
    double* flux_div,
    const double* Omend,
    int Ny, int Nz,
    int Ng, int NmuNom,
    double dy, double dz
) {
    int Ny1 = Ny + 1;
    int Nz1 = Nz + 1;
    long long nyz = static_cast<long long>(Ny1) * Nz1;
    long long yz_blocks = (nyz + blockDim.x - 1) / blockDim.x;

    long long flat_block = static_cast<long long>(blockIdx.x);
    long long plane = flat_block / yz_blocks;
    long long yz_block = flat_block - plane * yz_blocks;
    if (plane >= static_cast<long long>(Ng) * NmuNom) return;
    int ek = static_cast<int>(plane / NmuNom);
    int ang = static_cast<int>(plane - static_cast<long long>(ek) * NmuNom);
    long long s = yz_block * blockDim.x + threadIdx.x;
    if (s >= nyz) return;

    int j = static_cast<int>(s / Ny1);
    int i = static_cast<int>(s - static_cast<long long>(j) * Ny1);
    long long base = (static_cast<long long>(ek) * nyz + s) * NmuNom + ang;
    flux_div[base] = spatial_flux_divergence_value(
        F_in, Omend, ek, ang, i, j, Ny, Nz, NmuNom, dy, dz
    );
}

__global__ void spatialPredictorKernel(
    const double* F,
    const double* flux_div,
    double* pred,
    double dt,
    bool preserve_sign,
    long long total
) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    double value = F[idx] - dt * flux_div[idx];
    pred[idx] = preserve_sign ? value : max(value, 0.0);
}

__global__ void spatialCorrectorKernel(
    const double* F,
    const double* flux_old,
    const double* flux_pred,
    double* out,
    double dt,
    bool preserve_sign,
    long long total
) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    double value = F[idx] - 0.5 * dt * (flux_old[idx] + flux_pred[idx]);
    out[idx] = preserve_sign ? value : max(value, 0.0);
}

void SpatialTransportSolver::initializeWaveNumbers() {
    std::vector<double> h_ky(nyz), h_kz(nyz);
    
    // 计算波数 (FFT shift后的顺序)
    for (int j = 0; j <= Nz; j++) {
        for (int i = 0; i <= Ny; i++) {
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
                                    cudaStream_t stream,
                                    bool preserve_sign) {
    solveEq18(F, Omend, Ng, NmuNom, dt, stream, preserve_sign);
}

void SpatialTransportSolver::solveEq18(double* F, const double* Omend,
                                        int Ng, int NmuNom, double dt,
                                        cudaStream_t stream,
                                        bool preserve_sign) {
    long long Ny1 = Ny + 1;
    long long Nz1 = Nz + 1;
    long long nyz_ll = Ny1 * Nz1;
    long long total = static_cast<long long>(Ng) * nyz_ll * NmuNom;
    
    if (d_F_pred.getSize() < static_cast<size_t>(total)) {
        d_F_pred.allocate(static_cast<size_t>(total));
        d_flux_old.allocate(static_cast<size_t>(total));
        d_flux_pred.allocate(static_cast<size_t>(total));
    }
    
    int block_size = 256;
    int grid_size  = static_cast<int>((total + block_size - 1) / block_size);
    long long yz_blocks = (nyz_ll + block_size - 1) / block_size;
    long long plane_blocks = static_cast<long long>(Ng) * NmuNom * yz_blocks;

    spatialFluxDivergencePlaneKernel<<<static_cast<int>(plane_blocks), block_size, 0, stream>>>(
        F,
        d_flux_old.data(),
        Omend,
        Ny, Nz,
        Ng, NmuNom,
        dy, dz
    );
    CUDA_CHECK(cudaGetLastError());

    spatialPredictorKernel<<<grid_size, block_size, 0, stream>>>(
        F,
        d_flux_old.data(),
        d_F_pred.data(),
        dt,
        preserve_sign,
        total
    );
    CUDA_CHECK(cudaGetLastError());

    spatialFluxDivergencePlaneKernel<<<static_cast<int>(plane_blocks), block_size, 0, stream>>>(
        d_F_pred.data(),
        d_flux_pred.data(),
        Omend,
        Ny, Nz,
        Ng, NmuNom,
        dy, dz
    );
    CUDA_CHECK(cudaGetLastError());

    spatialCorrectorKernel<<<grid_size, block_size, 0, stream>>>(
        F,
        d_flux_old.data(),
        d_flux_pred.data(),
        F,
        dt,
        preserve_sign,
        total
    );
    CUDA_CHECK(cudaGetLastError());
}
