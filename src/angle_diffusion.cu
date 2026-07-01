// src/angle_diffusion.cu
#include "angle_diffusion.cuh"
#include "../include/config.hpp"
#include <vector>

// Kernel: 构建扩散因子 (频域)
__global__ void buildDiffusionFactorKernel(
    cufftDoubleComplex* factor,
    const double* sig_trg,
    int Nom, int Nmu, int n_angle, int Ng,
    double du, double dv, double dt,
    double density
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_angle * Ng;
    if (idx >= total) return;
    
    int angle_idx = idx % n_angle;
    int ek = idx / n_angle;
    int i = angle_idx % Nom;  // u index
    int j = angle_idx / Nom;  // v index
    
    // Eq. (20)-(21), DST-I modes m=1..Nom and n=1..Nmu on zero boundaries.
    double lambda_u = (2.0 - 2.0 * cos(Physics::PI * (i + 1) / (Nom + 1))) / (du * du);
    double lambda_v = (2.0 - 2.0 * cos(Physics::PI * (j + 1) / (Nmu + 1))) / (dv * dv);
    double D = 0.5 * density * sig_trg[ek];
    double lambda = D * (lambda_u + lambda_v);
    double numerator = 2.0/dt - lambda;
    double denominator = 2.0/dt + lambda;
    double ratio = numerator / denominator;
    
    factor[idx].x = ratio;
    factor[idx].y = 0.0;
}

// Kernel: 复数乘法
__global__ void complexMultiplyInPlaceKernel(
    cufftDoubleComplex* data,
    const cufftDoubleComplex* factor,
    int n_angle,
    int nyz,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    double a = data[idx].x;
    double b = data[idx].y;
    int slice = idx / n_angle;
    int ek = slice / nyz;
    int factor_idx = ek * n_angle + (idx % n_angle);
    double c = factor[factor_idx].x;
    double d = factor[factor_idx].y;
    
    data[idx].x = a * c - b * d;
    data[idx].y = a * d + b * c;
}

// Kernel: 取绝对值 (ifft后的实数结果)
__global__ void takeAbsoluteKernel(double* out, const cufftDoubleComplex* in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    out[idx] = sqrt(in[idx].x * in[idx].x + in[idx].y * in[idx].y);
}

__global__ void realToComplexKernel(cufftDoubleComplex* out, const double* in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    out[idx].x = in[idx];
    out[idx].y = 0.0;
}

__global__ void complexToRealKernel(double* out, const cufftDoubleComplex* in, double scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    out[idx] = in[idx].x * scale;
}

__global__ void packOddExtensionKernel(
    cufftDoubleComplex* out,
    const double* in,
    int Nom,
    int Nmu,
    int extNom,
    int extNmu,
    int batch
) {
    long long total = static_cast<long long>(batch) * Nmu * Nom;
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    int angle = static_cast<int>(gid % (Nmu * Nom));
    int b = static_cast<int>(gid / (Nmu * Nom));
    int u = angle % Nom;
    int v = angle / Nom;
    int uu = u + 1;
    int vv = v + 1;
    double value = in[gid];

    long long base = static_cast<long long>(b) * extNmu * extNom;
    long long p1 = base + static_cast<long long>(vv) * extNom + uu;
    long long p2 = base + static_cast<long long>(extNmu - vv) * extNom + uu;
    long long p3 = base + static_cast<long long>(vv) * extNom + (extNom - uu);
    long long p4 = base + static_cast<long long>(extNmu - vv) * extNom + (extNom - uu);

    out[p1].x = value;
    out[p1].y = 0.0;
    out[p2].x = -value;
    out[p2].y = 0.0;
    out[p3].x = -value;
    out[p3].y = 0.0;
    out[p4].x = value;
    out[p4].y = 0.0;
}

__global__ void zeroComplexKernel(cufftDoubleComplex* data, long long total) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx].x = 0.0;
    data[idx].y = 0.0;
}

__global__ void multiplyOddSineModesKernel(
    cufftDoubleComplex* data,
    const cufftDoubleComplex* factor,
    int Nom,
    int Nmu,
    int extNom,
    int extNmu,
    int n_angle,
    int nyz,
    int batch
) {
    long long total = static_cast<long long>(batch) * Nmu * Nom;
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    int angle = static_cast<int>(gid % n_angle);
    int b = static_cast<int>(gid / n_angle);
    int ek = b / nyz;
    int u = angle % Nom;
    int v = angle / Nom;
    int uu = u + 1;
    int vv = v + 1;
    double ratio = factor[static_cast<long long>(ek) * n_angle + angle].x;

    long long base = static_cast<long long>(b) * extNmu * extNom;
    long long p1 = base + static_cast<long long>(vv) * extNom + uu;
    long long p2 = base + static_cast<long long>(extNmu - vv) * extNom + uu;
    long long p3 = base + static_cast<long long>(vv) * extNom + (extNom - uu);
    long long p4 = base + static_cast<long long>(extNmu - vv) * extNom + (extNom - uu);

    data[p1].x *= ratio;
    data[p1].y *= ratio;
    data[p2].x *= ratio;
    data[p2].y *= ratio;
    data[p3].x *= ratio;
    data[p3].y *= ratio;
    data[p4].x *= ratio;
    data[p4].y *= ratio;
}

__global__ void unpackOddExtensionKernel(
    double* out,
    const cufftDoubleComplex* in,
    int Nom,
    int Nmu,
    int extNom,
    int extNmu,
    int batch,
    double scale
) {
    long long total = static_cast<long long>(batch) * Nmu * Nom;
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    int angle = static_cast<int>(gid % (Nmu * Nom));
    int b = static_cast<int>(gid / (Nmu * Nom));
    int u = angle % Nom;
    int v = angle / Nom;
    long long src = static_cast<long long>(b) * extNmu * extNom
                  + static_cast<long long>(v + 1) * extNom + (u + 1);
    out[gid] = in[src].x * scale;
}

__global__ void sineForwardKernel(
    const double* in,
    double* coeff,
    const double* coeff_factor,
    const double* sine_u,
    const double* sine_v,
    int Nom,
    int Nmu,
    int n_angle,
    int nyz,
    int Ng
) {
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(Ng) * nyz * n_angle;
    if (gid >= total) return;

    int mode = static_cast<int>(gid % n_angle);
    long long slice = gid / n_angle;
    int ek = static_cast<int>(slice / nyz);
    int mu = mode % Nom;
    int mv = mode / Nom;

    double sum = 0.0;
    for (int j = 0; j < Nmu; ++j) {
        double sv = sine_v[mv * Nmu + j];
        for (int i = 0; i < Nom; ++i) {
            double su = sine_u[mu * Nom + i];
            long long idx = slice * n_angle + j * Nom + i;
            sum += in[idx] * su * sv;
        }
    }
    coeff[gid] = sum * coeff_factor[static_cast<long long>(ek) * n_angle + mode];
}

__global__ void sineInverseKernel(
    const double* coeff,
    double* out,
    const double* sine_u,
    const double* sine_v,
    int Nom,
    int Nmu,
    int n_angle,
    int nyz,
    int Ng
) {
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long total = static_cast<long long>(Ng) * nyz * n_angle;
    if (gid >= total) return;

    int angle = static_cast<int>(gid % n_angle);
    long long slice = gid / n_angle;
    int i = angle % Nom;
    int j = angle / Nom;

    double value = 0.0;
    for (int mv = 0; mv < Nmu; ++mv) {
        double sv = sine_v[mv * Nmu + j];
        for (int mu = 0; mu < Nom; ++mu) {
            double su = sine_u[mu * Nom + i];
            value += coeff[slice * n_angle + mv * Nom + mu] * su * sv;
        }
    }
    out[gid] = value;
}

AngleDiffusionSolver::AngleDiffusionSolver(int nmu, int nom, cublasHandle_t handle)
    : Nmu(nmu), Nom(nom), n_angle(nmu * nom), planned_batch(0),
      planned_ng(0), planned_nyz(0),
      extended_nmu(2 * (nmu + 1)), extended_nom(2 * (nom + 1)),
      extended_angle(2 * (nmu + 1) * 2 * (nom + 1)),
      cached_du(0.0), cached_dv(0.0), cached_density(1.0),
      plan_forward(0), plan_inverse(0), cublas_handle(handle) {
    
    d_fft_buffer.allocate(n_angle);
    d_diffusion_factor.allocate(n_angle);
}

AngleDiffusionSolver::~AngleDiffusionSolver() {
    if (plan_forward) cufftDestroy(plan_forward);
    if (plan_inverse) cufftDestroy(plan_inverse);
}

void AngleDiffusionSolver::initializeDiffusionFactor(
    const double* sig_trg, double du, double dv, double dt, double density) {
    cached_du = du;
    cached_dv = dv;
    cached_density = density;
    
    // solve() builds the energy-dependent factor once Ng is known.
    buildDiffusionFactorKernel<<<(n_angle + 255) / 256, 256>>>(
        d_fft_buffer.data(), sig_trg,
        Nom, Nmu, n_angle, 1, du, dv, dt, density
    );
    
    d_diffusion_factor.copyFromDevice(d_fft_buffer.data(), n_angle);

    std::vector<double> h_sine_u(static_cast<size_t>(Nom) * Nom);
    std::vector<double> h_sine_v(static_cast<size_t>(Nmu) * Nmu);
    for (int mu = 0; mu < Nom; ++mu) {
        for (int i = 0; i < Nom; ++i) {
            h_sine_u[static_cast<size_t>(mu) * Nom + i] =
                sin(Physics::PI * (mu + 1) * (i + 1) / (Nom + 1));
        }
    }
    for (int mv = 0; mv < Nmu; ++mv) {
        for (int j = 0; j < Nmu; ++j) {
            h_sine_v[static_cast<size_t>(mv) * Nmu + j] =
                sin(Physics::PI * (mv + 1) * (j + 1) / (Nmu + 1));
        }
    }
    d_sine_u.allocate(h_sine_u.size());
    d_sine_v.allocate(h_sine_v.size());
    d_sine_u.copyFromHost(h_sine_u.data(), h_sine_u.size());
    d_sine_v.copyFromHost(h_sine_v.data(), h_sine_v.size());
}

void AngleDiffusionSolver::solve(double* F, const double* sig_trg,
                                  int nyz, int Ng, double dt, double density, cudaStream_t stream) {
    int batch = Ng * nyz;
    int total = batch * n_angle;
    if (batch <= 0) return;

    if (Nom <= 32 && Nmu <= 32) {
        if (d_F_tmp.getSize() < static_cast<size_t>(total)) {
            d_F_tmp.allocate(static_cast<size_t>(total));
        }
        if (d_dst_coeff.getSize() < static_cast<size_t>(total)) {
            d_dst_coeff.allocate(static_cast<size_t>(total));
        }
        if (d_sine_coeff.getSize() < static_cast<size_t>(Ng) * n_angle) {
            d_sine_coeff.allocate(static_cast<size_t>(Ng) * n_angle);
        }
        buildDiffusionFactorKernel<<<(Ng * n_angle + 255) / 256, 256, 0, stream>>>(
            d_diffusion_factor.data(), sig_trg,
            Nom, Nmu, n_angle, Ng, cached_du, cached_dv, dt, density
        );
        CUDA_CHECK(cudaGetLastError());

        double norm = 4.0 / ((Nom + 1.0) * (Nmu + 1.0));
        complexToRealKernel<<<(Ng * n_angle + 255) / 256, 256, 0, stream>>>(
            d_sine_coeff.data(),
            d_diffusion_factor.data(),
            norm,
            Ng * n_angle
        );
        CUDA_CHECK(cudaGetLastError());

        int block_size = 256;
        int grid_size = (total + block_size - 1) / block_size;
        sineForwardKernel<<<grid_size, block_size, 0, stream>>>(
            F,
            d_dst_coeff.data(),
            d_sine_coeff.data(),
            d_sine_u.data(),
            d_sine_v.data(),
            Nom,
            Nmu,
            n_angle,
            nyz,
            Ng
        );
        CUDA_CHECK(cudaGetLastError());
        sineInverseKernel<<<grid_size, block_size, 0, stream>>>(
            d_dst_coeff.data(),
            d_F_tmp.data(),
            d_sine_u.data(),
            d_sine_v.data(),
            Nom,
            Nmu,
            n_angle,
            nyz,
            Ng
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(F, d_F_tmp.data(),
                                   static_cast<size_t>(total) * sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream));
        return;
    }
    
    if (planned_batch != batch || planned_ng != Ng || planned_nyz != nyz) {
        if (plan_forward) cufftDestroy(plan_forward);
        if (plan_inverse) cufftDestroy(plan_inverse);

        int rank = 2;
        int n[] = {extended_nmu, extended_nom};
        int inembed[] = {extended_nmu, extended_nom};
        int onembed[] = {extended_nmu, extended_nom};
        int istride = 1;
        int ostride = 1;
        int idist = extended_angle;
        int odist = extended_angle;

        CUFFT_CHECK(cufftPlanMany(&plan_forward, rank, n,
                                  inembed, istride, idist,
                                  onembed, ostride, odist,
                                  CUFFT_Z2Z, batch));
        CUFFT_CHECK(cufftPlanMany(&plan_inverse, rank, n,
                                  inembed, istride, idist,
                                  onembed, ostride, odist,
                                  CUFFT_Z2Z, batch));

        d_fft_buffer.allocate(static_cast<size_t>(batch) * extended_angle);
        d_sine_buffer.allocate(static_cast<size_t>(batch) * extended_angle);
        d_diffusion_factor.allocate(static_cast<size_t>(Ng) * n_angle);
        planned_batch = batch;
        planned_ng = Ng;
        planned_nyz = nyz;
    }

    cufftSetStream(plan_forward, stream);
    cufftSetStream(plan_inverse, stream);

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    long long ext_total = static_cast<long long>(batch) * extended_angle;
    int ext_grid_size = static_cast<int>((ext_total + block_size - 1) / block_size);

    buildDiffusionFactorKernel<<<(Ng * n_angle + block_size - 1) / block_size,
                                 block_size, 0, stream>>>(
        d_diffusion_factor.data(), sig_trg,
        Nom, Nmu, n_angle, Ng, cached_du, cached_dv, dt, density
    );
    CUDA_CHECK(cudaGetLastError());

    zeroComplexKernel<<<ext_grid_size, block_size, 0, stream>>>(
        d_fft_buffer.data(), ext_total
    );
    CUDA_CHECK(cudaGetLastError());

    packOddExtensionKernel<<<grid_size, block_size, 0, stream>>>(
        d_fft_buffer.data(), F, Nom, Nmu, extended_nom, extended_nmu, batch
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecZ2Z(
        plan_forward, d_fft_buffer.data(), d_fft_buffer.data(), CUFFT_FORWARD
    ));

    multiplyOddSineModesKernel<<<grid_size, block_size, 0, stream>>>(
        d_fft_buffer.data(), d_diffusion_factor.data(), Nom, Nmu,
        extended_nom, extended_nmu, n_angle, nyz, batch
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecZ2Z(
        plan_inverse, d_fft_buffer.data(), d_fft_buffer.data(), CUFFT_INVERSE
    ));

    double scale = 1.0 / static_cast<double>(extended_angle);
    unpackOddExtensionKernel<<<grid_size, block_size, 0, stream>>>(
        F, d_fft_buffer.data(), Nom, Nmu, extended_nom, extended_nmu, batch, scale
    );
    CUDA_CHECK(cudaGetLastError());
}
