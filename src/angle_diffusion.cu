// src/angle_diffusion.cu
#include "angle_diffusion.cuh"
#include "../include/config.hpp"

// Kernel: 构建扩散因子 (频域)
__global__ void buildDiffusionFactorKernel(
    cufftDoubleComplex* factor,
    const double* sig_trg,
    int Nom, int Nmu, int n_angle, int Ng,
    double du, double dv, double dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_angle * Ng;
    if (idx >= total) return;
    
    int angle_idx = idx % n_angle;
    int ek = idx / n_angle;
    int i = angle_idx % Nom;  // u index
    int j = angle_idx / Nom;  // v index
    
    // 计算频率 (假设标准FFT顺序)
    double ku = (i < Nom/2) ? i : i - Nom;
    double kv = (j < Nmu/2) ? j : j - Nmu;
    
    // 归一化频率
    ku *= 2.0 * Physics::PI / Nom;
    kv *= 2.0 * Physics::PI / Nmu;
    
    // 扩散因子: (2/dt + D*k^2) / (2/dt - D*k^2)
    // 其中 D = sig_trg / (4*du*dv)
    double D = sig_trg[ek] / (4.0 * du * du);
    
    double k2 = ku*ku + kv*kv;
    double numerator = 2.0/dt + D * k2;
    double denominator = 2.0/dt - D * k2;
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

AngleDiffusionSolver::AngleDiffusionSolver(int nmu, int nom, cublasHandle_t handle)
    : Nmu(nmu), Nom(nom), n_angle(nmu * nom), planned_batch(0),
      planned_ng(0), planned_nyz(0),
      cached_du(0.0), cached_dv(0.0),
      plan_forward(0), plan_inverse(0), cublas_handle(handle) {
    
    d_fft_buffer.allocate(n_angle);
    d_diffusion_factor.allocate(n_angle);
}

AngleDiffusionSolver::~AngleDiffusionSolver() {
    if (plan_forward) cufftDestroy(plan_forward);
    if (plan_inverse) cufftDestroy(plan_inverse);
}

void AngleDiffusionSolver::initializeDiffusionFactor(
    const double* sig_trg, double du, double dv, double dt) {
    cached_du = du;
    cached_dv = dv;
    
    // solve() builds the energy-dependent factor once Ng is known.
    buildDiffusionFactorKernel<<<(n_angle + 255) / 256, 256>>>(
        d_fft_buffer.data(), sig_trg,
        Nom, Nmu, n_angle, 1, du, dv, dt
    );
    
    d_diffusion_factor.copyFromDevice(d_fft_buffer.data(), n_angle);
}

void AngleDiffusionSolver::solve(double* F, const double* sig_trg,
                                  int nyz, int Ng, double dt, cudaStream_t stream) {
    int batch = Ng * nyz;
    int total = batch * n_angle;
    if (batch <= 0) return;
    
    if (planned_batch != batch || planned_ng != Ng || planned_nyz != nyz) {
        if (plan_forward) cufftDestroy(plan_forward);
        if (plan_inverse) cufftDestroy(plan_inverse);

        int rank = 2;
        int n[] = {Nmu, Nom};
        int inembed[] = {Nmu, Nom};
        int onembed[] = {Nmu, Nom};
        int istride = 1;
        int ostride = 1;
        int idist = n_angle;
        int odist = n_angle;

        CUFFT_CHECK(cufftPlanMany(&plan_forward, rank, n,
                                  inembed, istride, idist,
                                  onembed, ostride, odist,
                                  CUFFT_Z2Z, batch));
        CUFFT_CHECK(cufftPlanMany(&plan_inverse, rank, n,
                                  inembed, istride, idist,
                                  onembed, ostride, odist,
                                  CUFFT_Z2Z, batch));

        d_fft_buffer.allocate(static_cast<size_t>(total));
        d_diffusion_factor.allocate(static_cast<size_t>(Ng) * n_angle);
        planned_batch = batch;
        planned_ng = Ng;
        planned_nyz = nyz;
    }

    cufftSetStream(plan_forward, stream);
    cufftSetStream(plan_inverse, stream);

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    buildDiffusionFactorKernel<<<(Ng * n_angle + block_size - 1) / block_size,
                                 block_size, 0, stream>>>(
        d_diffusion_factor.data(), sig_trg,
        Nom, Nmu, n_angle, Ng, cached_du, cached_dv, dt
    );
    CUDA_CHECK(cudaGetLastError());

    realToComplexKernel<<<grid_size, block_size, 0, stream>>>(
        d_fft_buffer.data(), F, total
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecZ2Z(
        plan_forward, d_fft_buffer.data(), d_fft_buffer.data(), CUFFT_FORWARD
    ));

    complexMultiplyInPlaceKernel<<<grid_size, block_size, 0, stream>>>(
        d_fft_buffer.data(), d_diffusion_factor.data(), n_angle, nyz, total
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecZ2Z(
        plan_inverse, d_fft_buffer.data(), d_fft_buffer.data(), CUFFT_INVERSE
    ));

    double scale = 1.0 / n_angle;
    complexToRealKernel<<<grid_size, block_size, 0, stream>>>(
        F, d_fft_buffer.data(), scale, total
    );
    CUDA_CHECK(cudaGetLastError());
}
