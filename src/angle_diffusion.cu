// src/angle_diffusion.cu
#include "angle_diffusion.cuh"
#include "../include/config.hpp"

// Kernel: 构建扩散因子 (频域)
__global__ void buildDiffusionFactorKernel(
    cufftDoubleComplex* factor,
    const double* sig_trg,
    int Nom, int Nmu, int n_angle,
    double du, double dv, double dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_angle) return;
    
    int i = idx % Nom;  // u index
    int j = idx / Nom;  // v index
    
    // 计算频率 (假设标准FFT顺序)
    double ku = (i < Nom/2) ? i : i - Nom;
    double kv = (j < Nmu/2) ? j : j - Nmu;
    
    // 归一化频率
    ku *= 2.0 * Physics::PI / Nom;
    kv *= 2.0 * Physics::PI / Nmu;
    
    // 扩散因子: (2/dt + D*k^2) / (2/dt - D*k^2)
    // 其中 D = sig_trg / (4*du*dv)
    double D = sig_trg[0] / (4.0 * du * du);  // 简化，实际需要按能量索引
    
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
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    double a = data[idx].x;
    double b = data[idx].y;
    double c = factor[idx].x;
    double d = factor[idx].y;
    
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
    : Nmu(nmu), Nom(nom), n_angle(nmu * nom), cublas_handle(handle) {
    
    // 创建2D复数FFT计划，避免实数FFT半谱尺寸和原地布局约束
    CUFFT_CHECK(cufftPlan2d(&plan_forward, Nmu, Nom, CUFFT_Z2Z));
    CUFFT_CHECK(cufftPlan2d(&plan_inverse, Nmu, Nom, CUFFT_Z2Z));
    
    // 分配缓冲区
    d_fft_buffer.allocate(n_angle);
    d_diffusion_factor.allocate(n_angle);
}

AngleDiffusionSolver::~AngleDiffusionSolver() {
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
}

void AngleDiffusionSolver::initializeDiffusionFactor(
    const double* sig_trg, double du, double dv, double dt) {
    
    // 对每个能量层初始化 (这里简化，使用第一个)
    buildDiffusionFactorKernel<<<(n_angle + 255) / 256, 256>>>(
        d_fft_buffer.data(), sig_trg,
        Nom, Nmu, n_angle, du, dv, dt
    );
    
    d_diffusion_factor.copyFromDevice(d_fft_buffer.data(), n_angle);
}

void AngleDiffusionSolver::solve(double* F, const double* sig_trg,
                                  int nyz, int Ng, double dt, cudaStream_t stream) {
    
    // 设置流
    cufftSetStream(plan_forward, stream);
    cufftSetStream(plan_inverse, stream);
    
    for (int ek = 0; ek < Ng; ek++) {
        double* F_ek = F + ek * nyz * n_angle;
        
        for (int pos = 0; pos < nyz; pos++) {
            double* F_slice = F_ek + pos * n_angle;
            
            realToComplexKernel<<<(n_angle + 255) / 256, 256, 0, stream>>>(
                d_fft_buffer.data(), F_slice, n_angle
            );
            CUDA_CHECK(cudaGetLastError());

            // 前向FFT
            CUFFT_CHECK(cufftExecZ2Z(
                plan_forward, d_fft_buffer.data(), d_fft_buffer.data(), CUFFT_FORWARD
            ));
            
            // 乘以扩散因子
            complexMultiplyInPlaceKernel<<<(n_angle + 255) / 256, 256, 0, stream>>>(
                d_fft_buffer.data(), d_diffusion_factor.data(), n_angle
            );
            
            // 逆FFT
            CUFFT_CHECK(cufftExecZ2Z(
                plan_inverse, d_fft_buffer.data(), d_fft_buffer.data(), CUFFT_INVERSE
            ));

            // 归一化并写回实数部分 (cuFFT不进行归一化)
            double scale = 1.0 / n_angle;
            complexToRealKernel<<<(n_angle + 255) / 256, 256, 0, stream>>>(
                F_slice, d_fft_buffer.data(), scale, n_angle
            );
            CUDA_CHECK(cudaGetLastError());
        }
    }
}
