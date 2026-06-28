// src/angle_diffusion.cuh
#ifndef ANGLE_DIFFUSION_CUH
#define ANGLE_DIFFUSION_CUH

#include "utils.cuh"
#include <cufft.h>

// 角度扩散求解器 (Fokker-Planck项)
class AngleDiffusionSolver {
private:
    int Nmu, Nom;
    int n_angle;
    int planned_batch;
    int planned_ng;
    int planned_nyz;
    double cached_du;
    double cached_dv;
    
    // FFT计划
    cufftHandle plan_forward;
    cufftHandle plan_inverse;
    
    // 设备缓冲区
    DeviceArray<cufftDoubleComplex> d_fft_buffer;
    DeviceArray<cufftDoubleComplex> d_diffusion_factor;
    
    // cuBLAS句柄
    cublasHandle_t cublas_handle;

public:
    AngleDiffusionSolver(int nmu, int nom, cublasHandle_t handle);
    ~AngleDiffusionSolver();
    
    // 初始化扩散系数 (包含sig_trg)
    void initializeDiffusionFactor(const double* sig_trg, double du, double dv, double dt);
    
    // 执行角度扩散: 对每个空间点和能量层
    void solve(double* F, const double* sig_trg, 
               int nyz, int Ng, double dt, cudaStream_t stream = 0);
    
    // 批量处理版本
    void solveBatch(double* F, const double* sig_trg,
                    int nyz, int Ng, double dt, 
                    int batch_size, cudaStream_t stream = 0);
};

#endif
