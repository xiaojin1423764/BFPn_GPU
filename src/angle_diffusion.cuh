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
    int extended_nmu;
    int extended_nom;
    int extended_angle;
    double cached_du;
    double cached_dv;
    double cached_density;
    
    // FFT计划
    cufftHandle plan_forward;
    cufftHandle plan_inverse;
    
    // 设备缓冲区
    DeviceArray<cufftDoubleComplex> d_fft_buffer;
    DeviceArray<cufftDoubleComplex> d_diffusion_factor;
    DeviceArray<cufftDoubleComplex> d_sine_buffer;
    DeviceArray<double> d_F_tmp;
    DeviceArray<double> d_sine_coeff;
    DeviceArray<double> d_sine_u;
    DeviceArray<double> d_sine_v;
    DeviceArray<double> d_dst_coeff;
    
    // cuBLAS句柄
    cublasHandle_t cublas_handle;

public:
    AngleDiffusionSolver(int nmu, int nom, cublasHandle_t handle);
    ~AngleDiffusionSolver();
    
    // 初始化扩散系数 (包含sig_trg)
    void initializeDiffusionFactor(const double* sig_trg, double du, double dv, double dt,
                                   double density);
    
    // Eq. (19): run one angular UV plane for every (energy, YZ cell) pair.
    void solve(double* F, const double* sig_trg, 
               int nyz, int Ng, double dt, double density, cudaStream_t stream = 0);
    void solveEq19(double* F, const double* sig_trg,
                   int nyz, int Ng, double dt, double density,
                   cudaStream_t stream = 0);
    
    // 批量处理版本
    void solveBatch(double* F, const double* sig_trg,
                    int nyz, int Ng, double dt, 
                    int batch_size, cudaStream_t stream = 0);
};

#endif
