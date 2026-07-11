// src/spatial_transport.cuh
#ifndef SPATIAL_TRANSPORT_CUH
#define SPATIAL_TRANSPORT_CUH

#include "utils.cuh"
#include <cufft.h>

// 空间传输求解器 (使用FFT)
class SpatialTransportSolver {
private:
    int Ny, Nz;
    int nyz;
    double dy, dz;
    
    cufftHandle plan_forward;
    cufftHandle plan_inverse;
    
    DeviceArray<cufftDoubleComplex> d_fft_buffer;
    DeviceArray<cufftDoubleComplex> d_transport_factor;
    
    // 波数网格
    DeviceArray<double> d_ky, d_kz;
    // 显式有限差分用的临时数组
    DeviceArray<double> d_F_pred;
    DeviceArray<double> d_flux_old;
    DeviceArray<double> d_flux_pred;

public:
    SpatialTransportSolver(int ny, int nz, double dy, double dz);
    ~SpatialTransportSolver();
    
    void initializeWaveNumbers();
    void initializeTransportFactor(const double* Omend, 
                                    double dt, int NmuNom);
    
    // Eq. (18): run one YZ transport plane for every (energy, angle) pair.
    void solve(double* F, const double* Omend,
               int Ng, int NmuNom, double dt, 
               cudaStream_t stream = 0,
               bool preserve_sign = false);
    void solveEq18(double* F, const double* Omend,
                   int Ng, int NmuNom, double dt,
                   cudaStream_t stream = 0,
                   bool preserve_sign = false);
};

#endif
