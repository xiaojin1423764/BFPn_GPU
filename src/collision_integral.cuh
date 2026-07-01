// src/collision_integral.cuh
#ifndef COLLISION_INTEGRAL_CUH
#define COLLISION_INTEGRAL_CUH

#include "utils.cuh"
#include <cusparse.h>
#include <vector>

// 基于稀疏矩阵的碰撞积分
class CollisionIntegral {
private:
    cusparseHandle_t cusparse_handle;
    
    // 角度核 (CSR格式，每层能量一个)
    struct SparseLayer {
        DeviceArray<double> values;
        DeviceArray<int> row_ptr;
        DeviceArray<int> col_idx;
        int nnz;
    };
    std::vector<SparseLayer> angle_kernels;
    
    // 能量核 (密集)
    DeviceArray<double> d_ker_e1;
    DeviceArray<double> d_ker_e2;
    
    // 工作缓冲区
    DeviceArray<double> d_F_temp_reordered;
    DeviceArray<double> d_F_tempp;
    DeviceArray<char> d_cusparse_buffer;
    size_t cusparse_buffer_size;
    
    int n_angle;
    int Ng;
    int NmuNom;

public:
    CollisionIntegral(int nmu, int nom, int ng, cusparseHandle_t handle);
    ~CollisionIntegral();
    
    // 从密集矩阵构建稀疏角度核
    void buildFromDense(const double* ker_v_host, double threshold);
    
    // 执行碰撞积分
    void compute(
        double* F_out, const double* F_in,
        int nyz,
        cudaStream_t stream = 0
    );
    
    // 计算二次质子源项
    void computeSource(
        double* source,
        const double* F_primary, const double* f_F_primary,
        const double* F_secondary, const double* f_F_secondary,
        const double* sigma_c,
        int nyz,
        cudaStream_t stream = 0
    );

private:
    void reorderForComputation(
        double* out, const double* in,
        int nyz, cudaStream_t stream
    );
    
    void reorderFromComputation(
        double* out, const double* in,
        int nyz, cudaStream_t stream
    );
};

#endif
