// src/collision_integral.cu
#include "collision_integral.cuh"
#include "../include/config.hpp"

// 提取非零元素的kernel
__global__ void extractSparseStructureKernel(
    const double* dense,
    int* nnz_per_row,
    int nrows, int ncols,
    double threshold
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    
    int count = 0;
    for (int col = 0; col < ncols; col++) {
        if (fabs(dense[row * ncols + col]) > threshold) {
            count++;
        }
    }
    nnz_per_row[row] = count;
}

__global__ void fillSparseValuesKernel(
    const double* dense,
    double* values, int* col_idx,
    int* row_ptr, int nrows, int ncols,
    double threshold
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    
    int write_pos = row_ptr[row];
    for (int col = 0; col < ncols; col++) {
        double val = dense[row * ncols + col];
        if (fabs(val) > threshold) {
            values[write_pos] = val;
            col_idx[write_pos] = col;
            write_pos++;
        }
    }
}

__global__ void buildLocalSecondarySourceKernel(
    double* source,
    const double* F_primary,
    const double* f_F_primary,
    const double* sigma_c,
    int Ng,
    int nyz,
    int NmuNom
) {
    long long total = static_cast<long long>(Ng) * nyz * NmuNom;
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    long long n_per_E = static_cast<long long>(nyz) * NmuNom;
    int ek = static_cast<int>(gid / n_per_E);
    double primary = F_primary[gid] + (f_F_primary ? f_F_primary[gid] : 0.0);
    source[gid] = max(sigma_c[ek], 0.0) * max(primary, 0.0);
}

CollisionIntegral::CollisionIntegral(int nmu, int nom, int ng, cusparseHandle_t handle)
    : NmuNom(nmu * nom), Ng(ng), n_angle(nmu * nom), cusparse_handle(handle),
      cusparse_buffer_size(0) {
    
    d_ker_e1.allocate(Ng * Ng);
    d_ker_e2.allocate(Ng * Ng);
}

CollisionIntegral::~CollisionIntegral() {
    // 自动释放
}

void CollisionIntegral::buildFromDense(const double* ker_v_host, double threshold) {
    angle_kernels.resize(Ng - 1);
    
    // 临时设备缓冲区
    DeviceArray<double> d_dense(n_angle * n_angle);
    
    for (int ek = 0; ek < Ng - 1; ek++) {
        // 复制到设备
        d_dense.copyFromHost(ker_v_host + ek * n_angle * n_angle, n_angle * n_angle);
        
        // 阈值处理
        // ...
        
        // 计算每行非零元素数
        DeviceArray<int> d_nnz_per_row(n_angle);
        extractSparseStructureKernel<<<(n_angle + 255) / 256, 256>>>(
            d_dense.data(), d_nnz_per_row.data(),
            n_angle, n_angle, threshold
        );
        
        // 前缀和得到row_ptr
        std::vector<int> h_nnz_per_row(n_angle);
        d_nnz_per_row.copyToHost(h_nnz_per_row.data(), n_angle);
        
        std::vector<int> h_row_ptr(n_angle + 1);
        h_row_ptr[0] = 0;
        for (int i = 0; i < n_angle; i++) {
            h_row_ptr[i + 1] = h_row_ptr[i] + h_nnz_per_row[i];
        }
        int total_nnz = h_row_ptr[n_angle];
        
        // 分配稀疏矩阵存储
        angle_kernels[ek].values.allocate(total_nnz);
        angle_kernels[ek].col_idx.allocate(total_nnz);
        angle_kernels[ek].row_ptr.allocate(n_angle + 1);
        angle_kernels[ek].nnz = total_nnz;
        
        angle_kernels[ek].row_ptr.copyFromHost(h_row_ptr.data(), n_angle + 1);
        
        // 填充数值
        fillSparseValuesKernel<<<(n_angle + 255) / 256, 256>>>(
            d_dense.data(),
            angle_kernels[ek].values.data(),
            angle_kernels[ek].col_idx.data(),
            angle_kernels[ek].row_ptr.data(),
            n_angle, n_angle, threshold
        );
    }
}

void CollisionIntegral::compute(double* F_out, const double* F_in,
                                 int nyz, cudaStream_t stream) {
    // 重排序: [nyz][Ng][NmuNom] -> [Ng][nyz][NmuNom]
    reorderForComputation(d_F_temp_reordered.data(), F_in, nyz, stream);
    
    // 对每个能量层执行稀疏矩阵乘法
    // F_tempp[ek] = F_temp[ek] * ker_v[ek] * ker_e1[ek]
    
    // 使用cusparseSpMM
    // ...
    
    // 重排序回来
    reorderFromComputation(F_out, d_F_tempp.data(), nyz, stream);
}

void CollisionIntegral::computeSource(
    double* source,
    const double* F_primary, const double* f_F_primary,
    const double* F_secondary, const double* f_F_secondary,
    const double* sigma_c,
    int nyz,
    cudaStream_t stream) {
    long long total = static_cast<long long>(nyz) * NmuNom * Ng;
    int block_size = 256;
    int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    buildLocalSecondarySourceKernel<<<grid_size, block_size, 0, stream>>>(
        source,
        F_primary,
        f_F_primary,
        sigma_c,
        Ng,
        nyz,
        NmuNom
    );
    CUDA_CHECK(cudaGetLastError());
}

void CollisionIntegral::reorderForComputation(
    double* out, const double* in,
    int nyz, cudaStream_t stream) {
    
    // Kernel实现重排序
    // ...
}

void CollisionIntegral::reorderFromComputation(
    double* out, const double* in,
    int nyz, cudaStream_t stream) {
    
    // Kernel实现重排序
    // ...
}
