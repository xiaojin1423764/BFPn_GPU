// src/energy_solver.cuh
#ifndef ENERGY_SOLVER_CUH
#define ENERGY_SOLVER_CUH

#include "utils.cuh"

struct EnergyTiming {
    double copy_in_seconds = 0.0;
    double legacy_kernel_seconds = 0.0;
    double precompute_seconds = 0.0;
    double recurrence_seconds = 0.0;
    double copy_out_seconds = 0.0;
    double secondary_source_seconds = 0.0;
    double secondary_update_seconds = 0.0;
};

// PCR (Parallel Cyclic Reduction) 三对角求解器
template<int BLOCK_SIZE>
class PCRSolver {
public:
    static void solve(
        double* x,
        const double* a, const double* b, const double* c, const double* d,
        int n_systems, int n_equations,
        cudaStream_t stream = 0
    );
};

// 能量沉积主求解器
class EnergyDepositionSolver {
private:
    int max_batch;
    int n_equations;  // Ng
    
    //  pitched 存储的三对角矩阵
    PitchedArray<double> d_a;  // 下对角线
    PitchedArray<double> d_b;  // 主对角线
    PitchedArray<double> d_c;  // 上对角线
    PitchedArray<double> d_d;  // 右端项
    PitchedArray<double> d_x;  // 解
    
    // 工作缓冲区
    DeviceArray<double> d_rhs_buffer;
    DeviceArray<double> d_F_old;
    DeviceArray<double> d_f_old;
    DeviceArray<double> d_F_new;
    DeviceArray<double> d_f_new;
    DeviceArray<double> d_F_from_f;
    DeviceArray<double> d_F_from_hi;
    DeviceArray<double> d_rhs4;
    DeviceArray<double> d_rhs_high_coeff;
    DeviceArray<double> d_coe;
    DeviceArray<double> d_collision_F_temp;
    DeviceArray<double> d_collision_f_temp;
    
    // cuBLAS句柄
    cublasHandle_t cublas_handle;

public:
    EnergyDepositionSolver(int max_batch_size, int n_eqs, cublasHandle_t handle);
    ~EnergyDepositionSolver() = default;
    
    // 构建并求解三对角系统
    void solve(
        double* F, double* f_F,
        const double* F_half, const double* f_F_half,
        const double* S_s, const double* sigma_c, const double* T_c,
        double dt, double dg, double density,
        int nyz, int NmuNom,
        bool is_secondary, const double* source_term,
        const double* source_term_f = nullptr,
        bool use_legacy = false,
        bool enable_straggling = false,
        cudaStream_t stream = 0,
        EnergyTiming* timing = nullptr
    );

    void solveSecondaryFromPrimary(
        double* F_secondary, double* f_secondary,
        const double* F_primary, const double* f_primary,
        const double* ker_e1, const double* ker_e2, const double* ker_v,
        const int* ker_e_begin, const int* ker_e_end,
        const double* S_s, const double* sigma_c, const double* T_c,
        double dt, double dg, double density,
        int nyz, int NmuNom,
        bool use_legacy = false,
        bool enable_straggling = false,
        cudaStream_t stream = 0,
        EnergyTiming* timing = nullptr
    );
    
    // 批量求解
    void solveBatch(
        double* F, double* f_F,
        const double* F_half, const double* f_F_half,
        const double* S_s, const double* sigma_c, const double* T_c,
        double dt, double dg, double density,
        int nyz, int NmuNom, int batch_start, int batch_size,
        bool is_secondary, const double* source_term,
        const double* source_term_f = nullptr,
        cudaStream_t stream = 0
    );

private:
    void buildSystem(
        double* a, double* b, double* c, double* d,
        const double* F_half, const double* f_F_half,
        const double* S_s, const double* sigma_c, const double* T_c,
        double dt, double dg,
        int nyz, int pos, int ang, int Ng,
        bool is_secondary, const double* source_term
    );
};

#endif
