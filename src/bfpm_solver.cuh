// src/bfpm_solver.cuh
#ifndef BFPM_SOLVER_CUH
#define BFPM_SOLVER_CUH

#include "utils.cuh"
#include "angle_diffusion.cuh"
#include "spatial_transport.cuh"
#include "energy_solver.cuh"
#include "collision_integral.cuh"
#include "../include/config.hpp"

#include <cublas_v2.h>
#include <cusparse.h>
#include <memory>
#include <vector>
#include <string>

// 网格参数结构体
struct GridParams {
    int Nx, Ny, Nz, Ng, Nmu, Nom;
    double Lx, Ly, Lz, Lu, Lv, Lg;
    double dt, dy, dz, dg, du, dv;
    
    void computeDeltas();
};

// 物理参数
struct PhysicsParams {
    double sig_E;
    double a_1, a_2;
    double C_c;
};

// BFPn求解器主类
class BFPnSolver {
private:
    GridParams grid;
    PhysicsParams phys;
    
    // CUDA句柄
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    
    // 子求解器
    std::unique_ptr<AngleDiffusionSolver> angle_solver;
    std::unique_ptr<SpatialTransportSolver> transport_solver;
    std::unique_ptr<EnergyDepositionSolver> energy_solver;
    std::unique_ptr<CollisionIntegral> collision_solver;
    
    // 多流并行
    cudaStream_t stream_primary[2];
    cudaStream_t stream_secondary;
    cudaStream_t stream_collision;
    cudaEvent_t event_primary_ready;
    cudaEvent_t event_collision_done;
    
    // 设备数据
    DeviceArray<double> d_F, d_f_F;       // 主质子
    DeviceArray<double> d_F1, d_f_F1;     // 二次质子
    DeviceArray<double> d_F_tot, d_f_Ftot;
    DeviceArray<double> d_source;          // 碰撞源项
    
    // 物理量数组
    DeviceArray<double> d_mu;              // 截面数据 [Ng][3]
    DeviceArray<double> d_sigma_c;         // 总截面 [Ng]
    DeviceArray<double> d_S_s;             // 阻止本领 [Ng+1]
    DeviceArray<double> d_T_c;             // 能量离散 [Ng]
    DeviceArray<double> d_sig_trg;         // 传输截面 [Ng]
    DeviceArray<double> d_en;              // 能量网格 [Ng+1]
    DeviceArray<double> d_y, d_z;          // 空间网格
    DeviceArray<double> d_Omend;           // 角度方向 [NmuNom][2]
    DeviceArray<double> d_utemp, d_vtemp;  // 角度辅助
    
    // 内核数组
    DeviceArray<double> d_ker_e, d_ker_e1, d_ker_e2;  // 能量核
    DeviceArray<double> d_ker_v;                      // 角度核 (密集，后续转稀疏)

public:
    BFPnSolver(const GridParams& g, const PhysicsParams& p);
    ~BFPnSolver();
    
    // 禁止拷贝
    BFPnSolver(const BFPnSolver&) = delete;
    BFPnSolver& operator=(const BFPnSolver&) = delete;
    
    // 初始化
    void initialize(const std::string& data_path);
    void initializeDistribution();
    
    // 求解
    void solve(double tFinal);
    
    // 保存结果
    void saveResults(const std::vector<double>& dose, const std::string& filename);

private:
    void allocateMemory();
    void freeMemory();
    void initializeKernels();
    void initializePhysicsQuantities();
    
    // 单步求解
    void stepPrimary(int ping, double t);
    void stepSecondary(int ping, int pong, double t);
    void computeCollisionSource(int ping);
    double computeDose();
    
    // 辅助函数
    void computeCrossSections();
    void computeStoppingPower();
    void computeStraggling();
    void computeTransportCrossSection();
};

#endif
