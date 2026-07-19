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
#include <unordered_map>
#include <utility>

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
    double beam_energy;
    double density;
    double energy_density;
    std::string material_name;
    bool lite_memory;
    bool streaming_full;
    bool primary_only;
    bool legacy_energy;
    bool eq15_straggling;
    bool save_energy_moments;
    bool energy_only;
    bool no_transport;
    bool no_angle;
    bool no_spatial_clipping;
    bool no_zero_chunk_skip;
    bool no_fused_stream_boundary;
    bool profile_steps;
    bool discrete_angle_delta;
    bool normalize_initial_mass;
    bool trapezoidal_yz;
    bool dose_catastrophic_loss;
    bool calibrate_cross_sections;
    int streaming_lane_chunk;
    int streaming_energy_chunk;
    std::string streaming_dir;
    std::vector<double> spot_depths;
    std::string spot_prefix;
    int idd_stride;
};

struct StepProfile {
    double energy_seconds = 0.0;
    double transport_seconds = 0.0;
    double angle_seconds = 0.0;
    double diagnostics_seconds = 0.0;
    double total_seconds = 0.0;
    EnergyTiming energy_timing;
    double primary_energy_solve_seconds = 0.0;
    size_t streaming_chunks_total = 0;
    size_t streaming_chunks_skipped = 0;
    int steps = 0;
};

class HostDoubleBuffer {
private:
    double* ptr;
    size_t capacity;
    bool pinned;
    std::vector<double> pageable;

    void releasePinned();

public:
    HostDoubleBuffer();
    ~HostDoubleBuffer();
    HostDoubleBuffer(const HostDoubleBuffer&) = delete;
    HostDoubleBuffer& operator=(const HostDoubleBuffer&) = delete;

    void resize(size_t count);
    double* data();
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
    DeviceArray<double> d_reduction_sums;  // GPU归约临时缓冲
    DeviceArray<double> d_reduction_scratch;
    DeviceArray<double> d_energy_moments;   // [Ng][2] energy moment diagnostics
    DeviceArray<double> d_spot_plane;      // YZ dose plane output buffer
    
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
    DeviceArray<double> d_ker_v;                      // 角度核 [Ng][Nangle][Nangle]
    DeviceArray<int> d_ker_v_col_ptr, d_ker_v_in_idx; // streaming source sparse angle kernel
    DeviceArray<double> d_ker_v_sparse_values;
    DeviceArray<int> d_ker_e_begin, d_ker_e_end;      // 每个输出能量的非零源能量范围

    // Out-of-core streaming state for full-grid development.
    DeviceArray<double> d_stream_F, d_stream_f_F;
    DeviceArray<double> d_stream_F1, d_stream_f_F1;
    DeviceArray<double> d_stream_primary_F, d_stream_primary_f_F;
    DeviceArray<double> d_stream_old_F, d_stream_old_f_F;
    DeviceArray<double> d_stream_carry_old_F, d_stream_carry_old_f;
    DeviceArray<double> d_stream_carry_new_F, d_stream_carry_new_f;
    std::string stream_F_path, stream_f_F_path;
    std::string stream_F1_path, stream_f_F1_path;
    std::string stream_source_path;
    mutable std::unordered_map<std::string, int> stream_fds;
    std::unordered_map<std::string, int> stream_max_active_energy;
    bool stream_source_energy_nonincreasing = true;
    HostDoubleBuffer h_stream_lane;
    HostDoubleBuffer h_stream_f_lane;
    HostDoubleBuffer h_stream_primary_lane;
    HostDoubleBuffer h_stream_primary_f_lane;
    HostDoubleBuffer h_stream_chunk;
    HostDoubleBuffer h_stream_f_chunk;

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
    void saveResults(const std::vector<double>& values, const std::string& filename);
    void saveDepthResults(const std::vector<std::pair<double, double>>& values,
                          const std::string& filename);
    void saveEnergyMoments(const std::vector<double>& values, const std::string& filename);

private:
    void allocateMemory();
    void preflightMemory() const;
    void freeMemory();
    void initializeKernels();
    void initializePhysicsQuantities();
    
    // 单步求解
    void stepPrimary(int ping, double t, StepProfile* profile = nullptr);
    void stepPrimaryEnergyOnly(StepProfile* profile = nullptr);
    void stepSecondary(int ping, int pong, double t, StepProfile* profile = nullptr);
    void stepLitePrimary();
    void solveStreamingFull(double tFinal);
    void initializeStreamingStore();
    void cleanupStreamingStore();
    void readStore(const std::string& path, size_t element_offset,
                   double* dst, size_t count) const;
    void writeStore(const std::string& path, size_t element_offset,
                    const double* src, size_t count) const;
    int getStoreFd(const std::string& path) const;
    const double* reduceDevicePartials(size_t count, cudaStream_t stream = 0);
    double reducePartialsToHost(size_t count, cudaStream_t stream = 0);
    void streamingEnergyStep(const std::string& F_path,
                             const std::string& f_F_path,
                             const std::string* primary_F_path,
                             const std::string* primary_f_F_path,
                             bool is_secondary,
                             double dt,
                             double* idd_accum = nullptr,
                             bool reuse_secondary_source = false,
                             const std::string* secondary_source_cache_path = nullptr,
                             bool read_cached_secondary_source = false,
                             bool write_cached_secondary_source = false,
                             EnergyTiming* timing = nullptr);
    void streamingTransportStep(const std::string& F_path, double dt, bool preserve_sign = false);
    void streamingTransportPairStep(const std::string& F_path,
                                    const std::string& f_F_path,
                                    double dt);
    void streamingAngleStep(const std::string& F_path, double dt);
    void streamingAnglePairStep(const std::string& F_path,
                                const std::string& f_F_path,
                                double dt);
    void streamingTransportAngleTransportPairStep(const std::string& F_path,
                                                  const std::string& f_F_path,
                                                  double transport_dt,
                                                  double angle_dt,
                                                  StepProfile* profile = nullptr,
                                                  bool fuse_energy = false,
                                                  double energy_dt = 0.0,
                                                  double* idd_accum = nullptr);
    double computeScalarDoseProxy();
    double computeEnergyFlux(const double* F, const double* f_F);
    double computeIntegratedDepthDose();
    double computeIntegratedDepthDoseLite();
    std::vector<double> computeEnergyMoments();
    std::vector<double> computeSpotDosePlane();
    void saveSpotPlane(double requested_depth, double actual_depth);
    void saveSpotPlaneStreaming(double requested_depth, double actual_depth);
    double computeEnergyFluxStreaming(const std::string& F_path,
                                      const std::string& f_F_path);
    double computeIntegratedDepthDoseStreaming();
    
    // 辅助函数
    void computeCrossSections();
    void computeStoppingPower();
    void computeStraggling();
    void computeTransportCrossSection();
};

#endif
