// src/bfpm_solver.cu
#include "bfpm_solver.cuh"
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>

void GridParams::computeDeltas() {
    dt = Lx / Nx;
    dy = Ly / Ny;
    dz = Lz / Nz;
    dg = Lg / Ng;
    du = Lu / (Nmu - 1);
    dv = Lv / (Nom - 1);
}

BFPnSolver::BFPnSolver(const GridParams& g, const PhysicsParams& p)
    : grid(g), phys(p) {
    
    grid.computeDeltas();
    
    // 创建CUDA句柄
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
    
    // 创建流
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaStreamCreate(&stream_primary[i]));
    }
    CUDA_CHECK(cudaStreamCreate(&stream_secondary));
    CUDA_CHECK(cudaStreamCreate(&stream_collision));
    
    // 创建事件
    CUDA_CHECK(cudaEventCreate(&event_primary_ready));
    CUDA_CHECK(cudaEventCreate(&event_collision_done));
    
    // 创建子求解器
    angle_solver = std::make_unique<AngleDiffusionSolver>(
        grid.Nmu, grid.Nom, cublas_handle);
    transport_solver = std::make_unique<SpatialTransportSolver>(
        grid.Ny, grid.Nz, grid.dy, grid.dz);
    energy_solver = std::make_unique<EnergyDepositionSolver>(
        Numerics::MAX_BATCH_SIZE, grid.Ng, cublas_handle);
    collision_solver = std::make_unique<CollisionIntegral>(
        grid.Nmu, grid.Nom, grid.Ng, cusparse_handle);
    
    allocateMemory();
}

BFPnSolver::~BFPnSolver() {
    // 自动释放设备数组
    
    // 销毁流和事件
    for (int i = 0; i < 2; i++) {
        cudaStreamDestroy(stream_primary[i]);
    }
    cudaStreamDestroy(stream_secondary);
    cudaStreamDestroy(stream_collision);
    cudaEventDestroy(event_primary_ready);
    cudaEventDestroy(event_collision_done);
    
    // 销毁句柄
    cublasDestroy(cublas_handle);
    cusparseDestroy(cusparse_handle);
}

void BFPnSolver::allocateMemory() {
    size_t nyz = (grid.Ny + 1) * (grid.Nz + 1);
    size_t n_angle = grid.Nmu * grid.Nom;
    size_t total_size = nyz * grid.Ng * n_angle;
    
    // 分布函数
    d_F.allocate(total_size);
    d_f_F.allocate(total_size);
    d_F1.allocate(total_size);
    d_f_F1.allocate(total_size);
    d_F_tot.allocate(total_size);
    d_f_Ftot.allocate(total_size);
    d_source.allocate(total_size);
    
    // 网格数据
    d_y.allocate(grid.Ny + 1);
    d_z.allocate(grid.Nz + 1);
    d_en.allocate(grid.Ng + 1);
    d_Omend.allocate(n_angle * 2);
    d_utemp.allocate(n_angle);
    d_vtemp.allocate(n_angle);
    
    // 物理量
    d_mu.allocate(grid.Ng * 3);
    d_sigma_c.allocate(grid.Ng);
    d_S_s.allocate(grid.Ng + 1);
    d_T_c.allocate(grid.Ng);
    d_sig_trg.allocate(grid.Ng);
    
    // 内核
    d_ker_e.allocate(grid.Ng * grid.Ng);
    d_ker_e1.allocate(grid.Ng * grid.Ng);
    d_ker_e2.allocate(grid.Ng * grid.Ng);
    d_ker_v.allocate((grid.Ng - 1) * n_angle * n_angle);
}

void BFPnSolver::initialize(const std::string& data_path) {
    // 读取截面数据
    std::vector<double> h_mu(grid.Ng * 3);
    std::vector<double> h_sigma_c(grid.Ng);
    
    std::ifstream mu_file(data_path + "/data_cross.txt");
    for (int i = 0; i < grid.Ng && mu_file; i++) {
        mu_file >> h_mu[i*3] >> h_mu[i*3+1] >> h_mu[i*3+2];
    }
    
    std::ifstream sigma_file(data_path + "/cross_total.txt");
    for (int i = 0; i < grid.Ng && sigma_file; i++) {
        sigma_file >> h_sigma_c[i];
    }
    
    // 应用修正
    for (auto& s : h_sigma_c) s *= 0.9;
    for (int i = 401; i < 444 && i < grid.Ng; i++) {
        h_sigma_c[i] -= 0.0002 * (0.5 + 0.5 * (i - 401));
    }
    
    d_mu.copyFromHost(h_mu.data(), grid.Ng * 3);
    d_sigma_c.copyFromHost(h_sigma_c.data(), grid.Ng);
    
    // 初始化网格
    std::vector<double> h_y(grid.Ny + 1), h_z(grid.Nz + 1), h_en(grid.Ng + 1);
    for (int i = 0; i <= grid.Ny; i++) h_y[i] = i * grid.dy - 1.0;
    for (int i = 0; i <= grid.Nz; i++) h_z[i] = i * grid.dz - 1.0;
    for (int i = 0; i <= grid.Ng; i++) h_en[i] = i * grid.dg + 1.0;
    
    d_y.copyFromHost(h_y.data(), grid.Ny + 1);
    d_z.copyFromHost(h_z.data(), grid.Nz + 1);
    d_en.copyFromHost(h_en.data(), grid.Ng + 1);
    
    // 初始化角度方向
    int n_angle = grid.Nmu * grid.Nom;
    std::vector<double> h_Omend(n_angle * 2);
    std::vector<double> h_utemp(n_angle), h_vtemp(n_angle);
    std::vector<double> u(grid.Nmu), v(grid.Nom);
    
    for (int i = 0; i < grid.Nmu; i++) u[i] = -0.5 + i * grid.du;
    for (int i = 0; i < grid.Nom; i++) v[i] = -0.5 + i * grid.dv;
    
    for (int j = 0; j < grid.Nmu; j++) {
        for (int i = 0; i < grid.Nom; i++) {
            int idx = j * grid.Nom + i;
            h_Omend[idx*2] = -v[grid.Nmu - 1 - j];
            h_Omend[idx*2+1] = -u[i];
            h_utemp[idx] = 2.0 * cos(Physics::PI * i / (grid.Nom - 1)) - 2.0;
            h_vtemp[idx] = 2.0 * cos(Physics::PI * j / (grid.Nmu - 1)) - 2.0;
        }
    }
    
    d_Omend.copyFromHost(h_Omend.data(), n_angle * 2);
    d_utemp.copyFromHost(h_utemp.data(), n_angle);
    d_vtemp.copyFromHost(h_vtemp.data(), n_angle);
    
    // 计算物理量
    computeStoppingPower();
    computeStraggling();
    computeTransportCrossSection();
    
    // 初始化内核
    initializeKernels();
    
    // 初始化分布
    initializeDistribution();
    
    // 初始化子求解器
    angle_solver->initializeDiffusionFactor(d_sig_trg.data(), grid.du, grid.dv, grid.dt);
    transport_solver->initializeWaveNumbers();
    transport_solver->initializeTransportFactor(d_Omend.data(), grid.dt, n_angle);
}

void BFPnSolver::computeStoppingPower() {
    std::vector<double> h_S_s(grid.Ng + 1);
    
    for (int i = 0; i <= grid.Ng; i++) {
        double E = (i == 0) ? 1.0 : (i == grid.Ng) ? grid.Lg + 1.0 : (i - 0.5) * grid.dg + 1.0;
        
        // 相对论计算
        double beta_2 = E * (E + 2.0 * 938.3) / pow(E + 938.3, 2);
        double F_beta = log(1.02e6 * beta_2 / (1.0 - beta_2)) - beta_2 - 4.31;
        h_S_s[i] = 0.170 / beta_2 * F_beta + 0.02;
    }
    
    d_S_s.copyFromHost(h_S_s.data(), grid.Ng + 1);
}

void BFPnSolver::computeStraggling() {
    std::vector<double> h_T_c(grid.Ng);
    
    for (int i = 0; i < grid.Ng; i++) {
        double E = (i + 0.5) * grid.dg + 1.0;
        double v = sqrt(2.0 * E * Physics::EV_TO_J / Physics::MP);
        double beta = v / Physics::C_LIGHT;
        
        // 简化计算
        double eta = 8.99e18 * 4.0 * Physics::PI * pow(1.6e-19, 4) 
                    / pow(1.6e-13, 2) / 100.0;
        
        h_T_c[i] = eta * 3.34e29 * 1.2;  // 简化公式
    }
    
    d_T_c.copyFromHost(h_T_c.data(), grid.Ng);
}

void BFPnSolver::computeTransportCrossSection() {
    std::vector<double> h_sig_trg(grid.Ng);
    
    for (int i = 0; i < grid.Ng; i++) {
        double E = (i + 0.5) * grid.dg + 1.0;
        double v = sqrt(2.0 * E * Physics::EV_TO_J / Physics::MP);
        double beta = v / (3.0 * Physics::C_LIGHT);
        
        double Zt = 3;  // 目标原子序数 (锂)
        double eta = pow(Zt, 2.0/3.0) * pow(Physics::ALPHA, 2) 
                    * pow(Physics::ME / Physics::MP, 2) / (beta * beta);
        
        double mid = log((eta + 1.0) / eta) - 1.0 / (eta + 1.0);
        h_sig_trg[i] = 2.0 * Physics::PI * 3.34 * pow(Physics::HBARC, 2) 
                      / pow(Physics::ALPHA, 2) / 4.0 * Zt * Zt 
                      / (E * E) * mid / 1000.0;
    }
    
    d_sig_trg.copyFromHost(h_sig_trg.data(), grid.Ng);
}

void BFPnSolver::initializeKernels() {
    // 能量核
    std::vector<double> h_ker_e(grid.Ng * grid.Ng, 0.0);
    
    // 读取mu数据到host (简化，实际应该保存host副本)
    std::vector<double> h_mu(grid.Ng * 3);
    d_mu.copyToHost(h_mu.data(), grid.Ng * 3);
    
    for (int j = 0; j < grid.Ng - 1; j++) {
        for (int i = 0; i < grid.Ng - 1; i++) {
            if (h_mu[j*3+2] == 0) {
                h_ker_e[i*grid.Ng + j] = 0.0;
            } else if (i < j) {
                h_ker_e[i*grid.Ng + j] = 0.0;
            } else {
                h_ker_e[i*grid.Ng + j] = grid.dg / h_mu[j*3+2] 
                    * exp(-(i - j + 0.5) * grid.dg / h_mu[j*3+2]);
            }
        }
    }
    
    // 重排序和对称化 (按MATLAB逻辑)
    // ...
    
    d_ker_e.copyFromHost(h_ker_e.data(), grid.Ng * grid.Ng);
    
    // 计算ker_e1, ker_e2
    // ker_eh(1:Ng,:) = ker_e; ker_eh(Ng+1,:) = 0;
    // ker_e1 = (ker_eh(1:Ng,:) + ker_eh(2:Ng+1,:)) / 2;
    // ker_e2 = (-ker_eh(1:Ng,:) + ker_eh(2:Ng+1,:)) / 2;
    
    // 角度核 (密集计算后转稀疏)
    // ...
}

void BFPnSolver::initializeDistribution() {
    size_t nyz = (grid.Ny + 1) * (grid.Nz + 1);
    size_t n_angle = grid.Nmu * grid.Nom;
    
    std::vector<double> h_F(nyz * grid.Ng * n_angle, 0.0);
    
    // 初始化f_1 (空间-角度)
    for (int k = 0; k < n_angle; k++) {
        // 需要Omend数据
        double omega1 = 0.0, omega2 = 0.0;  // 从d_Omend读取
        
        for (int j = 0; j <= grid.Nz; j++) {
            for (int i = 0; i <= grid.Ny; i++) {
                double y = i * grid.dy - 1.0;
                double z = j * grid.dz - 1.0;
                
                double f1 = exp(-(phys.a_1 * y * y + phys.a_2 * omega1 * omega1))
                          * exp(-(phys.a_1 * z * z + phys.a_2 * omega2 * omega2));
                
                // 初始化能量分布 (高斯)
                for (int ek = 0; ek < grid.Ng; ek++) {
                    double E = (ek + 0.5) * grid.dg + 1.0;
                    double f2 = 1.0 / sqrt(2.0 * Physics::PI) / phys.sig_E 
                               * exp(-pow((E - 230.0) / phys.sig_E / sqrt(2.0), 2));
                    
                    int idx = (ek * nyz + j * (grid.Ny+1) + i) * n_angle + k;
                    h_F[idx] = f1 * f2;
                }
            }
        }
    }
    
    d_F.copyFromHost(h_F.data(), nyz * grid.Ng * n_angle);
    d_f_F.setZero();
    d_F1.setZero();
    d_f_F1.setZero();
}

void BFPnSolver::solve(double tFinal) {
    std::vector<double> dose_history;
    double t = 0.0;
    int step = 0;
    int ping = 0, pong = 1;
    
    std::cout << "Starting BFPn solver..." << std::endl;
    std::cout << "Grid: " << grid.Nx << "x" << grid.Ny << "x" << grid.Nz 
              << ", Energy: " << grid.Ng << ", Angle: " << grid.Nmu << "x" << grid.Nom << std::endl;
    
    while (t < tFinal) {
        // Primary质子: 角度扩散 -> 空间传输 -> 能量沉积
        stepPrimary(ping, t);
        
        // 计算碰撞源项 (基于当前primary)
        computeCollisionSource(ping);
        
        // Secondary质子: 演化 (包含源项)
        if (step > 0) {
            stepSecondary(ping, pong, t);
            std::swap(ping, pong);
        }
        
        // 计算剂量
        double dose = computeDose();
        dose_history.push_back(dose);
        
        t += grid.dt;
        step++;
        
        if (step % 100 == 0) {
            std::cout << "Step " << step << ", t = " << t << "/" << tFinal 
                      << ", dose = " << dose << std::endl;
        }
    }
    
    saveResults(dose_history, "dose_output.txt");
    std::cout << "Solver completed. Results saved to dose_output.txt" << std::endl;
}

void BFPnSolver::stepPrimary(int ping, double t) {
    // 角度扩散
    angle_solver->solve(d_F.data(), d_sig_trg.data(),
                       (grid.Ny+1)*(grid.Nz+1), grid.Ng, grid.dt,
                       stream_primary[0]);
    
    // 空间传输 (可以重叠)
    transport_solver->solve(d_F.data(), d_Omend.data(),
                           grid.Ng, grid.Nmu*grid.Nom, grid.dt,
                           stream_primary[1]);
    
    // 同步后能量沉积
    cudaStreamSynchronize(stream_primary[0]);
    cudaStreamSynchronize(stream_primary[1]);
    
    // 能量沉积 (需要完整数据)
    energy_solver->solve(d_F.data(), d_f_F.data(),
                        d_F.data(), d_f_F.data(),  // 半步值，简化
                        d_S_s.data(), d_sigma_c.data(), d_T_c.data(),
                        grid.dt, grid.dg,
                        (grid.Ny+1)*(grid.Nz+1), grid.Nmu*grid.Nom,
                        false, nullptr, 0);
}

void BFPnSolver::stepSecondary(int ping, int pong, double t) {
    // 类似primary，但包含源项
    angle_solver->solve(d_F1.data(), d_sig_trg.data(),
                       (grid.Ny+1)*(grid.Nz+1), grid.Ng, grid.dt, 0);
    
    transport_solver->solve(d_F1.data(), d_Omend.data(),
                             grid.Ng, grid.Nmu*grid.Nom, grid.dt, 0);
    
    energy_solver->solve(d_F1.data(), d_f_F1.data(),
                        d_F1.data(), d_f_F1.data(),
                        d_S_s.data(), d_sigma_c.data(), d_T_c.data(),
                        grid.dt, grid.dg,
                        (grid.Ny+1)*(grid.Nz+1), grid.Nmu*grid.Nom,
                        true, d_source.data(), 0);
}

void BFPnSolver::computeCollisionSource(int ping) {
    // 使用collision_solver计算源项
    collision_solver->computeSource(d_source.data(),
                                     d_F.data(), d_f_F.data(),
                                     d_F1.data(), d_f_F1.data(),
                                     (grid.Ny+1)*(grid.Nz+1), 0);
}

double BFPnSolver::computeDose() {
    // 简化的剂量计算
    // 实际应该实现完整的积分，这里先基于 f_F 做一个能量沉积分近似
    size_t nyz = (grid.Ny + 1) * (grid.Nz + 1);
    size_t n_angle = grid.Nmu * grid.Nom;
    size_t total = nyz * grid.Ng * n_angle;
    
    // 临时host拷贝计算 (实际应该在GPU上完成)
    std::vector<double> h_F(total), h_f_F(total);
    d_F.copyToHost(h_F.data(), total);
    d_f_F.copyToHost(h_f_F.data(), total);
    
    // 这里使用一个非常简化的剂量近似：把当前时间步在所有网格和角度上损失的“通量”累加，
    // 视为与沉积能量成正比，再乘以能群宽度 dg 作为积分权重。
    double dose_step = 0.0;
    for (size_t i = 0; i < total; ++i) {
        dose_step += h_f_F[i];
    }
    dose_step *= grid.dg;
    
    return dose_step;
}

void BFPnSolver::saveResults(const std::vector<double>& dose, const std::string& filename) {
    std::ofstream out(filename);
    for (size_t i = 0; i < dose.size(); i++) {
        out << i * grid.dt << " " << dose[i] << "\n";
    }
}
