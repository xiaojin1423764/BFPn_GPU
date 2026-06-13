// src/main.cu
#include "bfpm_solver.cuh"
#include "../include/config.hpp"
#include <iostream>
#include <string>
#include <cuda_runtime.h>

void printGPUInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads/Block: " << prop.maxThreadsPerBlock << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // 解析命令行参数
    std::string data_path = "./data";
    double tFinal = 40.0;
    bool test_mode = false;
    int nx = DefaultGrid::NX;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--time" && i + 1 < argc) {
            tFinal = std::stod(argv[++i]);
        } else if (arg == "--test") {
            test_mode = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  <Nx>             Optional x grid points (default: " << DefaultGrid::NX << ")\n"
                      << "  --data <path>    Data file directory (default: ./data)\n"
                      << "  --time <value>   Final simulation time (default: 40.0)\n"
                      << "  --test           Run in test mode\n"
                      << "  --help           Show this help\n";
            return 0;
        } else if (!arg.empty() && arg[0] != '-') {
            nx = std::stoi(arg);
        }
    }
    
    // 打印GPU信息
    printGPUInfo();
    
    if (test_mode) {
        std::cout << "Running in test mode..." << std::endl;
        // 执行简单测试
        return 0;
    }
    
    // 配置网格参数
    GridParams grid;
    grid.Nx = nx;
    grid.Ny = DefaultGrid::NY;
    grid.Nz = DefaultGrid::NZ;
    grid.Ng = DefaultGrid::NG;
    grid.Nmu = DefaultGrid::NMU;
    grid.Nom = DefaultGrid::NOM;
    
    grid.Lx = DefaultGrid::LX;
    grid.Ly = DefaultGrid::LY;
    grid.Lz = DefaultGrid::LZ;
    grid.Lu = DefaultGrid::LU;
    grid.Lv = DefaultGrid::LV;
    grid.Lg = DefaultGrid::LG;
    
    // 配置物理参数
    PhysicsParams phys;
    phys.sig_E = 1.0;
    phys.a_1 = 1.0 / 2.0 / 0.1 / 0.1;
    phys.a_2 = 1.0 / 2.0 / 1e-6 / 1e-6;
    phys.C_c = 1.0 / sqrt(2.0 * Physics::PI * 0.1 * 1e-6);
    
    try {
        // 创建并运行求解器
        BFPnSolver solver(grid, phys);
        solver.initialize(data_path);
        solver.solve(tFinal);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
