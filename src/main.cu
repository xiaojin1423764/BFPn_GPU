// src/main.cu
#include "bfpm_solver.cuh"
#include "../include/config.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

namespace {
std::vector<double> parseDepthList(const std::string& text) {
    std::vector<double> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::stod(item));
        }
    }
    return values;
}
}

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
    bool data_path_set = false;
    std::string material_name;
    double tFinal = 40.0;
    bool test_mode = false;
    int nx = DefaultGrid::NX;
    int ny = DefaultGrid::NY;
    int nz = DefaultGrid::NZ;
    int ng = DefaultGrid::NG;
    int nmu = DefaultGrid::NMU;
    int nom = DefaultGrid::NOM;
    double ly = DefaultGrid::LY;
    double lz = DefaultGrid::LZ;
    double lu = DefaultGrid::LU;
    double lv = DefaultGrid::LV;
    double beam_energy = 230.0;
    double sigma_e = 1.0;
    double sigma_yz = 0.1;
    double density = 1.0;
    double energy_density = 1.0;
    bool lite_memory = false;
    bool streaming_full = false;
    bool primary_only = false;
    bool legacy_energy = false;
    bool eq15_straggling = false;
    bool save_energy_moments = false;
    bool energy_only = false;
    bool no_transport = false;
    bool no_angle = false;
    bool no_spatial_clipping = false;
    bool no_zero_chunk_skip = false;
    bool no_fused_stream_boundary = false;
    bool profile_steps = false;
    bool discrete_angle_delta = false;
    bool normalize_initial_mass = false;
    bool trapezoidal_yz = false;
    int streaming_lane_chunk = 262144;
    int streaming_energy_chunk = 128;
    int idd_stride = 1;
    std::string streaming_dir = "/tmp";
    std::vector<double> spot_depths;
    std::string spot_prefix = "spot";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_path = argv[++i];
            data_path_set = true;
        } else if (arg == "--time" && i + 1 < argc) {
            tFinal = std::stod(argv[++i]);
        } else if (arg == "--energy" && i + 1 < argc) {
            beam_energy = std::stod(argv[++i]);
        } else if (arg == "--sigma-e" && i + 1 < argc) {
            sigma_e = std::stod(argv[++i]);
        } else if (arg == "--rho" && i + 1 < argc) {
            density = std::stod(argv[++i]);
            energy_density = density;
        } else if (arg == "--ny" && i + 1 < argc) {
            ny = std::stoi(argv[++i]);
        } else if (arg == "--nz" && i + 1 < argc) {
            nz = std::stoi(argv[++i]);
        } else if (arg == "--ly" && i + 1 < argc) {
            ly = std::stod(argv[++i]);
        } else if (arg == "--lz" && i + 1 < argc) {
            lz = std::stod(argv[++i]);
        } else if (arg == "--lu" && i + 1 < argc) {
            lu = std::stod(argv[++i]);
        } else if (arg == "--lv" && i + 1 < argc) {
            lv = std::stod(argv[++i]);
        } else if (arg == "--ng" && i + 1 < argc) {
            ng = std::stoi(argv[++i]);
        } else if (arg == "--nmu" && i + 1 < argc) {
            nmu = std::stoi(argv[++i]);
        } else if (arg == "--nom" && i + 1 < argc) {
            nom = std::stoi(argv[++i]);
        } else if (arg == "--material" && i + 1 < argc) {
            std::string material = argv[++i];
            material_name = material;
            if (material == "water") {
                density = 1.0;
                energy_density = 1.0;
            } else if (material == "bone") {
                density = 1.757;
                energy_density = density;
            } else if (material == "air") {
                density = 0.001205;
                energy_density = 0.001205;
            } else {
                std::cerr << "Unknown material: " << material << std::endl;
                return 1;
            }
        } else if (arg == "--test") {
            test_mode = true;
        } else if (arg == "--lite-memory") {
            lite_memory = true;
        } else if (arg == "--streaming-full") {
            streaming_full = true;
        } else if (arg == "--primary-only") {
            primary_only = true;
        } else if (arg == "--energy-model" && i + 1 < argc) {
            std::string model = argv[++i];
            if (model == "legacy") {
                legacy_energy = true;
            } else if (model == "eq15") {
                legacy_energy = false;
            } else {
                std::cerr << "Unknown energy model: " << model << std::endl;
                return 1;
            }
        } else if (arg == "--eq15-straggling") {
            eq15_straggling = true;
        } else if (arg == "--save-energy-moments") {
            save_energy_moments = true;
        } else if (arg == "--energy-only") {
            energy_only = true;
        } else if (arg == "--no-transport") {
            no_transport = true;
        } else if (arg == "--no-angle") {
            no_angle = true;
        } else if (arg == "--no-spatial-clipping") {
            no_spatial_clipping = true;
        } else if (arg == "--no-zero-chunk-skip") {
            no_zero_chunk_skip = true;
        } else if (arg == "--no-fused-stream-boundary") {
            no_fused_stream_boundary = true;
        } else if (arg == "--profile-steps") {
            profile_steps = true;
        } else if (arg == "--discrete-angle-delta") {
            discrete_angle_delta = true;
        } else if (arg == "--normalize-initial-mass") {
            normalize_initial_mass = true;
        } else if (arg == "--trapezoidal-yz") {
            trapezoidal_yz = true;
        } else if (arg == "--sigma-yz" && i + 1 < argc) {
            sigma_yz = std::stod(argv[++i]);
        } else if (arg == "--lane-chunk" && i + 1 < argc) {
            streaming_lane_chunk = std::stoi(argv[++i]);
        } else if (arg == "--energy-chunk" && i + 1 < argc) {
            streaming_energy_chunk = std::stoi(argv[++i]);
        } else if (arg == "--stream-dir" && i + 1 < argc) {
            streaming_dir = argv[++i];
        } else if (arg == "--idd-stride" && i + 1 < argc) {
            idd_stride = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--spot-depths" && i + 1 < argc) {
            spot_depths = parseDepthList(argv[++i]);
        } else if (arg == "--spot-prefix" && i + 1 < argc) {
            spot_prefix = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  <Nx>             Optional x grid points (default: " << DefaultGrid::NX << ")\n"
                      << "  --data <path>    Data file directory (default: ./data)\n"
                      << "  --time <value>   Final simulation time (default: 40.0)\n"
                      << "  --energy <MeV>   Incoming beam energy E0 (default: 230.0)\n"
                      << "  --sigma-e <MeV>  Initial energy Gaussian sigma (default: 1.0)\n"
                      << "  --material <name> Material density preset: water, bone, air\n"
                      << "  --rho <value>    Override material density in g/cm^3\n"
                      << "  --ny <value>     Y grid cells (paper Figure 3 uses 80)\n"
                      << "  --nz <value>     Z grid cells (paper Figure 3 uses 80)\n"
                      << "  --ly <cm>        Y domain width (default: " << DefaultGrid::LY << ")\n"
                      << "  --lz <cm>        Z domain width (default: " << DefaultGrid::LZ << ")\n"
                      << "  --lu <value>     U angular domain width (default: " << DefaultGrid::LU << ")\n"
                      << "  --lv <value>     V angular domain width (default: " << DefaultGrid::LV << ")\n"
                      << "  --ng <value>     Energy groups (paper Figure 3 uses 500)\n"
                      << "  --nmu <value>    Angular u grid points (paper Figure 3 uses 20)\n"
                      << "  --nom <value>    Angular v grid points (paper Figure 3 uses 20)\n"
                      << "  --lite-memory    Primary-only low-memory mode for large grid development\n"
                      << "  --streaming-full Out-of-core full mode using host-backed chunks\n"
                      << "  --primary-only   Skip secondary proton evolution/source\n"
                      << "  --energy-model <eq15|legacy> Energy update model (default: eq15)\n"
                      << "  --eq15-straggling Enable Eq.15 straggling/SIPG terms during calibration\n"
                      << "  --save-energy-moments Save paper Eq.25 energy-moment diagnostics\n"
                      << "  --energy-only    Only advance the primary energy subsystem for convergence isolation\n"
                      << "  --no-transport   Skip the spatial transport subsystem\n"
                      << "  --no-angle       Skip the angular diffusion subsystem\n"
                      << "  --no-spatial-clipping Disable positivity clipping in spatial transport\n"
                      << "  --no-zero-chunk-skip Disable inactive high-energy chunk skipping\n"
                      << "  --no-fused-stream-boundary Disable fused transport/second-energy pass\n"
                      << "  --profile-steps  Print energy/transport/angle step timing summary\n"
                      << "  --discrete-angle-delta Use a quadrature-normalized mono-directional beam\n"
                      << "  --normalize-initial-mass Normalize the discrete y/z/u/v/E beam mass\n"
                      << "  --trapezoidal-yz Use trapezoidal y/z weights in normalization and IDD\n"
                      << "  --sigma-yz <cm>  Initial transverse Gaussian sigma (default: 0.1; paper spot figures use 0.3)\n"
                      << "  --lane-chunk <n> Streaming energy lane chunk (default: 262144)\n"
                      << "  --energy-chunk <n> Streaming transport/angle energy chunk (default: 128)\n"
                      << "  --stream-dir <path> Streaming backing file directory (default: /tmp)\n"
                      << "  --idd-stride <n> Compute/save IDD every n steps in streaming mode (default: 1)\n"
                      << "  --spot-depths <x,...> Save YZ dose planes near requested depths\n"
                      << "  --spot-prefix <path> Prefix for spot plane files (default: spot)\n"
                      << "  --test           Run in test mode\n"
                      << "  --help           Show this help\n";
            return 0;
        } else if (!arg.empty() && arg[0] != '-') {
            nx = std::stoi(arg);
        }
    }

    if (!data_path_set && !material_name.empty()) {
        data_path = "BFPn_CPU_Solver/" + material_name;
    }

    if (lite_memory && streaming_full) {
        std::cerr << "--lite-memory and --streaming-full are mutually exclusive" << std::endl;
        return 1;
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
    grid.Ny = ny;
    grid.Nz = nz;
    grid.Ng = ng;
    grid.Nmu = nmu;
    grid.Nom = nom;
    
    grid.Lx = DefaultGrid::LX;
    grid.Ly = ly;
    grid.Lz = lz;
    grid.Lu = lu;
    grid.Lv = lv;
    grid.Lg = DefaultGrid::LG;
    
    // 配置物理参数
    PhysicsParams phys;
    phys.sig_E = sigma_e;
    phys.a_1 = 1.0 / 2.0 / sigma_yz / sigma_yz;
    phys.a_2 = 1.0 / 2.0 / 1e-6 / 1e-6;
    phys.C_c = 1.0 / sqrt(2.0 * Physics::PI * 0.1 * 1e-6);
    phys.beam_energy = beam_energy;
    phys.density = density;
    phys.energy_density = energy_density;
    phys.material_name = material_name;
    phys.lite_memory = lite_memory;
    phys.streaming_full = streaming_full;
    phys.primary_only = primary_only;
    phys.legacy_energy = legacy_energy;
    phys.eq15_straggling = eq15_straggling;
    phys.save_energy_moments = save_energy_moments;
    phys.energy_only = energy_only;
    phys.no_transport = no_transport;
    phys.no_angle = no_angle;
    phys.no_spatial_clipping = no_spatial_clipping;
    phys.no_zero_chunk_skip = no_zero_chunk_skip;
    phys.no_fused_stream_boundary = no_fused_stream_boundary;
    phys.profile_steps = profile_steps;
    phys.discrete_angle_delta = discrete_angle_delta;
    phys.normalize_initial_mass = normalize_initial_mass;
    phys.trapezoidal_yz = trapezoidal_yz;
    phys.streaming_lane_chunk = streaming_lane_chunk;
    phys.streaming_energy_chunk = streaming_energy_chunk;
    phys.streaming_dir = streaming_dir;
    phys.spot_depths = spot_depths;
    phys.spot_prefix = spot_prefix;
    phys.idd_stride = idd_stride;

    std::cout << "Domain widths: Lx=" << grid.Lx
              << ", Ly=" << grid.Ly << ", Lz=" << grid.Lz
              << ", Lu=" << grid.Lu << ", Lv=" << grid.Lv << std::endl;
    std::cout << "Initial beam sigmas: yz=" << sigma_yz
              << " cm, E=" << sigma_e << " MeV, angular=1e-6" << std::endl;
    std::cout << "Initial quadrature: angle="
              << (discrete_angle_delta ? "discrete-delta" : "sampled-gaussian")
              << ", mass normalization=" << (normalize_initial_mass ? "on" : "off")
              << ", y/z trapezoidal weights=" << (trapezoidal_yz ? "on" : "off")
              << std::endl;
    std::cout << "Spatial positivity clipping: "
              << (no_spatial_clipping ? "disabled" : "enabled") << std::endl;
    
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
