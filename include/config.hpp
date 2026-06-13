// include/config.hpp
#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cmath>

// 物理常数
namespace Physics {
    constexpr double PI = 3.14159265358979323846;
    constexpr double ME = 9.10956e-31;      // 电子质量 (kg)
    constexpr double MP = 1.673e-27;        // 质子质量 (kg)
    constexpr double ALPHA = 1.0/137.0;     // 精细结构常数
    constexpr double C_LIGHT = 2.998e8;     // 光速 (m/s)
    constexpr double HBARC = 197.3;         // MeV·fm
    constexpr double E_CHARGE = 1.6e-19;    // 元电荷 (C)
    constexpr double EV_TO_J = 1.6e-19;     // eV转焦耳
}

// 默认网格配置
namespace DefaultGrid {
    constexpr int NX = 4000;
    constexpr int NY = 10;
    constexpr int NZ = 10;
    constexpr int NG = 500;
    constexpr int NMU = 11;
    constexpr int NOM = 11;
    
    constexpr double LX = 40.0;
    constexpr double LY = 1.0;
    constexpr double LZ = 1.0;
    constexpr double LU = 1.0;
    constexpr double LV = 1.0;
    constexpr double LG = 259.0;
}

// 数值参数
namespace Numerics {
    constexpr double EPSILON = 1e-10;
    constexpr double KERNEL_THRESHOLD = 1e-3;
    constexpr double SPARSE_THRESHOLD = 1e-5;
    constexpr int MAX_BATCH_SIZE = 1024;
    constexpr int PCR_BLOCK_SIZE = 512;
}

#endif
