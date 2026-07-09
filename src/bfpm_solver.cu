// src/bfpm_solver.cu
#include "bfpm_solver.cuh"
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <cstdio>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cerrno>

namespace {
double bytesToGiB(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

std::string formatGiB(size_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << bytesToGiB(bytes) << " GiB";
    return out.str();
}

size_t checkedMul(size_t a, size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(std::string("Grid is too large while estimating ") + label);
    }
    return a * b;
}

size_t alignedLaneChunk(size_t requested, size_t n_angle) {
    if (n_angle == 0) {
        return std::max<size_t>(1, requested);
    }
    return std::max(n_angle, (std::max<size_t>(1, requested) / n_angle) * n_angle);
}

struct ElementFraction {
    int Z;
    double weight;
};

double meanExcitationEV(int Z) {
    if (Z == 1) return 19.0;
    if (Z <= 13) return 11.2 + 11.7 * Z;
    return 52.8 + 8.71 * Z;
}

double atomicMassNumber(int Z) {
    switch (Z) {
        case 1: return 1.0;
        case 6: return 12.0;
        case 7: return 14.0;
        case 8: return 16.0;
        case 11: return 23.0;
        case 12: return 24.0;
        case 15: return 31.0;
        case 16: return 32.0;
        case 18: return 40.0;
        case 20: return 40.0;
        default: return 2.0 * Z;
    }
}

double betheStoppingElement(double E, int Z) {
    constexpr double kappa = 0.307;
    constexpr double me_c2 = 0.51099895e6;
    double beta2 = E * (E + 2.0 * 938.3) / ((E + 938.3) * (E + 938.3));
    beta2 = std::max(beta2, Numerics::EPSILON);
    double gamma2 = 1.0 / std::max(1.0 - beta2, Numerics::EPSILON);
    double I = meanExcitationEV(Z);
    double log_arg = std::max(2.0 * me_c2 * beta2 * gamma2 / I, 1.0 + Numerics::EPSILON);
    double bracket = std::log(log_arg) - beta2;
    return kappa * (static_cast<double>(Z) / atomicMassNumber(Z)) / beta2 * bracket;
}

std::vector<ElementFraction> materialComposition(const std::string& material) {
    if (material == "bone") {
        return {
            {1, 0.04200}, {6, 0.1940}, {7, 0.04000}, {8, 0.4250},
            {11, 0.001000}, {12, 0.002000}, {15, 0.09200}, {16, 0.003000},
            {20, 0.2010}
        };
    }
    if (material == "air") {
        return {{6, 0.0001248}, {7, 0.7553}, {8, 0.2318}, {18, 0.01283}};
    }
    return {{1, 0.1111}, {8, 0.8889}};
}
}

__global__ void reduceSumKernel(const double* in, double* block_sums, size_t n) {
    extern __shared__ double shared[];
    unsigned int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x * 2 + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x * 2;

    double sum = 0.0;
    while (idx < n) {
        sum += in[idx];
        if (idx + blockDim.x < n) {
            sum += in[idx + blockDim.x];
        }
        idx += stride;
    }

    shared[tid] = sum;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = shared[0];
    }
}

__global__ void integratedDepthDoseKernel(
    const double* F,
    const double* f_F,
    const double* F1,
    const double* f_F1,
    const double* S_s,
    const double* sigma_c,
    const double* en,
    double* block_sums,
    int Ng,
    int nyz,
    int NmuNom,
    double dg,
    double du,
    double dv,
    double dy,
    double dz,
    double density,
    size_t total
) {
    extern __shared__ double shared[];
    unsigned int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x * 2 + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x * 2;

    double sum = 0.0;
    const long long n_per_E = static_cast<long long>(nyz) * NmuNom;

    while (idx < total) {
        for (int pass = 0; pass < 2; ++pass) {
            size_t cur = idx + static_cast<size_t>(pass) * blockDim.x;
            if (cur >= total) continue;

            int ek = static_cast<int>(cur / n_per_E);
            double psi1 = F[cur] + F1[cur];
            double psi2 = f_F[cur] + f_F1[cur];

            double S_mid = 0.5 * (S_s[ek] + S_s[ek + 1]);
            double slope_weight = (S_s[ek + 1] - S_s[ek]) / 6.0;

            // Leading dose term: local fluence times stopping power,
            // integrated over energy and angle, then divided by density.
            sum += (psi1 * S_mid * dg + psi2 * slope_weight * dg) / density;
        }
        idx += stride;
    }

    shared[tid] = sum;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = shared[0] * du * dv * dy * dz;
    }
}

__global__ void energyMomentKernel(
    const double* F,
    const double* f_F,
    const double* F1,
    const double* f_F1,
    double* moments,
    int Ng,
    int nyz,
    int NmuNom,
    double du,
    double dv,
    double dy,
    double dz
) {
    int ek = blockIdx.x;
    int tid = threadIdx.x;
    if (ek >= Ng) return;

    extern __shared__ double shared[];
    double* shared_psi1 = shared;
    double* shared_psi2 = shared + blockDim.x;

    const long long n_per_E = static_cast<long long>(nyz) * NmuNom;
    const long long base = static_cast<long long>(ek) * n_per_E;
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (long long lane = tid; lane < n_per_E; lane += blockDim.x) {
        long long idx = base + lane;
        sum1 += F[idx] + F1[idx];
        sum2 += f_F[idx] + f_F1[idx];
    }

    shared_psi1[tid] = sum1;
    shared_psi2[tid] = sum2;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_psi1[tid] += shared_psi1[tid + offset];
            shared_psi2[tid] += shared_psi2[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double weight = du * dv * dy * dz;
        moments[2 * ek + 0] = shared_psi1[0] * weight;
        moments[2 * ek + 1] = shared_psi2[0] * weight;
    }
}

__global__ void pairDepthDoseLaneKernel(
    const double* F,
    const double* f_F,
    const double* S_s,
    double* block_sums,
    int Ng,
    int lanes,
    double dg,
    double density,
    double quadrature_weight,
    size_t total
) {
    extern __shared__ double shared[];
    unsigned int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x * 2 + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x * 2;

    double sum = 0.0;
    while (idx < total) {
        for (int pass = 0; pass < 2; ++pass) {
            size_t cur = idx + static_cast<size_t>(pass) * blockDim.x;
            if (cur >= total) continue;

            int ek = static_cast<int>(cur / lanes);
            double S_mid = 0.5 * (S_s[ek] + S_s[ek + 1]);
            double slope_weight = (S_s[ek + 1] - S_s[ek]) / 6.0;
            sum += (F[cur] * S_mid * dg + f_F[cur] * slope_weight * dg) / density;
        }
        idx += stride;
    }

    shared[tid] = sum;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = shared[0] * quadrature_weight;
    }
}

__global__ void spotDosePlaneKernel(
    const double* F,
    const double* f_F,
    const double* F1,
    const double* f_F1,
    const double* S_s,
    double* plane,
    int Ng,
    int nyz,
    int NmuNom,
    double dg,
    double du,
    double dv,
    double density
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nyz) return;

    double sum = 0.0;
    const long long n_per_E = static_cast<long long>(nyz) * NmuNom;
    for (int ek = 0; ek < Ng; ++ek) {
        double S_mid = 0.5 * (S_s[ek] + S_s[ek + 1]);
        double slope_weight = (S_s[ek + 1] - S_s[ek]) / 6.0;
        long long base = static_cast<long long>(ek) * n_per_E + static_cast<long long>(s) * NmuNom;
        for (int a = 0; a < NmuNom; ++a) {
            long long idx = base + a;
            double psi1 = F[idx] + F1[idx];
            double psi2 = f_F[idx] + f_F1[idx];
            sum += (psi1 * S_mid * dg + psi2 * slope_weight * dg) / density;
        }
    }
    plane[s] = sum * du * dv;
}

__global__ void energyFluxKernel(
    const double* F,
    const double* f_F,
    const double* en,
    double* block_sums,
    int Ng,
    int nyz,
    int NmuNom,
    double dg,
    double du,
    double dv,
    double dy,
    double dz,
    size_t total
) {
    extern __shared__ double shared[];
    unsigned int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x * 2 + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x * 2;

    double sum = 0.0;
    const long long n_per_E = static_cast<long long>(nyz) * NmuNom;
    while (idx < total) {
        for (int pass = 0; pass < 2; ++pass) {
            size_t cur = idx + static_cast<size_t>(pass) * blockDim.x;
            if (cur >= total) continue;
            int ek = static_cast<int>(cur / n_per_E);
            double E_mid = 0.5 * (en[ek] + en[ek + 1]);
            double psi1 = F[cur];
            double psi2 = f_F ? f_F[cur] : 0.0;
            sum += (E_mid * psi1 * dg + psi2 * dg * dg / 6.0);
        }
        idx += stride;
    }

    shared[tid] = sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) shared[tid] += shared[tid + offset];
        __syncthreads();
    }
    if (tid == 0) {
        block_sums[blockIdx.x] = shared[0] * du * dv * dy * dz;
    }
}

__global__ void integratedDepthDoseLiteKernel(
    const double* F,
    const double* S_s,
    double* block_sums,
    int Ng,
    int nyz,
    int NmuNom,
    double dg,
    double du,
    double dv,
    double dy,
    double dz,
    double density,
    size_t total
) {
    extern __shared__ double shared[];
    unsigned int tid = threadIdx.x;
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x * 2 + threadIdx.x;
    size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x * 2;

    double sum = 0.0;
    const long long n_per_E = static_cast<long long>(nyz) * NmuNom;

    while (idx < total) {
        for (int pass = 0; pass < 2; ++pass) {
            size_t cur = idx + static_cast<size_t>(pass) * blockDim.x;
            if (cur >= total) continue;

            int ek = static_cast<int>(cur / n_per_E);
            double S_mid = 0.5 * (S_s[ek] + S_s[ek + 1]);
            sum += F[cur] * S_mid * dg / density;
        }
        idx += stride;
    }

    shared[tid] = sum;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = shared[0] * du * dv * dy * dz;
    }
}

__global__ void litePrimaryEnergyStepKernel(
    double* F,
    const double* S_s,
    const double* sigma_c,
    int Ng,
    int nyz,
    int NmuNom,
    double dt,
    double dg
) {
    const long long n_per_E = static_cast<long long>(nyz) * NmuNom;
    long long lane = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lane >= n_per_E) return;

    double next_F = 0.0;
    for (int ek = Ng - 1; ek >= 0; --ek) {
        long long idx = static_cast<long long>(ek) * n_per_E + lane;
        double Fg = max(F[idx], 0.0);
        double sig = max(sigma_c[ek], 0.0);
        double S_lo = max(S_s[ek], 0.0);
        double S_hi = max(S_s[ek + 1], 0.0);
        double denom = 1.0 / dt + S_lo / dg + sig;
        double F_val = (Fg / dt + S_hi / dg * next_F)
                     / max(denom, Numerics::EPSILON);
        F[idx] = max(F_val, 0.0);
        next_F = F[idx];
    }
}

__global__ void streamingEnergyDgCnKernel(
    const double* F_old,
    const double* f_old,
    double* F_new,
    double* f_new,
    const double* source_term,
    const double* S_s,
    const double* sigma_c,
    const double* T_c,
    int Ng,
    int lanes,
    double dt,
    double dg,
    double density,
    bool use_legacy,
    bool enable_straggling,
    bool is_secondary
) {
    int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;

    const double dg2 = max(dg * dg, Numerics::EPSILON);
    auto load = [&](const double* arr, int ek) -> double {
        if (!arr || ek < 0 || ek >= Ng) return 0.0;
        return arr[static_cast<long long>(ek) * lanes + lane];
    };
    double next_F = 0.0;
    double next_f = 0.0;

    for (int ek = Ng - 1; ek >= 0; --ek) {
        long long idx = static_cast<long long>(ek) * lanes + lane;
        double Fg = F_old[idx];
        double fg = f_old ? f_old[idx] : 0.0;
        double sig = max(sigma_c[ek], 0.0);
        double S_lo = max(S_s[ek], 0.0);
        double S_hi = max(S_s[ek + 1], 0.0);
        double T_g = (enable_straggling && T_c) ? max(T_c[ek], 0.0) : 0.0;
        double T_hi = (enable_straggling && T_c && ek + 1 < Ng) ? max(T_c[ek + 1], 0.0) : 0.0;
        double T_lo = (enable_straggling && T_c && ek > 0) ? max(T_c[ek - 1], 0.0) : 0.0;
        double source = (is_secondary && source_term) ? source_term[idx] : 0.0;

        if (use_legacy) {
            double denom = 1.0 / dt + density * S_lo / dg + sig;
            double F_val = (max(Fg, 0.0) / dt + density * S_hi / dg * next_F + source)
                         / max(denom, Numerics::EPSILON);
            double denom_f = 1.0 / dt + density * S_lo / dg + sig;
            double f_val = (fg / dt + density * S_hi / dg * next_f)
                         / max(denom_f, Numerics::EPSILON);
            if (T_g > 0.0) {
                double Fhi = load(F_old, ek + 1);
                double Flo = load(F_old, ek - 1);
                double lap = max(Fhi, 0.0) - 2.0 * max(Fg, 0.0) + max(Flo, 0.0);
                F_val += 0.5 * dt * T_g / dg2 * lap;
            }
            F_new[idx] = max(F_val, 0.0);
            f_new[idx] = max(f_val, 0.0);
            next_F = F_new[idx];
            next_f = f_new[idx];
            continue;
        }

        const double p1_hi = load(F_old, ek + 1);
        const double p2_hi = load(f_old, ek + 1);
        const double p1_lo = load(F_old, ek - 1);
        const double up = p1_hi - p2_hi;
        const double cur = Fg - fg;

        const double den_F = max(1.0 / dt + 0.5 * density * S_lo / dg + 0.5 * sig,
                                 Numerics::EPSILON);
        const double inv_F = 1.0 / den_F;
        const double A = (1.5 * density * S_hi / dg) * inv_F;
        const double F_from_f = (0.5 * density * S_lo / dg) * inv_F;
        const double F_from_hi = (0.5 * density * S_hi / dg) * inv_F;
        const double strag_rhs = 0.5 * density * T_hi / dg2 * p1_hi
                               + 0.5 * density * T_lo / dg2 * p1_lo
                               - density * T_g / dg2 * Fg;
        const double common_rhs = 0.5 * density * S_hi / dg * up
                                - 0.5 * density * S_lo / dg * cur
                                - 0.5 * sig * Fg
                                + Fg / dt
                                + strag_rhs
                                + source;
        const double rhs4 = inv_F * common_rhs;

        const double rhs1e = fg / dt - 0.5 * sig * fg;
        const double rhs2e = 1.5 * density * S_hi / dg * up
                           + 1.5 * density * S_lo / dg * cur
                           - 1.5 * density * (S_lo + S_hi) / dg * Fg
                           - 0.5 * density * (S_hi - S_lo) / dg * fg;
        const double rhs3e = A * common_rhs;

        const double high2 = 1.5 * density * S_hi / dg;
        const double hi_delta = next_F - next_f;
        const double rhs = rhs1e + rhs2e - rhs3e
                         + (high2 - A * 0.5 * density * S_hi / dg) * hi_delta;
        const double coe = 1.0 / dt + 0.5 * sig
                         + 1.5 * density * S_lo / dg
                         + 0.5 * density * (S_hi - S_lo) / dg
                         + A * 0.5 * density * S_lo / dg;

        double f_val = rhs / max(coe, Numerics::EPSILON);
        double F_val = F_from_f * f_val + rhs4 + F_from_hi * hi_delta;

        F_new[idx] = F_val;
        f_new[idx] = f_val;
        next_F = F_val;
        next_f = f_val;
    }
}

__global__ void streamingSecondarySourceKernel(
    const double* primary_F,
    const double* primary_f,
    double* source_F,
    const double* ker_e1,
    const double* ker_e2,
    const double* ker_v,
    int Ng,
    int lanes,
    size_t lane0,
    int NmuNom,
    double dg
) {
    long long total = static_cast<long long>(Ng) * lanes;
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    int out_e = static_cast<int>(gid / lanes);
    int lane = static_cast<int>(gid - static_cast<long long>(out_e) * lanes);
    int out_ang = static_cast<int>((lane0 + static_cast<size_t>(lane)) % NmuNom);
    int lane_base = lane - out_ang;

    double q1 = 0.0;
    double q2 = 0.0;
    for (int src_e = 0; src_e < Ng; ++src_e) {
        double e1 = ker_e1[static_cast<long long>(out_e) * Ng + src_e];
        double e2 = ker_e2[static_cast<long long>(out_e) * Ng + src_e];
        if (e1 == 0.0 && e2 == 0.0) continue;

        const double* kv = ker_v + static_cast<long long>(src_e) * NmuNom * NmuNom;
        for (int in_ang = 0; in_ang < NmuNom; ++in_ang) {
            int src_lane = lane_base + in_ang;
            if (src_lane < 0 || src_lane >= lanes) continue;
            double w = kv[static_cast<long long>(in_ang) * NmuNom + out_ang];
            if (w == 0.0) continue;
            long long src = static_cast<long long>(src_e) * lanes + src_lane;
            q1 += max(primary_F[src], 0.0) * e1 * w;
            q2 += primary_f[src] * e2 * w;
        }
    }
    source_F[gid] = dg * (q1 + q2 / 3.0);
}

__global__ void initializeLiteDistributionKernel(
    double* F,
    const double* Omend,
    int Ny,
    int Nz,
    int Ng,
    int NmuNom,
    double Ly,
    double Lz,
    double dy,
    double dz,
    double dg,
    double a_1,
    double a_2,
    double sig_E,
    double beam_energy,
    double omega1_abs_center,
    double omega2_abs_center
) {
    long long nyz = static_cast<long long>(Ny + 1) * (Nz + 1);
    long long total = static_cast<long long>(Ng) * nyz * NmuNom;
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    int ang = static_cast<int>(gid % NmuNom);
    long long s_total = gid / NmuNom;
    int ek = static_cast<int>(s_total / nyz);
    long long s = s_total % nyz;
    int i = static_cast<int>(s % (Ny + 1));
    int j = static_cast<int>(s / (Ny + 1));

    double y = -0.5 * Ly + i * dy;
    double z = -0.5 * Lz + j * dz;
    double omega1 = Omend[2 * ang + 0];
    double omega2 = Omend[2 * ang + 1];
    double E = (ek + 0.5) * dg + 1.0;

    double domega1 = fabs(omega1) - omega1_abs_center;
    double domega2 = fabs(omega2) - omega2_abs_center;
    double f1 = exp(-(a_1 * y * y + a_2 * domega1 * domega1))
              * exp(-(a_1 * z * z + a_2 * domega2 * domega2));
    double f2 = 1.0 / sqrt(2.0 * Physics::PI) / sig_E
              * exp(-pow((E - beam_energy) / sig_E / sqrt(2.0), 2));

    F[gid] = f1 * f2;
}

__global__ void initializeStreamingDistributionKernel(
    double* F,
    const double* Omend,
    int Ny,
    int Nz,
    int NgTotal,
    int energy_start,
    int energy_count,
    int NmuNom,
    double Ly,
    double Lz,
    double dy,
    double dz,
    double dg,
    double a_1,
    double a_2,
    double sig_E,
    double beam_energy,
    double omega1_abs_center,
    double omega2_abs_center
) {
    long long nyz = static_cast<long long>(Ny + 1) * (Nz + 1);
    long long chunk_total = static_cast<long long>(energy_count) * nyz * NmuNom;
    long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= chunk_total) return;

    int ang = static_cast<int>(gid % NmuNom);
    long long s_total = gid / NmuNom;
    int local_ek = static_cast<int>(s_total / nyz);
    int ek = energy_start + local_ek;
    if (ek >= NgTotal) return;

    long long s = s_total % nyz;
    int i = static_cast<int>(s % (Ny + 1));
    int j = static_cast<int>(s / (Ny + 1));

    double y = -0.5 * Ly + i * dy;
    double z = -0.5 * Lz + j * dz;
    double omega1 = Omend[2 * ang + 0];
    double omega2 = Omend[2 * ang + 1];
    double E = (ek + 0.5) * dg + 1.0;

    double domega1 = fabs(omega1) - omega1_abs_center;
    double domega2 = fabs(omega2) - omega2_abs_center;
    double f1 = exp(-(a_1 * y * y + a_2 * domega1 * domega1))
              * exp(-(a_1 * z * z + a_2 * domega2 * domega2));
    double f2 = 1.0 / sqrt(2.0 * Physics::PI) / sig_E
              * exp(-pow((E - beam_energy) / sig_E / sqrt(2.0), 2));

    F[gid] = f1 * f2;
}

void GridParams::computeDeltas() {
    dt = Lx / Nx;
    dy = Ly / Ny;
    dz = Lz / Nz;
    dg = Lg / Ng;
    // Eq. (20)-(21) use zero angular boundary values and sine-series
    // interior unknowns.  Nom indexes the u/omega_z columns, Nmu indexes
    // the v/omega_y rows, so the physical boundary nodes are not stored.
    du = Lu / (Nom + 1);
    dv = Lv / (Nmu + 1);
}

HostDoubleBuffer::HostDoubleBuffer()
    : ptr(nullptr), capacity(0), pinned(false) {}

HostDoubleBuffer::~HostDoubleBuffer() {
    releasePinned();
}

void HostDoubleBuffer::releasePinned() {
    if (pinned && ptr) {
        cudaFreeHost(ptr);
    }
    ptr = nullptr;
    capacity = 0;
    pinned = false;
}

void HostDoubleBuffer::resize(size_t count) {
    if (count <= capacity) {
        return;
    }
    releasePinned();
    pageable.clear();

    void* raw = nullptr;
    cudaError_t err = cudaMallocHost(&raw, count * sizeof(double));
    if (err == cudaSuccess) {
        ptr = static_cast<double*>(raw);
        capacity = count;
        pinned = true;
        return;
    }

    cudaGetLastError();
    pageable.resize(count);
    ptr = pageable.data();
    capacity = count;
    pinned = false;
}

double* HostDoubleBuffer::data() {
    return ptr;
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
    cleanupStreamingStore();

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
    preflightMemory();

    size_t nyz = (grid.Ny + 1) * (grid.Nz + 1);
    size_t n_angle = grid.Nmu * grid.Nom;
    size_t total_size = nyz * grid.Ng * n_angle;

    if (phys.streaming_full) {
        size_t lane_chunk = alignedLaneChunk(
            static_cast<size_t>(std::max(1, phys.streaming_lane_chunk)),
            n_angle);
        size_t energy_chunk = static_cast<size_t>(std::max(1, phys.streaming_energy_chunk));
        size_t energy_chunk_size = energy_chunk * nyz * n_angle;
        size_t lane_chunk_size = static_cast<size_t>(grid.Ng) * lane_chunk;
        size_t stream_size = std::max(energy_chunk_size, lane_chunk_size);

        d_stream_F.allocate(stream_size);
        d_stream_f_F.allocate(stream_size);
        d_stream_F1.allocate(stream_size);
        d_stream_f_F1.allocate(stream_size);
        d_stream_primary_F.allocate(stream_size);
        d_stream_primary_f_F.allocate(stream_size);
        d_stream_old_F.allocate(stream_size);
        d_stream_old_f_F.allocate(stream_size);
        d_energy_moments.allocate(static_cast<size_t>(grid.Ng) * 2);

        constexpr int reduction_block_size = 256;
        size_t reduction_blocks = (stream_size + reduction_block_size * 2 - 1) /
                                  (reduction_block_size * 2);
        d_reduction_sums.allocate(reduction_blocks);

        d_y.allocate(grid.Ny + 1);
        d_z.allocate(grid.Nz + 1);
        d_en.allocate(grid.Ng + 1);
        d_Omend.allocate(n_angle * 2);
        d_utemp.allocate(n_angle);
        d_vtemp.allocate(n_angle);

        d_mu.allocate(grid.Ng * 3);
        d_sigma_c.allocate(grid.Ng);
        d_S_s.allocate(grid.Ng + 1);
        d_T_c.allocate(grid.Ng);
        d_sig_trg.allocate(grid.Ng);
        d_ker_e.allocate(grid.Ng * grid.Ng);
        d_ker_e1.allocate(grid.Ng * grid.Ng);
        d_ker_e2.allocate(grid.Ng * grid.Ng);
        d_ker_v.allocate(grid.Ng * n_angle * n_angle);
        return;
    }

    // 分布函数
    d_F.allocate(total_size);
    if (!phys.lite_memory) {
        d_f_F.allocate(total_size);
        d_F1.allocate(total_size);
        d_f_F1.allocate(total_size);
    }

    constexpr int reduction_block_size = 256;
    size_t reduction_blocks = (total_size + reduction_block_size * 2 - 1) /
                              (reduction_block_size * 2);
    d_reduction_sums.allocate(reduction_blocks);
    d_energy_moments.allocate(static_cast<size_t>(grid.Ng) * 2);

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

    if (!phys.lite_memory) {
        // 内核
        d_ker_e.allocate(grid.Ng * grid.Ng);
        d_ker_e1.allocate(grid.Ng * grid.Ng);
        d_ker_e2.allocate(grid.Ng * grid.Ng);
        d_ker_v.allocate(grid.Ng * n_angle * n_angle);
    }
}

void BFPnSolver::preflightMemory() const {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    const size_t nyz = checkedMul(static_cast<size_t>(grid.Ny + 1),
                                  static_cast<size_t>(grid.Nz + 1),
                                  "Ny*Nz");
    const size_t n_angle = checkedMul(static_cast<size_t>(grid.Nmu),
                                      static_cast<size_t>(grid.Nom),
                                      "Nmu*Nom");
    const size_t total_cells = checkedMul(checkedMul(nyz, static_cast<size_t>(grid.Ng), "phase space"),
                                          n_angle,
                                          "phase space");
    const size_t one_phase_array = checkedMul(total_cells, sizeof(double), "phase-space bytes");

    size_t estimated = 0;
    if (phys.lite_memory) {
        estimated += one_phase_array;  // d_F only
        estimated += checkedMul((total_cells + 511) / 512, sizeof(double), "reduction buffer");
        estimated += checkedMul(static_cast<size_t>(grid.Ng), sizeof(double), "physics arrays") * 5;
        estimated += checkedMul(checkedMul(static_cast<size_t>(grid.Ng), static_cast<size_t>(grid.Ng), "streaming energy kernels"),
                                3 * sizeof(double),
                                "streaming energy kernel bytes");
        estimated += checkedMul(checkedMul(static_cast<size_t>(grid.Ng), n_angle, "streaming angle kernel"),
                                checkedMul(n_angle, sizeof(double), "streaming angle kernel bytes"),
                                "streaming angle kernel bytes");
        estimated += checkedMul(n_angle, sizeof(double), "angle helper arrays") * 4;
    } else if (phys.streaming_full) {
        const size_t lane_chunk = alignedLaneChunk(
            static_cast<size_t>(std::max(1, phys.streaming_lane_chunk)),
            n_angle);
        const size_t energy_chunk = static_cast<size_t>(std::max(1, phys.streaming_energy_chunk));
        const size_t energy_chunk_cells = checkedMul(checkedMul(energy_chunk, nyz, "stream energy chunk"),
                                                     n_angle,
                                                     "stream energy chunk");
        const size_t lane_chunk_cells = checkedMul(checkedMul(static_cast<size_t>(grid.Ng), lane_chunk, "stream lane chunk"),
                                                   static_cast<size_t>(1),
                                                   "stream lane chunk");
        const size_t max_chunk_cells = std::max(energy_chunk_cells, lane_chunk_cells);
        const size_t stream_array = checkedMul(max_chunk_cells, sizeof(double), "stream chunk bytes");

        estimated += checkedMul(stream_array, 8, "stream F/f/F1/f1/source chunk buffers");
        estimated += checkedMul((max_chunk_cells + 511) / 512, sizeof(double), "stream reduction buffer");
        estimated += checkedMul(static_cast<size_t>(grid.Ng), sizeof(double), "physics arrays") * 5;
        estimated += checkedMul(checkedMul(static_cast<size_t>(grid.Ng), static_cast<size_t>(grid.Ng), "stream energy kernels"),
                                3 * sizeof(double),
                                "stream energy kernel bytes");
        estimated += checkedMul(checkedMul(static_cast<size_t>(grid.Ng), n_angle, "stream angle kernel"),
                                checkedMul(n_angle, sizeof(double), "stream angle kernel bytes"),
                                "stream angle kernel bytes");
        estimated += checkedMul(n_angle, sizeof(double), "angle helper arrays") * 4;
        estimated += checkedMul(static_cast<size_t>(grid.Ng), n_angle, "angle factor")
                   * sizeof(cufftDoubleComplex);
    } else {
        estimated += checkedMul(one_phase_array, 5, "F/f/source arrays");
        estimated += one_phase_array;  // SpatialTransportSolver::d_F_tmp
        estimated += checkedMul(one_phase_array, 4, "energy old/new buffers");
        estimated += checkedMul(total_cells, sizeof(cufftDoubleComplex), "angle FFT buffer");

        estimated += checkedMul(checkedMul(static_cast<size_t>(grid.Ng), static_cast<size_t>(grid.Ng), "energy kernels"),
                                3 * sizeof(double),
                                "energy kernel bytes");
        estimated += checkedMul(checkedMul(static_cast<size_t>(grid.Ng), n_angle, "angle kernel"),
                                checkedMul(n_angle, sizeof(double), "angle kernel bytes"),
                                "angle kernel bytes");
        estimated += checkedMul(static_cast<size_t>(grid.Ng), n_angle, "angle diffusion factor")
                   * sizeof(cufftDoubleComplex);
        estimated += checkedMul(static_cast<size_t>(Numerics::MAX_BATCH_SIZE),
                                static_cast<size_t>(grid.Ng),
                                "energy pitched buffers")
                   * sizeof(double) * 6;
    }

    // Leave room for cuFFT/cuBLAS workspaces, allocator fragmentation, contexts,
    // and short-lived host/device buffers created during initialization.
    const size_t safety_margin = static_cast<size_t>(0.15 * static_cast<double>(total_bytes))
                               + static_cast<size_t>(512ULL * 1024ULL * 1024ULL);
    const size_t required_with_margin = estimated + safety_margin;

    std::cout << "Memory estimate:" << std::endl;
    std::cout << "  Mode: "
              << (phys.lite_memory ? "lite-memory primary-only"
                  : (phys.streaming_full ? "streaming-full out-of-core" : "full"))
              << std::endl;
    std::cout << "  One phase-space double array: " << formatGiB(one_phase_array) << std::endl;
    std::cout << "  Estimated current peak: " << formatGiB(estimated)
              << " plus safety margin " << formatGiB(safety_margin) << std::endl;
    std::cout << "  GPU memory free/total: " << formatGiB(free_bytes)
              << " / " << formatGiB(total_bytes) << std::endl;

    if (required_with_margin > free_bytes) {
        std::ostringstream msg;
        msg << "Requested grid needs about " << formatGiB(required_with_margin)
            << " with the current full phase-space layout, but only "
            << formatGiB(free_bytes) << " is free.\n"
            << "Grid was Ny=" << grid.Ny << ", Nz=" << grid.Nz
            << ", Ng=" << grid.Ng << ", Nmu=" << grid.Nmu
            << ", Nom=" << grid.Nom << ".\n"
            << "On a 16 GB GPU, use for example:\n"
            << "  --ny 20 --nz 20 --nmu 20 --nom 20\n"
            << "or\n"
            << "  --ny 40 --nz 40 --nmu 11 --nom 11\n"
            << "For a memory-light primary-only paper-grid development run, add --lite-memory.\n"
            << "For out-of-core full-mode development, add --streaming-full.\n"
            << "A full paper-equivalent grid still requires a streaming/chunked implementation.";
        throw std::runtime_error(msg.str());
    }
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
    for (int i = 0; i <= grid.Ny; i++) h_y[i] = -0.5 * grid.Ly + i * grid.dy;
    for (int i = 0; i <= grid.Nz; i++) h_z[i] = -0.5 * grid.Lz + i * grid.dz;
    for (int i = 0; i <= grid.Ng; i++) h_en[i] = i * grid.dg + 1.0;

    d_y.copyFromHost(h_y.data(), grid.Ny + 1);
    d_z.copyFromHost(h_z.data(), grid.Nz + 1);
    d_en.copyFromHost(h_en.data(), grid.Ng + 1);

    // 初始化角度方向
    int n_angle = grid.Nmu * grid.Nom;
    std::vector<double> h_Omend(n_angle * 2);
    std::vector<double> h_utemp(n_angle), h_vtemp(n_angle);
    std::vector<double> u(grid.Nom), v(grid.Nmu);

    for (int i = 0; i < grid.Nom; i++) u[i] = -0.5 * grid.Lu + (i + 1) * grid.du;
    for (int j = 0; j < grid.Nmu; j++) v[j] = -0.5 * grid.Lv + (j + 1) * grid.dv;

    for (int j = 0; j < grid.Nmu; j++) {
        for (int i = 0; i < grid.Nom; i++) {
            int idx = j * grid.Nom + i;
            h_Omend[idx*2] = -v[grid.Nmu - 1 - j];
            h_Omend[idx*2+1] = -u[i];
            h_utemp[idx] = 2.0 * cos(Physics::PI * (i + 1) / (grid.Nom + 1)) - 2.0;
            h_vtemp[idx] = 2.0 * cos(Physics::PI * (j + 1) / (grid.Nmu + 1)) - 2.0;
        }
    }

    double omega1_abs_center = std::numeric_limits<double>::max();
    double omega2_abs_center = std::numeric_limits<double>::max();
    for (int idx = 0; idx < n_angle; ++idx) {
        omega1_abs_center = std::min(omega1_abs_center, std::abs(h_Omend[idx * 2 + 0]));
        omega2_abs_center = std::min(omega2_abs_center, std::abs(h_Omend[idx * 2 + 1]));
    }
    std::cout << "Discrete beam angle center: |omega_y|=" << omega1_abs_center
              << ", |omega_z|=" << omega2_abs_center << std::endl;

    d_Omend.copyFromHost(h_Omend.data(), n_angle * 2);
    d_utemp.copyFromHost(h_utemp.data(), n_angle);
    d_vtemp.copyFromHost(h_vtemp.data(), n_angle);

    // 计算物理量
    computeStoppingPower();
    computeStraggling();
    computeTransportCrossSection();

    if (!phys.lite_memory) {
        // 初始化内核
        initializeKernels();
    }

    // 初始化分布
    initializeDistribution();

    if (!phys.lite_memory) {
        // 初始化子求解器
        angle_solver->initializeDiffusionFactor(d_sig_trg.data(), grid.du, grid.dv, grid.dt,
                                                phys.density);
        transport_solver->initializeWaveNumbers();
        transport_solver->initializeTransportFactor(d_Omend.data(), grid.dt, n_angle);
    }
}

void BFPnSolver::computeStoppingPower() {
    std::vector<double> h_S_s(grid.Ng + 1);
    const auto composition = materialComposition(phys.material_name);

    std::vector<ElementFraction> water = materialComposition("water");

    for (int i = 0; i <= grid.Ng; i++) {
        double E = (i == 0) ? 1.0 : (i == grid.Ng) ? grid.Lg + 1.0 : (i - 0.5) * grid.dg + 1.0;

        double weighted = 0.0;
        for (const auto& elem : composition) {
            weighted += elem.weight * betheStoppingElement(E, elem.Z);
        }

        double water_weighted = 0.0;
        for (const auto& elem : water) {
            water_weighted += elem.weight * betheStoppingElement(E, elem.Z);
        }

        double beta_2 = E * (E + 2.0 * 938.3) / pow(E + 938.3, 2);
        double F_beta = log(1.02e6 * beta_2 / (1.0 - beta_2)) - beta_2 - 4.31;
        double water_legacy = 0.170 / beta_2 * F_beta + 0.02;
        double scale = (water_weighted > Numerics::EPSILON) ? weighted / water_weighted : 1.0;
        h_S_s[i] = water_legacy * scale;
    }

    d_S_s.copyFromHost(h_S_s.data(), grid.Ng + 1);
}

void BFPnSolver::computeStraggling() {
    std::vector<double> h_T_c(grid.Ng);

    for (int i = 0; i < grid.Ng; i++) {
        double E = (i + 0.5) * grid.dg + 1.0;
        double v = sqrt(2.0 * E * Physics::MEV_TO_J / Physics::MP);
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
        double v = sqrt(2.0 * E * Physics::MEV_TO_J / Physics::MP);
        double beta = v / (3.0 * Physics::C_LIGHT);

        double Zt = 3;  // 目标原子序数 (锂)
        double eta = pow(Zt, 2.0/3.0) * pow(Physics::ALPHA, 2)
                    * pow(Physics::ME / Physics::MP, 2) / (beta * beta);

        double mid = log((eta + 1.0) / eta) - 1.0 / (eta + 1.0);
        h_sig_trg[i] = 2.0 * Physics::PI * 3.34 * pow(Physics::HBARC, 2)
                      * pow(Physics::ALPHA, 2) / 4.0 * Zt * Zt
                      / (E * E) * mid / 1000.0;
    }

    d_sig_trg.copyFromHost(h_sig_trg.data(), grid.Ng);
}

void BFPnSolver::initializeKernels() {
    std::vector<double> h_ker_e(grid.Ng * grid.Ng, 0.0);

    std::vector<double> h_mu(grid.Ng * 3);
    std::vector<double> h_sigma(grid.Ng);
    d_mu.copyToHost(h_mu.data(), grid.Ng * 3);
    d_sigma_c.copyToHost(h_sigma.data(), grid.Ng);

    std::vector<double> initial_ker_e(grid.Ng * grid.Ng, 0.0);
    for (int j = 0; j < grid.Ng - 1; j++) {
        for (int i = 0; i < grid.Ng - 1; i++) {
            if (h_mu[j*3+2] == 0) {
                initial_ker_e[i*grid.Ng + j] = 0.0;
            } else if (i < j) {
                initial_ker_e[i*grid.Ng + j] = 0.0;
            } else {
                initial_ker_e[i*grid.Ng + j] = grid.dg / h_mu[j*3+2]
                    * exp(-(i - j + 0.5) * grid.dg / h_mu[j*3+2]);
            }
        }
    }

    std::vector<double> shifted(grid.Ng * grid.Ng, 0.0);
    for (int j = 0; j < grid.Ng - 1; ++j) {
        for (int i = 0; i < grid.Ng; ++i) {
            if (i == j) {
                shifted[i * grid.Ng + j] = 0.0;
            } else if (i < j) {
                shifted[i * grid.Ng + j] = initial_ker_e[i * grid.Ng + j];
            } else {
                shifted[i * grid.Ng + j] = initial_ker_e[(i - 1) * grid.Ng + j];
            }
        }
    }

    for (int row = 0; row < grid.Ng; ++row) {
        for (int col = 0; col < grid.Ng; ++col) {
            double value = shifted[(grid.Ng - 1 - row) * grid.Ng + (grid.Ng - 1 - col)];
            if (std::abs(value) < Numerics::SPARSE_THRESHOLD) {
                value = 0.0;
            }
            h_ker_e[row * grid.Ng + col] = value * h_sigma[col];
        }
    }

    d_ker_e.copyFromHost(h_ker_e.data(), grid.Ng * grid.Ng);

    std::vector<double> h_ker_e1(grid.Ng * grid.Ng, 0.0);
    std::vector<double> h_ker_e2(grid.Ng * grid.Ng, 0.0);
    for (int row = 0; row < grid.Ng; ++row) {
        for (int col = 0; col < grid.Ng; ++col) {
            const double cur = h_ker_e[row * grid.Ng + col];
            const double next = (row + 1 < grid.Ng) ? h_ker_e[(row + 1) * grid.Ng + col] : 0.0;
            h_ker_e1[row * grid.Ng + col] = 0.5 * (cur + next);
            h_ker_e2[row * grid.Ng + col] = 0.5 * (-cur + next);
        }
    }
    d_ker_e1.copyFromHost(h_ker_e1.data(), grid.Ng * grid.Ng);
    d_ker_e2.copyFromHost(h_ker_e2.data(), grid.Ng * grid.Ng);

    const int n_angle = grid.Nmu * grid.Nom;
    std::vector<double> h_omega(n_angle * 2);
    d_Omend.copyToHost(h_omega.data(), n_angle * 2);

    std::vector<double> angles(n_angle * n_angle, 0.0);
    for (int out = 0; out < n_angle; ++out) {
        const double out_y = h_omega[out * 2 + 0];
        const double out_z = h_omega[out * 2 + 1];
        const double out_norm = std::sqrt(1.0 + out_y * out_y + out_z * out_z);
        for (int in = 0; in < n_angle; ++in) {
            const double in_y = h_omega[in * 2 + 0];
            const double in_z = h_omega[in * 2 + 1];
            const double in_norm = std::sqrt(1.0 + in_y * in_y + in_z * in_z);
            double cos_theta = (1.0 + in_y * out_y + in_z * out_z) /
                               std::max(in_norm * out_norm, Numerics::EPSILON);
            cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
            angles[in * n_angle + out] = std::abs(std::acos(cos_theta) * 180.0 / Physics::PI);
        }
    }

    std::vector<double> h_ker_v(static_cast<size_t>(grid.Ng) * n_angle * n_angle, 0.0);
    for (int ek = 0; ek < grid.Ng - 1; ++ek) {
        if (h_mu[ek * 3 + 0] == 0.0 || h_mu[ek * 3 + 1] == 0.0) {
            continue;
        }

        for (int out = 0; out < n_angle; ++out) {
            double col_sum = 0.0;
            for (int in = 0; in < n_angle; ++in) {
                const double theta = angles[in * n_angle + out];
                const double in_y = h_omega[in * 2 + 0];
                const double in_z = h_omega[in * 2 + 1];
                double value = std::pow(theta, h_mu[ek * 3 + 0] - 1.0)
                             * std::exp(-theta / h_mu[ek * 3 + 1])
                             * std::pow(1.0 + in_y * in_y + in_z * in_z, -1.5);
                if (!std::isfinite(value)) {
                    value = 0.0;
                }
                h_ker_v[(static_cast<size_t>(ek) * n_angle + in) * n_angle + out] = value;
                col_sum += value;
            }

            if (col_sum > Numerics::EPSILON) {
                for (int in = 0; in < n_angle; ++in) {
                    double& value = h_ker_v[(static_cast<size_t>(ek) * n_angle + in) * n_angle + out];
                    value /= col_sum;
                    if (value < Numerics::KERNEL_THRESHOLD) {
                        value = 0.0;
                    }
                }
            }
        }
    }

    d_ker_v.copyFromHost(h_ker_v.data(), static_cast<size_t>(grid.Ng) * n_angle * n_angle);
}

void BFPnSolver::initializeDistribution() {
    size_t nyz = (grid.Ny + 1) * (grid.Nz + 1);
    size_t n_angle = grid.Nmu * grid.Nom;
    double omega1_abs_center = std::numeric_limits<double>::max();
    double omega2_abs_center = std::numeric_limits<double>::max();
    for (int j = 0; j < grid.Nmu; ++j) {
        double v = -0.5 * grid.Lv + (j + 1) * grid.dv;
        omega1_abs_center = std::min(omega1_abs_center, std::abs(v));
    }
    for (int i = 0; i < grid.Nom; ++i) {
        double u = -0.5 * grid.Lu + (i + 1) * grid.du;
        omega2_abs_center = std::min(omega2_abs_center, std::abs(u));
    }

    if (phys.lite_memory) {
        size_t total = nyz * grid.Ng * n_angle;
        constexpr int block_size = 256;
        int grid_size = static_cast<int>((total + block_size - 1) / block_size);
        initializeLiteDistributionKernel<<<grid_size, block_size>>>(
            d_F.data(),
            d_Omend.data(),
            grid.Ny,
            grid.Nz,
            grid.Ng,
            static_cast<int>(n_angle),
            grid.Ly,
            grid.Lz,
            grid.dy,
            grid.dz,
            grid.dg,
            phys.a_1,
            phys.a_2,
            phys.sig_E,
            phys.beam_energy,
            omega1_abs_center,
            omega2_abs_center
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        return;
    }

    if (phys.streaming_full) {
        initializeStreamingStore();

        int energy_chunk = std::max(1, phys.streaming_energy_chunk);
        constexpr int block_size = 256;
        for (int e0 = 0; e0 < grid.Ng; e0 += energy_chunk) {
            int ecount = std::min(energy_chunk, grid.Ng - e0);
            size_t chunk_size = static_cast<size_t>(ecount) * nyz * n_angle;
            int grid_size = static_cast<int>((chunk_size + block_size - 1) / block_size);
            initializeStreamingDistributionKernel<<<grid_size, block_size>>>(
                d_stream_F.data(),
                d_Omend.data(),
                grid.Ny,
                grid.Nz,
                grid.Ng,
                e0,
                ecount,
                static_cast<int>(n_angle),
                grid.Ly,
                grid.Lz,
                grid.dy,
                grid.dz,
                grid.dg,
                phys.a_1,
                phys.a_2,
                phys.sig_E,
                phys.beam_energy,
                omega1_abs_center,
                omega2_abs_center
            );
            CUDA_CHECK(cudaGetLastError());
            size_t host_offset = static_cast<size_t>(e0) * nyz * n_angle;
            std::vector<double> h_chunk(chunk_size);
            CUDA_CHECK(cudaMemcpy(h_chunk.data(),
                                  d_stream_F.data(),
                                  chunk_size * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            writeStore(stream_F_path, host_offset, h_chunk.data(), chunk_size);
        }
        return;
    }

    std::vector<double> h_F(nyz * grid.Ng * n_angle, 0.0);
    std::vector<double> h_Omend(n_angle * 2);
    d_Omend.copyToHost(h_Omend.data(), n_angle * 2);

    // 初始化f_1 (空间-角度)
    for (int k = 0; k < n_angle; k++) {
        double omega1 = h_Omend[k * 2 + 0];
        double omega2 = h_Omend[k * 2 + 1];

        for (int j = 0; j <= grid.Nz; j++) {
            for (int i = 0; i <= grid.Ny; i++) {
                double y = -0.5 * grid.Ly + i * grid.dy;
                double z = -0.5 * grid.Lz + j * grid.dz;

                double domega1 = std::abs(omega1) - omega1_abs_center;
                double domega2 = std::abs(omega2) - omega2_abs_center;
                double f1 = exp(-(phys.a_1 * y * y + phys.a_2 * domega1 * domega1))
                          * exp(-(phys.a_1 * z * z + phys.a_2 * domega2 * domega2));

                // 初始化能量分布 (高斯)
                for (int ek = 0; ek < grid.Ng; ek++) {
                    double E = (ek + 0.5) * grid.dg + 1.0;
                    double f2 = 1.0 / sqrt(2.0 * Physics::PI) / phys.sig_E
                               * exp(-pow((E - phys.beam_energy) / phys.sig_E / sqrt(2.0), 2));

                    int idx = (ek * nyz + j * (grid.Ny+1) + i) * n_angle + k;
                    h_F[idx] = f1 * f2;
                }
            }
        }
    }

    d_F.copyFromHost(h_F.data(), nyz * grid.Ng * n_angle);
    if (!phys.lite_memory) {
        d_f_F.setZero();
        d_F1.setZero();
        d_f_F1.setZero();
    }
}

void BFPnSolver::initializeStreamingStore() {
    const size_t nyz = static_cast<size_t>(grid.Ny + 1) * (grid.Nz + 1);
    const size_t n_angle = static_cast<size_t>(grid.Nmu) * grid.Nom;
    const size_t total = nyz * static_cast<size_t>(grid.Ng) * n_angle;
    const size_t bytes = total * sizeof(double);

    std::string dir = phys.streaming_dir.empty() ? std::string("/tmp") : phys.streaming_dir;
    if (!dir.empty() && dir.back() == '/') {
        dir.pop_back();
    }
    if (mkdir(dir.c_str(), 0777) != 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create streaming directory: " + dir);
    }
    std::string prefix = "bfpn_stream_" + std::to_string(static_cast<long long>(getpid())) + "_";
    stream_F_path = dir + "/" + prefix + "F.bin";
    stream_f_F_path = dir + "/" + prefix + "f_F.bin";
    stream_F1_path = dir + "/" + prefix + "F1.bin";
    stream_f_F1_path = dir + "/" + prefix + "f_F1.bin";

    const size_t effective_lane_chunk = alignedLaneChunk(
        static_cast<size_t>(std::max(1, phys.streaming_lane_chunk)),
        n_angle);
    const bool needs_source_cache = !phys.primary_only && effective_lane_chunk < nyz * n_angle;
    if (needs_source_cache) {
        stream_source_path = dir + "/" + prefix + "source.bin";
    }

    size_t file_count = phys.primary_only ? 2 : (needs_source_cache ? 5 : 4);
    std::cout << "Streaming backing store: " << formatGiB(bytes * file_count)
              << " across " << file_count << " files in " << dir << std::endl;

    auto create_zero_file = [bytes](const std::string& path) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("Cannot create streaming file: " + path);
        }
        if (bytes > 0) {
            out.seekp(static_cast<std::streamoff>(bytes - 1));
            char zero = 0;
            out.write(&zero, 1);
        }
        if (!out) {
            throw std::runtime_error("Cannot size streaming file: " + path);
        }
    };

    create_zero_file(stream_F_path);
    create_zero_file(stream_f_F_path);
    if (!phys.primary_only) {
        create_zero_file(stream_F1_path);
        create_zero_file(stream_f_F1_path);
        if (needs_source_cache) {
            create_zero_file(stream_source_path);
        }
    }
}

void BFPnSolver::cleanupStreamingStore() {
    for (auto& item : stream_fds) {
        if (item.second >= 0) {
            close(item.second);
        }
    }
    stream_fds.clear();

    auto remove_if_present = [](const std::string& path) {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    };
    remove_if_present(stream_F_path);
    remove_if_present(stream_f_F_path);
    remove_if_present(stream_F1_path);
    remove_if_present(stream_f_F1_path);
    remove_if_present(stream_source_path);
}

int BFPnSolver::getStoreFd(const std::string& path) const {
    auto it = stream_fds.find(path);
    if (it != stream_fds.end()) {
        return it->second;
    }
    int fd = open(path.c_str(), O_RDWR);
    if (fd < 0) {
        throw std::runtime_error("Cannot open streaming backing file: " + path);
    }
    stream_fds[path] = fd;
    return fd;
}

void BFPnSolver::readStore(const std::string& path,
                           size_t element_offset,
                           double* dst,
                           size_t count) const {
    int fd = getStoreFd(path);
    char* out = reinterpret_cast<char*>(dst);
    size_t bytes = count * sizeof(double);
    off_t offset = static_cast<off_t>(element_offset * sizeof(double));
    size_t done = 0;
    while (done < bytes) {
        ssize_t got = pread(fd, out + done, bytes - done, offset + static_cast<off_t>(done));
        if (got <= 0) {
            throw std::runtime_error("Cannot read streaming file chunk: " + path);
        }
        done += static_cast<size_t>(got);
    }
}

void BFPnSolver::writeStore(const std::string& path,
                            size_t element_offset,
                            const double* src,
                            size_t count) const {
    int fd = getStoreFd(path);
    const char* in = reinterpret_cast<const char*>(src);
    size_t bytes = count * sizeof(double);
    off_t offset = static_cast<off_t>(element_offset * sizeof(double));
    size_t done = 0;
    while (done < bytes) {
        ssize_t put = pwrite(fd, in + done, bytes - done, offset + static_cast<off_t>(done));
        if (put <= 0) {
            throw std::runtime_error("Cannot write streaming file chunk: " + path);
        }
        done += static_cast<size_t>(put);
    }
}

void BFPnSolver::solve(double tFinal) {
    if (phys.streaming_full) {
        solveStreamingFull(tFinal);
        return;
    }

    std::vector<double> idd_history;
    std::vector<double> proxy_history;
    std::vector<double> energy_moment_history;
    std::vector<char> spot_saved(phys.spot_depths.size(), 0);
    double t = 0.0;
    int step = 0;
    int max_steps = static_cast<int>(std::ceil(tFinal / grid.dt - 1e-12));
    int ping = 0, pong = 1;

    std::cout << "Starting BFPn solver..." << std::endl;
    std::cout << "Grid: " << grid.Nx << "x" << grid.Ny << "x" << grid.Nz
              << ", Energy: " << grid.Ng << ", Angle: " << grid.Nmu << "x" << grid.Nom << std::endl;
    if (phys.lite_memory) {
        std::cout << "Mode: lite-memory primary-only (no secondary, no DG slope, no YZ/angular transport)" << std::endl;
    }

    while (step < max_steps) {
        if (phys.lite_memory) {
            stepLitePrimary();

            double idd = computeIntegratedDepthDoseLite();
            idd_history.push_back(idd);
            proxy_history.push_back(0.0);

            t += grid.dt;
            step++;

            if (step % 100 == 0) {
                std::cout << "Step " << step << ", t = " << t << "/" << tFinal
                          << ", IDD = " << idd << std::endl;
            }
            continue;
        }

        if (phys.energy_only) {
            stepPrimaryEnergyOnly();
        } else {
            stepPrimary(ping, t);
        }

        // Secondary质子: 演化 (包含源项)
        if (!phys.energy_only && !phys.primary_only && step > 0) {
            stepSecondary(ping, pong, t);
            std::swap(ping, pong);
        }

        double idd = computeIntegratedDepthDose();
        double proxy = computeScalarDoseProxy();
        idd_history.push_back(idd);
        proxy_history.push_back(proxy);
        if (phys.save_energy_moments) {
            std::vector<double> moments = computeEnergyMoments();
            energy_moment_history.insert(energy_moment_history.end(),
                                         moments.begin(),
                                         moments.end());
        }

        t += grid.dt;
        step++;

        for (size_t i = 0; i < phys.spot_depths.size(); ++i) {
            if (!spot_saved[i] && t + 0.5 * grid.dt >= phys.spot_depths[i]) {
                saveSpotPlane(phys.spot_depths[i], t);
                spot_saved[i] = 1;
            }
        }

        if (step % 100 == 0) {
            std::cout << "Step " << step << ", t = " << t << "/" << tFinal
                      << ", IDD = " << idd << std::endl;
        }
    }

    saveResults(idd_history, "idd_output.txt");
    saveResults(proxy_history, "dose_output.txt");
    if (phys.save_energy_moments) {
        saveEnergyMoments(energy_moment_history, "energy_moments.txt");
    }
    std::cout << "Solver completed. Paper-style IDD saved to idd_output.txt" << std::endl;
    std::cout << "Diagnostic scalar proxy saved to dose_output.txt" << std::endl;
}

void BFPnSolver::streamingEnergyStep(const std::string& F_path,
                                     const std::string& f_F_path,
                                     const std::string* primary_F_path,
                                     const std::string* primary_f_F_path,
                                     bool is_secondary,
                                     double dt,
                                     double* idd_accum,
                                     bool reuse_secondary_source,
                                     const std::string* secondary_source_cache_path,
                                     bool read_cached_secondary_source,
                                     bool write_cached_secondary_source) {
    const size_t nyz = static_cast<size_t>(grid.Ny + 1) * (grid.Nz + 1);
    const size_t n_angle = static_cast<size_t>(grid.Nmu) * grid.Nom;
    const size_t n_per_E = nyz * n_angle;
    const size_t requested_lane_chunk = static_cast<size_t>(std::max(1, phys.streaming_lane_chunk));
    const size_t lane_chunk = alignedLaneChunk(requested_lane_chunk, n_angle);

    constexpr int block_size = 256;
    const size_t max_lanes = std::min(lane_chunk, n_per_E);
    const size_t max_chunk_size = static_cast<size_t>(grid.Ng) * max_lanes;
    h_stream_lane.resize(max_chunk_size);
    h_stream_f_lane.resize(max_chunk_size);
    if (is_secondary && primary_F_path && primary_f_F_path) {
        h_stream_primary_lane.resize(max_chunk_size);
        h_stream_primary_f_lane.resize(max_chunk_size);
    }
    cudaStream_t stream = stream_primary[0];

    for (size_t lane0 = 0; lane0 < n_per_E; lane0 += lane_chunk) {
        int lanes = static_cast<int>(std::min(lane_chunk, n_per_E - lane0));
        size_t chunk_size = static_cast<size_t>(grid.Ng) * lanes;
        const bool full_lane_chunk = lane0 == 0 && static_cast<size_t>(lanes) == n_per_E;
        auto read_lane_chunk = [&](const std::string& path, double* dst) {
            if (full_lane_chunk) {
                readStore(path, 0, dst, chunk_size);
            } else {
                for (int ek = 0; ek < grid.Ng; ++ek) {
                    size_t src = static_cast<size_t>(ek) * n_per_E + lane0;
                    size_t out = static_cast<size_t>(ek) * lanes;
                    readStore(path, src, dst + out, lanes);
                }
            }
        };
        auto write_lane_chunk = [&](const std::string& path, const double* src) {
            if (full_lane_chunk) {
                writeStore(path, 0, src, chunk_size);
            } else {
                for (int ek = 0; ek < grid.Ng; ++ek) {
                    size_t in = static_cast<size_t>(ek) * lanes;
                    size_t dst = static_cast<size_t>(ek) * n_per_E + lane0;
                    writeStore(path, dst, src + in, lanes);
                }
            }
        };

        read_lane_chunk(F_path, h_stream_lane.data());
        read_lane_chunk(f_F_path, h_stream_f_lane.data());
        CUDA_CHECK(cudaMemcpyAsync(d_stream_old_F.data(),
                                   h_stream_lane.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(d_stream_old_f_F.data(),
                                   h_stream_f_lane.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyHostToDevice,
                                   stream));

        const bool can_reuse_source = is_secondary && reuse_secondary_source &&
                                      primary_F_path && primary_f_F_path &&
                                      full_lane_chunk &&
                                      chunk_size <= d_stream_primary_F.getSize();
        const bool can_read_cached_source = is_secondary && !can_reuse_source &&
                                            read_cached_secondary_source &&
                                            secondary_source_cache_path;
        const bool can_write_cached_source = is_secondary && !can_reuse_source &&
                                             write_cached_secondary_source &&
                                             secondary_source_cache_path;
        if (can_read_cached_source) {
            read_lane_chunk(*secondary_source_cache_path, h_stream_primary_lane.data());
            CUDA_CHECK(cudaMemcpyAsync(d_stream_primary_F.data(),
                                       h_stream_primary_lane.data(),
                                       chunk_size * sizeof(double),
                                       cudaMemcpyHostToDevice,
                                       stream));
        } else if (is_secondary && primary_F_path && primary_f_F_path && !can_reuse_source) {
            read_lane_chunk(*primary_F_path, h_stream_primary_lane.data());
            read_lane_chunk(*primary_f_F_path, h_stream_primary_f_lane.data());
            CUDA_CHECK(cudaMemcpyAsync(d_stream_F1.data(),
                                       h_stream_primary_lane.data(),
                                       chunk_size * sizeof(double),
                                       cudaMemcpyHostToDevice,
                                       stream));
            CUDA_CHECK(cudaMemcpyAsync(d_stream_primary_f_F.data(),
                                       h_stream_primary_f_lane.data(),
                                       chunk_size * sizeof(double),
                                       cudaMemcpyHostToDevice,
                                       stream));
            CUDA_CHECK(cudaMemsetAsync(d_stream_primary_F.data(), 0,
                                       chunk_size * sizeof(double),
                                       stream));

            int source_grid = static_cast<int>((chunk_size + block_size - 1) / block_size);
            streamingSecondarySourceKernel<<<source_grid, block_size, 0, stream>>>(
                d_stream_F1.data(),
                d_stream_primary_f_F.data(),
                d_stream_primary_F.data(),
                d_ker_e1.data(),
                d_ker_e2.data(),
                d_ker_v.data(),
                grid.Ng,
                lanes,
                lane0,
                static_cast<int>(n_angle),
                grid.dg
            );
            CUDA_CHECK(cudaGetLastError());
        }

        int grid_size = static_cast<int>((lanes + block_size - 1) / block_size);
        streamingEnergyDgCnKernel<<<grid_size, block_size, 0, stream>>>(
            d_stream_old_F.data(),
            d_stream_old_f_F.data(),
            d_stream_F.data(),
            d_stream_f_F.data(),
            is_secondary ? d_stream_primary_F.data() : nullptr,
            d_S_s.data(),
            d_sigma_c.data(),
            d_T_c.data(),
            grid.Ng,
            lanes,
            dt,
            grid.dg,
            phys.energy_density,
            phys.legacy_energy,
            phys.eq15_straggling,
            is_secondary
        );
        CUDA_CHECK(cudaGetLastError());

        if (idd_accum) {
            size_t num_blocks = (chunk_size + block_size * 2 - 1) / (block_size * 2);
            if (num_blocks > d_reduction_sums.getSize()) {
                throw std::runtime_error("Streaming energy IDD reduction buffer is too small");
            }
            pairDepthDoseLaneKernel<<<static_cast<int>(num_blocks),
                                      block_size,
                                      block_size * sizeof(double),
                                      stream>>>(
                d_stream_F.data(),
                d_stream_f_F.data(),
                d_S_s.data(),
                d_reduction_sums.data(),
                grid.Ng,
                lanes,
                grid.dg,
                phys.density,
                grid.du * grid.dv * grid.dy * grid.dz,
                chunk_size
            );
            CUDA_CHECK(cudaGetLastError());
            std::vector<double> h_block_sums(num_blocks);
            CUDA_CHECK(cudaMemcpyAsync(h_block_sums.data(),
                                       d_reduction_sums.data(),
                                       num_blocks * sizeof(double),
                                       cudaMemcpyDeviceToHost,
                                       stream));
            CUDA_CHECK(cudaMemcpyAsync(h_stream_lane.data(),
                                       d_stream_F.data(),
                                       chunk_size * sizeof(double),
                                       cudaMemcpyDeviceToHost,
                                       stream));
            CUDA_CHECK(cudaMemcpyAsync(h_stream_f_lane.data(),
                                       d_stream_f_F.data(),
                                       chunk_size * sizeof(double),
                                       cudaMemcpyDeviceToHost,
                                       stream));
            if (can_write_cached_source) {
                CUDA_CHECK(cudaMemcpyAsync(h_stream_primary_lane.data(),
                                           d_stream_primary_F.data(),
                                           chunk_size * sizeof(double),
                                           cudaMemcpyDeviceToHost,
                                           stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (double partial : h_block_sums) {
                *idd_accum += partial;
            }
        } else {
            CUDA_CHECK(cudaMemcpyAsync(h_stream_lane.data(),
                                       d_stream_F.data(),
                                       chunk_size * sizeof(double),
                                       cudaMemcpyDeviceToHost,
                                       stream));
            CUDA_CHECK(cudaMemcpyAsync(h_stream_f_lane.data(),
                                       d_stream_f_F.data(),
                                       chunk_size * sizeof(double),
                                       cudaMemcpyDeviceToHost,
                                       stream));
            if (can_write_cached_source) {
                CUDA_CHECK(cudaMemcpyAsync(h_stream_primary_lane.data(),
                                           d_stream_primary_F.data(),
                                           chunk_size * sizeof(double),
                                           cudaMemcpyDeviceToHost,
                                           stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        if (can_write_cached_source) {
            write_lane_chunk(*secondary_source_cache_path, h_stream_primary_lane.data());
        }
        write_lane_chunk(F_path, h_stream_lane.data());
        write_lane_chunk(f_F_path, h_stream_f_lane.data());
    }
}

void BFPnSolver::streamingTransportStep(const std::string& F_path, double dt, bool preserve_sign) {
    const size_t nyz = static_cast<size_t>(grid.Ny + 1) * (grid.Nz + 1);
    const size_t n_angle = static_cast<size_t>(grid.Nmu) * grid.Nom;
    const size_t per_energy = nyz * n_angle;
    const int energy_chunk = std::max(1, phys.streaming_energy_chunk);
    const size_t max_chunk_size = static_cast<size_t>(std::min(energy_chunk, grid.Ng)) * per_energy;
    h_stream_chunk.resize(max_chunk_size);
    cudaStream_t stream = stream_primary[0];

    for (int e0 = 0; e0 < grid.Ng; e0 += energy_chunk) {
        int ecount = std::min(energy_chunk, grid.Ng - e0);
        size_t chunk_size = static_cast<size_t>(ecount) * per_energy;
        size_t offset = static_cast<size_t>(e0) * per_energy;

        readStore(F_path, offset, h_stream_chunk.data(), chunk_size);
        CUDA_CHECK(cudaMemcpyAsync(d_stream_F.data(),
                                   h_stream_chunk.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyHostToDevice,
                                   stream));
        transport_solver->solve(d_stream_F.data(),
                                d_Omend.data(),
                                ecount,
                                static_cast<int>(n_angle),
                                dt,
                                stream,
                                preserve_sign);
        CUDA_CHECK(cudaMemcpyAsync(h_stream_chunk.data(),
                                   d_stream_F.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        writeStore(F_path, offset, h_stream_chunk.data(), chunk_size);
    }
}

void BFPnSolver::streamingTransportPairStep(const std::string& F_path,
                                            const std::string& f_F_path,
                                            double dt) {
    const size_t nyz = static_cast<size_t>(grid.Ny + 1) * (grid.Nz + 1);
    const size_t n_angle = static_cast<size_t>(grid.Nmu) * grid.Nom;
    const size_t per_energy = nyz * n_angle;
    const int energy_chunk = std::max(1, phys.streaming_energy_chunk);
    const size_t max_chunk_size = static_cast<size_t>(std::min(energy_chunk, grid.Ng)) * per_energy;
    h_stream_chunk.resize(max_chunk_size);
    h_stream_f_lane.resize(max_chunk_size);
    cudaStream_t stream = stream_primary[0];

    for (int e0 = 0; e0 < grid.Ng; e0 += energy_chunk) {
        int ecount = std::min(energy_chunk, grid.Ng - e0);
        size_t chunk_size = static_cast<size_t>(ecount) * per_energy;
        size_t offset = static_cast<size_t>(e0) * per_energy;

        readStore(F_path, offset, h_stream_chunk.data(), chunk_size);
        readStore(f_F_path, offset, h_stream_f_lane.data(), chunk_size);
        CUDA_CHECK(cudaMemcpyAsync(d_stream_F.data(),
                                   h_stream_chunk.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(d_stream_f_F.data(),
                                   h_stream_f_lane.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyHostToDevice,
                                   stream));
        transport_solver->solve(d_stream_F.data(),
                                d_Omend.data(),
                                ecount,
                                static_cast<int>(n_angle),
                                dt,
                                stream,
                                false);
        transport_solver->solve(d_stream_f_F.data(),
                                d_Omend.data(),
                                ecount,
                                static_cast<int>(n_angle),
                                dt,
                                stream,
                                true);
        CUDA_CHECK(cudaMemcpyAsync(h_stream_chunk.data(),
                                   d_stream_F.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(h_stream_f_lane.data(),
                                   d_stream_f_F.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        writeStore(F_path, offset, h_stream_chunk.data(), chunk_size);
        writeStore(f_F_path, offset, h_stream_f_lane.data(), chunk_size);
    }
}

void BFPnSolver::streamingAngleStep(const std::string& F_path, double dt) {
    const size_t nyz = static_cast<size_t>(grid.Ny + 1) * (grid.Nz + 1);
    const size_t n_angle = static_cast<size_t>(grid.Nmu) * grid.Nom;
    const size_t per_energy = nyz * n_angle;
    const int energy_chunk = std::max(1, phys.streaming_energy_chunk);
    const size_t max_chunk_size = static_cast<size_t>(std::min(energy_chunk, grid.Ng)) * per_energy;
    h_stream_chunk.resize(max_chunk_size);
    cudaStream_t stream = stream_primary[0];

    for (int e0 = 0; e0 < grid.Ng; e0 += energy_chunk) {
        int ecount = std::min(energy_chunk, grid.Ng - e0);
        size_t chunk_size = static_cast<size_t>(ecount) * per_energy;
        size_t offset = static_cast<size_t>(e0) * per_energy;

        readStore(F_path, offset, h_stream_chunk.data(), chunk_size);
        CUDA_CHECK(cudaMemcpyAsync(d_stream_F.data(),
                                   h_stream_chunk.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyHostToDevice,
                                   stream));
        angle_solver->solve(d_stream_F.data(),
                            d_sig_trg.data() + e0,
                            static_cast<int>(nyz),
                            ecount,
                            dt,
                            phys.density,
                            stream);
        CUDA_CHECK(cudaMemcpyAsync(h_stream_chunk.data(),
                                   d_stream_F.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        writeStore(F_path, offset, h_stream_chunk.data(), chunk_size);
    }
}

void BFPnSolver::streamingAnglePairStep(const std::string& F_path,
                                        const std::string& f_F_path,
                                        double dt) {
    const size_t nyz = static_cast<size_t>(grid.Ny + 1) * (grid.Nz + 1);
    const size_t n_angle = static_cast<size_t>(grid.Nmu) * grid.Nom;
    const size_t per_energy = nyz * n_angle;
    const int energy_chunk = std::max(1, phys.streaming_energy_chunk);
    const size_t max_chunk_size = static_cast<size_t>(std::min(energy_chunk, grid.Ng)) * per_energy;
    h_stream_chunk.resize(max_chunk_size);
    h_stream_f_lane.resize(max_chunk_size);
    cudaStream_t stream = stream_primary[0];

    for (int e0 = 0; e0 < grid.Ng; e0 += energy_chunk) {
        int ecount = std::min(energy_chunk, grid.Ng - e0);
        size_t chunk_size = static_cast<size_t>(ecount) * per_energy;
        size_t offset = static_cast<size_t>(e0) * per_energy;

        readStore(F_path, offset, h_stream_chunk.data(), chunk_size);
        readStore(f_F_path, offset, h_stream_f_lane.data(), chunk_size);
        CUDA_CHECK(cudaMemcpyAsync(d_stream_F.data(),
                                   h_stream_chunk.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(d_stream_f_F.data(),
                                   h_stream_f_lane.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyHostToDevice,
                                   stream));
        angle_solver->solve(d_stream_F.data(),
                            d_sig_trg.data() + e0,
                            static_cast<int>(nyz),
                            ecount,
                            dt,
                            phys.density,
                            stream);
        angle_solver->solve(d_stream_f_F.data(),
                            d_sig_trg.data() + e0,
                            static_cast<int>(nyz),
                            ecount,
                            dt,
                            phys.density,
                            stream);
        CUDA_CHECK(cudaMemcpyAsync(h_stream_chunk.data(),
                                   d_stream_F.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(h_stream_f_lane.data(),
                                   d_stream_f_F.data(),
                                   chunk_size * sizeof(double),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        writeStore(F_path, offset, h_stream_chunk.data(), chunk_size);
        writeStore(f_F_path, offset, h_stream_f_lane.data(), chunk_size);
    }
}

void BFPnSolver::solveStreamingFull(double tFinal) {
    std::vector<std::pair<double, double>> idd_history;
    std::vector<std::pair<double, double>> proxy_history;
    std::vector<char> spot_saved(phys.spot_depths.size(), 0);
    const double half_dt = 0.5 * grid.dt;
    int max_steps = static_cast<int>(std::ceil(tFinal / grid.dt - 1e-12));
    int idd_stride = std::max(1, phys.idd_stride);
    const size_t n_angle = static_cast<size_t>(grid.Nmu) * grid.Nom;
    const size_t n_per_E = static_cast<size_t>(grid.Ny + 1) * (grid.Nz + 1) * n_angle;
    const size_t effective_lane_chunk = alignedLaneChunk(
        static_cast<size_t>(std::max(1, phys.streaming_lane_chunk)),
        n_angle);
    const bool can_reuse_secondary_source = effective_lane_chunk >= n_per_E;
    const bool cache_secondary_source = !can_reuse_secondary_source &&
                                        !stream_source_path.empty();

    std::cout << "Starting BFPn solver..." << std::endl;
    std::cout << "Grid: " << grid.Nx << "x" << grid.Ny << "x" << grid.Nz
              << ", Energy: " << grid.Ng << ", Angle: " << grid.Nmu << "x" << grid.Nom << std::endl;
    std::cout << "Mode: streaming-full out-of-core"
              << ", lane chunk " << effective_lane_chunk
              << " (requested " << std::max(1, phys.streaming_lane_chunk) << ")"
              << ", energy chunk " << std::max(1, phys.streaming_energy_chunk)
              << ", IDD stride " << idd_stride
              << ", secondary source cache " << (cache_secondary_source ? "on" : "off")
              << std::endl;

    for (int step = 0; step < max_steps; ++step) {
        bool compute_idd = ((step + 1) % idd_stride == 0) || (step + 1 == max_steps);
        double idd = 0.0;

        streamingEnergyStep(stream_F_path, stream_f_F_path, nullptr, nullptr, false, half_dt);
        streamingTransportPairStep(stream_F_path, stream_f_F_path, half_dt);
        streamingAnglePairStep(stream_F_path, stream_f_F_path, grid.dt);
        streamingTransportPairStep(stream_F_path, stream_f_F_path, half_dt);
        streamingEnergyStep(stream_F_path, stream_f_F_path, nullptr, nullptr, false, half_dt,
                            compute_idd ? &idd : nullptr);

        if (!phys.primary_only && step > 0) {
            streamingEnergyStep(stream_F1_path, stream_f_F1_path,
                                &stream_F_path, &stream_f_F_path,
                                true, half_dt,
                                nullptr,
                                false,
                                cache_secondary_source ? &stream_source_path : nullptr,
                                false,
                                cache_secondary_source);
            streamingTransportPairStep(stream_F1_path, stream_f_F1_path, half_dt);
            streamingAnglePairStep(stream_F1_path, stream_f_F1_path, grid.dt);
            streamingTransportPairStep(stream_F1_path, stream_f_F1_path, half_dt);
            streamingEnergyStep(stream_F1_path, stream_f_F1_path,
                                &stream_F_path, &stream_f_F_path,
                                true, half_dt,
                                compute_idd ? &idd : nullptr,
                                can_reuse_secondary_source,
                                cache_secondary_source ? &stream_source_path : nullptr,
                                cache_secondary_source,
                                false);
        }

        double t = (step + 1) * grid.dt;
        if (compute_idd) {
            idd_history.emplace_back(t, idd);
            proxy_history.emplace_back(t, 0.0);
        }

        for (size_t i = 0; i < phys.spot_depths.size(); ++i) {
            if (!spot_saved[i] && t + 0.5 * grid.dt >= phys.spot_depths[i]) {
                saveSpotPlaneStreaming(phys.spot_depths[i], t);
                spot_saved[i] = 1;
            }
        }

        if ((step + 1) % 100 == 0) {
            std::cout << "Step " << (step + 1) << ", t = " << t << "/" << tFinal;
            if (compute_idd) {
                std::cout << ", IDD = " << idd;
            }
            std::cout << std::endl;
        }
    }

    saveDepthResults(idd_history, "idd_output.txt");
    saveDepthResults(proxy_history, "dose_output.txt");
    std::cout << "Solver completed. Paper-style IDD saved to idd_output.txt" << std::endl;
    std::cout << "Diagnostic scalar proxy saved to dose_output.txt" << std::endl;
}

void BFPnSolver::stepPrimary(int ping, double t) {
    const double half_dt = 0.5 * grid.dt;

    energy_solver->solve(d_F.data(), d_f_F.data(),
                        d_F.data(), d_f_F.data(),
                        d_S_s.data(), d_sigma_c.data(), d_T_c.data(),
                        half_dt, grid.dg, phys.energy_density,
                        (grid.Ny+1)*(grid.Nz+1), grid.Nmu*grid.Nom,
                        false, nullptr, nullptr,
                        phys.legacy_energy, phys.eq15_straggling, stream_primary[0]);
    CUDA_CHECK(cudaStreamSynchronize(stream_primary[0]));

    if (!phys.no_transport) {
        transport_solver->solveEq18(d_F.data(), d_Omend.data(),
                                    grid.Ng, grid.Nmu*grid.Nom, half_dt,
                                    stream_primary[0], phys.no_spatial_clipping);
        CUDA_CHECK(cudaStreamSynchronize(stream_primary[0]));
        transport_solver->solveEq18(d_f_F.data(), d_Omend.data(),
                                    grid.Ng, grid.Nmu*grid.Nom, half_dt,
                                    stream_primary[0], true);
        CUDA_CHECK(cudaStreamSynchronize(stream_primary[0]));
    }

    if (!phys.no_angle) {
        angle_solver->solveEq19(d_F.data(), d_sig_trg.data(),
                                (grid.Ny+1)*(grid.Nz+1), grid.Ng, grid.dt,
                                phys.density, stream_primary[0]);
        CUDA_CHECK(cudaStreamSynchronize(stream_primary[0]));
        angle_solver->solveEq19(d_f_F.data(), d_sig_trg.data(),
                                (grid.Ny+1)*(grid.Nz+1), grid.Ng, grid.dt,
                                phys.density, stream_primary[0]);
        CUDA_CHECK(cudaStreamSynchronize(stream_primary[0]));
    }

    if (!phys.no_transport) {
        transport_solver->solveEq18(d_F.data(), d_Omend.data(),
                                    grid.Ng, grid.Nmu*grid.Nom, half_dt,
                                    stream_primary[0], phys.no_spatial_clipping);
        CUDA_CHECK(cudaStreamSynchronize(stream_primary[0]));
        transport_solver->solveEq18(d_f_F.data(), d_Omend.data(),
                                    grid.Ng, grid.Nmu*grid.Nom, half_dt,
                                    stream_primary[0], true);
        CUDA_CHECK(cudaStreamSynchronize(stream_primary[0]));
    }

    energy_solver->solve(d_F.data(), d_f_F.data(),
                        d_F.data(), d_f_F.data(),
                        d_S_s.data(), d_sigma_c.data(), d_T_c.data(),
                        half_dt, grid.dg, phys.energy_density,
                        (grid.Ny+1)*(grid.Nz+1), grid.Nmu*grid.Nom,
                        false, nullptr, nullptr,
                        phys.legacy_energy, phys.eq15_straggling, stream_primary[0]);
    CUDA_CHECK(cudaStreamSynchronize(stream_primary[0]));
}

void BFPnSolver::stepPrimaryEnergyOnly() {
    const double half_dt = 0.5 * grid.dt;
    for (int pass = 0; pass < 2; ++pass) {
        energy_solver->solve(d_F.data(), d_f_F.data(),
                            d_F.data(), d_f_F.data(),
                            d_S_s.data(), d_sigma_c.data(), d_T_c.data(),
                            half_dt, grid.dg, phys.energy_density,
                            (grid.Ny+1)*(grid.Nz+1), grid.Nmu*grid.Nom,
                            false, nullptr, nullptr,
                            phys.legacy_energy, phys.eq15_straggling,
                            stream_primary[0]);
        CUDA_CHECK(cudaStreamSynchronize(stream_primary[0]));
    }
}

void BFPnSolver::stepSecondary(int ping, int pong, double t) {
    const double half_dt = 0.5 * grid.dt;

    energy_solver->solveSecondaryFromPrimary(d_F1.data(), d_f_F1.data(),
                        d_F.data(), d_f_F.data(),
                        d_ker_e1.data(), d_ker_e2.data(), d_ker_v.data(),
                        d_S_s.data(), d_sigma_c.data(), d_T_c.data(),
                        half_dt, grid.dg, phys.energy_density,
                        (grid.Ny+1)*(grid.Nz+1), grid.Nmu*grid.Nom,
                        phys.legacy_energy, phys.eq15_straggling,
                        stream_secondary);
    CUDA_CHECK(cudaStreamSynchronize(stream_secondary));

    transport_solver->solveEq18(d_F1.data(), d_Omend.data(),
                                grid.Ng, grid.Nmu*grid.Nom, half_dt, stream_secondary);
    CUDA_CHECK(cudaStreamSynchronize(stream_secondary));
    transport_solver->solveEq18(d_f_F1.data(), d_Omend.data(),
                                grid.Ng, grid.Nmu*grid.Nom, half_dt, stream_secondary, true);
    CUDA_CHECK(cudaStreamSynchronize(stream_secondary));

    angle_solver->solveEq19(d_F1.data(), d_sig_trg.data(),
                            (grid.Ny+1)*(grid.Nz+1), grid.Ng, grid.dt,
                            phys.density, stream_secondary);
    CUDA_CHECK(cudaStreamSynchronize(stream_secondary));
    angle_solver->solveEq19(d_f_F1.data(), d_sig_trg.data(),
                            (grid.Ny+1)*(grid.Nz+1), grid.Ng, grid.dt,
                            phys.density, stream_secondary);
    CUDA_CHECK(cudaStreamSynchronize(stream_secondary));

    transport_solver->solveEq18(d_F1.data(), d_Omend.data(),
                                grid.Ng, grid.Nmu*grid.Nom, half_dt, stream_secondary);
    CUDA_CHECK(cudaStreamSynchronize(stream_secondary));
    transport_solver->solveEq18(d_f_F1.data(), d_Omend.data(),
                                grid.Ng, grid.Nmu*grid.Nom, half_dt, stream_secondary, true);
    CUDA_CHECK(cudaStreamSynchronize(stream_secondary));

    energy_solver->solveSecondaryFromPrimary(d_F1.data(), d_f_F1.data(),
                        d_F.data(), d_f_F.data(),
                        d_ker_e1.data(), d_ker_e2.data(), d_ker_v.data(),
                        d_S_s.data(), d_sigma_c.data(), d_T_c.data(),
                        half_dt, grid.dg, phys.energy_density,
                        (grid.Ny+1)*(grid.Nz+1), grid.Nmu*grid.Nom,
                        phys.legacy_energy, phys.eq15_straggling,
                        stream_secondary);
    CUDA_CHECK(cudaStreamSynchronize(stream_secondary));
}

void BFPnSolver::stepLitePrimary() {
    const int nyz = (grid.Ny + 1) * (grid.Nz + 1);
    const int n_angle = grid.Nmu * grid.Nom;
    const long long n_per_E = static_cast<long long>(nyz) * n_angle;

    constexpr int block_size = 256;
    int grid_size = static_cast<int>((n_per_E + block_size - 1) / block_size);

    litePrimaryEnergyStepKernel<<<grid_size, block_size>>>(
        d_F.data(),
        d_S_s.data(),
        d_sigma_c.data(),
        grid.Ng,
        nyz,
        n_angle,
        grid.dt,
        grid.dg
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

double BFPnSolver::computeScalarDoseProxy() {
    // 简化的剂量计算
    // 实际应该实现完整的积分，这里先基于 f_F 做一个能量沉积分近似
    size_t nyz = (grid.Ny + 1) * (grid.Nz + 1);
    size_t n_angle = grid.Nmu * grid.Nom;
    size_t total = nyz * grid.Ng * n_angle;

    // 这里使用一个非常简化的剂量近似：把当前时间步在所有网格和角度上损失的“通量”累加，
    // 视为与沉积能量成正比，再乘以能群宽度 dg 作为积分权重。
    constexpr int block_size = 256;
    size_t num_blocks = (total + block_size * 2 - 1) / (block_size * 2);
    if (num_blocks > d_reduction_sums.getSize()) {
        throw std::runtime_error("Dose reduction buffer is too small");
    }

    reduceSumKernel<<<static_cast<int>(num_blocks),
                      block_size,
                      block_size * sizeof(double)>>>(
        d_f_F.data(), d_reduction_sums.data(), total
    );
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h_block_sums(num_blocks);
    d_reduction_sums.copyToHost(h_block_sums.data(), num_blocks);

    double dose_step = 0.0;
    for (double partial : h_block_sums) {
        dose_step += partial;
    }
    dose_step *= grid.dg;

    return dose_step;
}

double BFPnSolver::computeEnergyFlux(const double* F, const double* f_F) {
    size_t nyz = (grid.Ny + 1) * (grid.Nz + 1);
    size_t n_angle = grid.Nmu * grid.Nom;
    size_t total = nyz * grid.Ng * n_angle;

    constexpr int block_size = 256;
    size_t num_blocks = (total + block_size * 2 - 1) / (block_size * 2);
    if (num_blocks > d_reduction_sums.getSize()) {
        throw std::runtime_error("Energy flux reduction buffer is too small");
    }

    energyFluxKernel<<<static_cast<int>(num_blocks),
                       block_size,
                       block_size * sizeof(double)>>>(
        F,
        f_F,
        d_en.data(),
        d_reduction_sums.data(),
        grid.Ng, static_cast<int>(nyz), static_cast<int>(n_angle),
        grid.dg, grid.du, grid.dv, grid.dy, grid.dz,
        total
    );
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h_block_sums(num_blocks);
    d_reduction_sums.copyToHost(h_block_sums.data(), num_blocks);
    double flux = 0.0;
    for (double partial : h_block_sums) {
        flux += partial;
    }
    return flux;
}

double BFPnSolver::computeIntegratedDepthDose() {
    size_t nyz = (grid.Ny + 1) * (grid.Nz + 1);
    size_t n_angle = grid.Nmu * grid.Nom;
    size_t total = nyz * grid.Ng * n_angle;

    constexpr int block_size = 256;
    size_t num_blocks = (total + block_size * 2 - 1) / (block_size * 2);
    if (num_blocks > d_reduction_sums.getSize()) {
        throw std::runtime_error("IDD reduction buffer is too small");
    }

    integratedDepthDoseKernel<<<static_cast<int>(num_blocks),
                                block_size,
                                block_size * sizeof(double)>>>(
        d_F.data(), d_f_F.data(),
        d_F1.data(), d_f_F1.data(),
        d_S_s.data(), d_sigma_c.data(), d_en.data(),
        d_reduction_sums.data(),
        grid.Ng, static_cast<int>(nyz), static_cast<int>(n_angle),
        grid.dg, grid.du, grid.dv, grid.dy, grid.dz,
        phys.density,
        total
    );
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h_block_sums(num_blocks);
    d_reduction_sums.copyToHost(h_block_sums.data(), num_blocks);

    double idd = 0.0;
    for (double partial : h_block_sums) {
        idd += partial;
    }
    return idd;
}

std::vector<double> BFPnSolver::computeEnergyMoments() {
    int nyz = (grid.Ny + 1) * (grid.Nz + 1);
    int n_angle = grid.Nmu * grid.Nom;
    if (d_energy_moments.getSize() < static_cast<size_t>(grid.Ng) * 2) {
        d_energy_moments.allocate(static_cast<size_t>(grid.Ng) * 2);
    }

    constexpr int block_size = 256;
    energyMomentKernel<<<grid.Ng,
                         block_size,
                         2 * block_size * sizeof(double)>>>(
        d_F.data(), d_f_F.data(),
        d_F1.data(), d_f_F1.data(),
        d_energy_moments.data(),
        grid.Ng, nyz, n_angle,
        grid.du, grid.dv, grid.dy, grid.dz
    );
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> moments(static_cast<size_t>(grid.Ng) * 2);
    d_energy_moments.copyToHost(moments.data(), moments.size());
    return moments;
}

double BFPnSolver::computeIntegratedDepthDoseLite() {
    size_t nyz = (grid.Ny + 1) * (grid.Nz + 1);
    size_t n_angle = grid.Nmu * grid.Nom;
    size_t total = nyz * grid.Ng * n_angle;

    constexpr int block_size = 256;
    size_t num_blocks = (total + block_size * 2 - 1) / (block_size * 2);
    if (num_blocks > d_reduction_sums.getSize()) {
        throw std::runtime_error("IDD reduction buffer is too small");
    }

    integratedDepthDoseLiteKernel<<<static_cast<int>(num_blocks),
                                    block_size,
                                    block_size * sizeof(double)>>>(
        d_F.data(),
        d_S_s.data(),
        d_reduction_sums.data(),
        grid.Ng, static_cast<int>(nyz), static_cast<int>(n_angle),
        grid.dg, grid.du, grid.dv, grid.dy, grid.dz,
        phys.density,
        total
    );
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h_block_sums(num_blocks);
    d_reduction_sums.copyToHost(h_block_sums.data(), num_blocks);

    double idd = 0.0;
    for (double partial : h_block_sums) {
        idd += partial;
    }
    return idd;
}

std::vector<double> BFPnSolver::computeSpotDosePlane() {
    int nyz = (grid.Ny + 1) * (grid.Nz + 1);
    int n_angle = grid.Nmu * grid.Nom;
    if (d_spot_plane.getSize() < static_cast<size_t>(nyz)) {
        d_spot_plane.allocate(nyz);
    }

    constexpr int block_size = 256;
    int grid_size = (nyz + block_size - 1) / block_size;
    spotDosePlaneKernel<<<grid_size, block_size>>>(
        d_F.data(), d_f_F.data(),
        d_F1.data(), d_f_F1.data(),
        d_S_s.data(),
        d_spot_plane.data(),
        grid.Ng, nyz, n_angle,
        grid.dg, grid.du, grid.dv,
        phys.density
    );
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h_plane(nyz);
    d_spot_plane.copyToHost(h_plane.data(), h_plane.size());
    return h_plane;
}

void BFPnSolver::saveSpotPlane(double requested_depth, double actual_depth) {
    if (phys.spot_prefix.empty()) return;

    std::vector<double> plane = computeSpotDosePlane();
    std::ostringstream name;
    name << phys.spot_prefix
         << "_req" << std::fixed << std::setprecision(2) << requested_depth
         << "_x" << std::fixed << std::setprecision(2) << actual_depth
         << ".txt";

    std::ofstream out(name.str());
    out << "# requested_x_cm " << requested_depth << "\n";
    out << "# actual_x_cm " << actual_depth << "\n";
    out << "# y_cm z_cm dose\n";
    int Ny1 = grid.Ny + 1;
    int Nz1 = grid.Nz + 1;
    for (int j = 0; j < Nz1; ++j) {
        double z = -0.5 * grid.Lz + j * grid.dz;
        for (int i = 0; i < Ny1; ++i) {
            double y = -0.5 * grid.Ly + i * grid.dy;
            out << y << " " << z << " " << plane[j * Ny1 + i] << "\n";
        }
        out << "\n";
    }
    std::cout << "Spot dose plane saved to " << name.str() << std::endl;
}

void BFPnSolver::saveSpotPlaneStreaming(double requested_depth, double actual_depth) {
    if (phys.spot_prefix.empty()) return;

    const size_t nyz = static_cast<size_t>(grid.Ny + 1) * (grid.Nz + 1);
    const size_t n_angle = static_cast<size_t>(grid.Nmu) * grid.Nom;
    const size_t per_energy = nyz * n_angle;
    const int energy_chunk = std::max(1, phys.streaming_energy_chunk);
    constexpr int block_size = 256;
    std::vector<double> h_total_plane(nyz, 0.0);
    std::vector<double> h_chunk;
    std::vector<double> h_plane(nyz);

    if (d_spot_plane.getSize() < nyz) {
        d_spot_plane.allocate(nyz);
    }

    for (int e0 = 0; e0 < grid.Ng; e0 += energy_chunk) {
        int ecount = std::min(energy_chunk, grid.Ng - e0);
        size_t chunk_size = static_cast<size_t>(ecount) * per_energy;
        size_t offset = static_cast<size_t>(e0) * per_energy;
        h_chunk.resize(chunk_size);

        readStore(stream_F_path, offset, h_chunk.data(), chunk_size);
        CUDA_CHECK(cudaMemcpy(d_stream_F.data(), h_chunk.data(),
                              chunk_size * sizeof(double), cudaMemcpyHostToDevice));
        readStore(stream_f_F_path, offset, h_chunk.data(), chunk_size);
        CUDA_CHECK(cudaMemcpy(d_stream_f_F.data(), h_chunk.data(),
                              chunk_size * sizeof(double), cudaMemcpyHostToDevice));

        if (phys.primary_only) {
            CUDA_CHECK(cudaMemset(d_stream_F1.data(), 0, chunk_size * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_stream_f_F1.data(), 0, chunk_size * sizeof(double)));
        } else {
            readStore(stream_F1_path, offset, h_chunk.data(), chunk_size);
            CUDA_CHECK(cudaMemcpy(d_stream_F1.data(), h_chunk.data(),
                                  chunk_size * sizeof(double), cudaMemcpyHostToDevice));
            readStore(stream_f_F1_path, offset, h_chunk.data(), chunk_size);
            CUDA_CHECK(cudaMemcpy(d_stream_f_F1.data(), h_chunk.data(),
                                  chunk_size * sizeof(double), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaMemset(d_spot_plane.data(), 0, nyz * sizeof(double)));
        int grid_size = static_cast<int>((nyz + block_size - 1) / block_size);
        spotDosePlaneKernel<<<grid_size, block_size>>>(
            d_stream_F.data(), d_stream_f_F.data(),
            d_stream_F1.data(), d_stream_f_F1.data(),
            d_S_s.data() + e0,
            d_spot_plane.data(),
            ecount, static_cast<int>(nyz), static_cast<int>(n_angle),
            grid.dg, grid.du, grid.dv,
            phys.density
        );
        CUDA_CHECK(cudaGetLastError());
        d_spot_plane.copyToHost(h_plane.data(), h_plane.size());
        for (size_t i = 0; i < nyz; ++i) {
            h_total_plane[i] += h_plane[i];
        }
    }

    std::ostringstream name;
    name << phys.spot_prefix
         << "_req" << std::fixed << std::setprecision(2) << requested_depth
         << "_x" << std::fixed << std::setprecision(2) << actual_depth
         << ".txt";
    std::ofstream out(name.str());
    out << "# requested_x_cm " << requested_depth << "\n";
    out << "# actual_x_cm " << actual_depth << "\n";
    out << "# y_cm z_cm dose\n";
    int Ny1 = grid.Ny + 1;
    int Nz1 = grid.Nz + 1;
    for (int j = 0; j < Nz1; ++j) {
        double z = -0.5 * grid.Lz + j * grid.dz;
        for (int i = 0; i < Ny1; ++i) {
            double y = -0.5 * grid.Ly + i * grid.dy;
            out << y << " " << z << " " << h_total_plane[static_cast<size_t>(j) * Ny1 + i] << "\n";
        }
        out << "\n";
    }
    std::cout << "Spot dose plane saved to " << name.str() << std::endl;
}

double BFPnSolver::computeEnergyFluxStreaming(const std::string& F_path,
                                              const std::string& f_F_path) {
    const size_t nyz = static_cast<size_t>(grid.Ny + 1) * (grid.Nz + 1);
    const size_t n_angle = static_cast<size_t>(grid.Nmu) * grid.Nom;
    const size_t per_energy = nyz * n_angle;
    const int energy_chunk = std::max(1, phys.streaming_energy_chunk);
    constexpr int block_size = 256;

    double flux = 0.0;
    for (int e0 = 0; e0 < grid.Ng; e0 += energy_chunk) {
        int ecount = std::min(energy_chunk, grid.Ng - e0);
        size_t chunk_size = static_cast<size_t>(ecount) * per_energy;
        size_t offset = static_cast<size_t>(e0) * per_energy;
        std::vector<double> h_chunk(chunk_size);

        readStore(F_path, offset, h_chunk.data(), chunk_size);
        CUDA_CHECK(cudaMemcpy(d_stream_F.data(), h_chunk.data(),
                              chunk_size * sizeof(double), cudaMemcpyHostToDevice));
        readStore(f_F_path, offset, h_chunk.data(), chunk_size);
        CUDA_CHECK(cudaMemcpy(d_stream_f_F.data(), h_chunk.data(),
                              chunk_size * sizeof(double), cudaMemcpyHostToDevice));

        size_t num_blocks = (chunk_size + block_size * 2 - 1) / (block_size * 2);
        if (num_blocks > d_reduction_sums.getSize()) {
            throw std::runtime_error("Streaming energy flux reduction buffer is too small");
        }
        energyFluxKernel<<<static_cast<int>(num_blocks),
                           block_size,
                           block_size * sizeof(double)>>>(
            d_stream_F.data(), d_stream_f_F.data(),
            d_en.data() + e0,
            d_reduction_sums.data(),
            ecount, static_cast<int>(nyz), static_cast<int>(n_angle),
            grid.dg, grid.du, grid.dv, grid.dy, grid.dz,
            chunk_size
        );
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> h_block_sums(num_blocks);
        d_reduction_sums.copyToHost(h_block_sums.data(), num_blocks);
        for (double partial : h_block_sums) {
            flux += partial;
        }
    }
    return flux;
}

double BFPnSolver::computeIntegratedDepthDoseStreaming() {
    const size_t nyz = static_cast<size_t>(grid.Ny + 1) * (grid.Nz + 1);
    const size_t n_angle = static_cast<size_t>(grid.Nmu) * grid.Nom;
    const size_t per_energy = nyz * n_angle;
    const int energy_chunk = std::max(1, phys.streaming_energy_chunk);
    constexpr int block_size = 256;

    double idd = 0.0;
    for (int e0 = 0; e0 < grid.Ng; e0 += energy_chunk) {
        int ecount = std::min(energy_chunk, grid.Ng - e0);
        size_t chunk_size = static_cast<size_t>(ecount) * per_energy;
        size_t offset = static_cast<size_t>(e0) * per_energy;
        std::vector<double> h_chunk(chunk_size);

        readStore(stream_F_path, offset, h_chunk.data(), chunk_size);
        CUDA_CHECK(cudaMemcpy(d_stream_F.data(),
                              h_chunk.data(),
                              chunk_size * sizeof(double),
                              cudaMemcpyHostToDevice));
        readStore(stream_f_F_path, offset, h_chunk.data(), chunk_size);
        CUDA_CHECK(cudaMemcpy(d_stream_f_F.data(),
                              h_chunk.data(),
                              chunk_size * sizeof(double),
                              cudaMemcpyHostToDevice));
        if (phys.primary_only) {
            CUDA_CHECK(cudaMemset(d_stream_F1.data(), 0, chunk_size * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_stream_f_F1.data(), 0, chunk_size * sizeof(double)));
        } else {
            readStore(stream_F1_path, offset, h_chunk.data(), chunk_size);
            CUDA_CHECK(cudaMemcpy(d_stream_F1.data(),
                                  h_chunk.data(),
                                  chunk_size * sizeof(double),
                                  cudaMemcpyHostToDevice));
            readStore(stream_f_F1_path, offset, h_chunk.data(), chunk_size);
            CUDA_CHECK(cudaMemcpy(d_stream_f_F1.data(),
                                  h_chunk.data(),
                                  chunk_size * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }

        size_t num_blocks = (chunk_size + block_size * 2 - 1) / (block_size * 2);
        if (num_blocks > d_reduction_sums.getSize()) {
            throw std::runtime_error("Streaming IDD reduction buffer is too small");
        }

        integratedDepthDoseKernel<<<static_cast<int>(num_blocks),
                                    block_size,
                                    block_size * sizeof(double)>>>(
            d_stream_F.data(), d_stream_f_F.data(),
            d_stream_F1.data(), d_stream_f_F1.data(),
            d_S_s.data() + e0, d_sigma_c.data() + e0, d_en.data() + e0,
            d_reduction_sums.data(),
            ecount, static_cast<int>(nyz), static_cast<int>(n_angle),
            grid.dg, grid.du, grid.dv, grid.dy, grid.dz,
            phys.density,
            chunk_size
        );
        CUDA_CHECK(cudaGetLastError());

        std::vector<double> h_block_sums(num_blocks);
        d_reduction_sums.copyToHost(h_block_sums.data(), num_blocks);
        for (double partial : h_block_sums) {
            idd += partial;
        }
    }

    return idd;
}

void BFPnSolver::saveResults(const std::vector<double>& values, const std::string& filename) {
    std::ofstream out(filename);
    out << "# x_cm value\n";
    for (size_t i = 0; i < values.size(); i++) {
        out << (i + 1) * grid.dt << " " << values[i] << "\n";
    }
}

void BFPnSolver::saveEnergyMoments(const std::vector<double>& values,
                                   const std::string& filename) {
    std::ofstream out(filename);
    out << "# step x_cm g psi1_integral psi2_integral\n";
    const size_t per_step = static_cast<size_t>(grid.Ng) * 2;
    if (per_step == 0) return;
    const size_t steps = values.size() / per_step;
    for (size_t step = 0; step < steps; ++step) {
        double x = (step + 1) * grid.dt;
        size_t base = step * per_step;
        for (int g = 0; g < grid.Ng; ++g) {
            out << (step + 1) << " "
                << x << " "
                << g << " "
                << values[base + static_cast<size_t>(2 * g)] << " "
                << values[base + static_cast<size_t>(2 * g + 1)] << "\n";
        }
    }
}

void BFPnSolver::saveDepthResults(const std::vector<std::pair<double, double>>& values,
                                  const std::string& filename) {
    std::ofstream out(filename);
    out << "# x_cm value\n";
    for (const auto& item : values) {
        out << item.first << " " << item.second << "\n";
    }
}
