#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

struct Grid {
    int Nx = 100;
    int Ny = 10;
    int Nz = 10;
    int Ng = 500;
    int Nmu = 11;
    int Nom = 11;
    double Lx = 40.0;
    double Ly = 1.0;
    double Lz = 1.0;
    double Lu = 1.0;
    double Lv = 1.0;
    double Lg = 259.0;
    double dt = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double dg = 0.0;
    double du = 0.0;
    double dv = 0.0;

    void computeDeltas() {
        dt = Lx / Nx;
        dy = Ly / Ny;
        dz = Lz / Nz;
        dg = Lg / Ng;
        du = Lu / (Nmu - 1);
        dv = Lv / (Nom - 1);
    }
};

struct Args {
    int nx = 100;
    double t_final = 0.4;
    std::string data_path = "data";
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            args.data_path = argv[++i];
        } else if (arg == "--time" && i + 1 < argc) {
            args.t_final = std::stod(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [Nx] [--data path] [--time value]\n";
            std::exit(0);
        } else if (!arg.empty() && arg[0] != '-') {
            args.nx = std::stoi(arg);
        }
    }
    return args;
}

std::vector<double> loadColumnFile(const std::string& path, int rows, int cols) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open " + path);
    }

    std::vector<double> values(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!(in >> values[static_cast<size_t>(r) * cols + c])) {
                throw std::runtime_error("not enough numeric data in " + path);
            }
        }
    }
    return values;
}

size_t indexF(const Grid& g, int ek, int pos, int ang) {
    const int nyz = (g.Ny + 1) * (g.Nz + 1);
    const int n_angle = g.Nmu * g.Nom;
    return (static_cast<size_t>(ek) * nyz + pos) * n_angle + ang;
}

void dft2(std::vector<std::complex<double>>& data, int rows, int cols, bool inverse) {
    std::vector<std::complex<double>> out(data.size());
    const double sign = inverse ? 1.0 : -1.0;
    for (int p = 0; p < rows; ++p) {
        for (int q = 0; q < cols; ++q) {
            std::complex<double> sum = 0.0;
            for (int m = 0; m < rows; ++m) {
                for (int n = 0; n < cols; ++n) {
                    const double angle = sign * 2.0 * PI *
                        (static_cast<double>(p * m) / rows + static_cast<double>(q * n) / cols);
                    sum += data[static_cast<size_t>(m) * cols + n] *
                           std::complex<double>(std::cos(angle), std::sin(angle));
                }
            }
            if (inverse) {
                sum /= static_cast<double>(rows * cols);
            }
            out[static_cast<size_t>(p) * cols + q] = sum;
        }
    }
    data.swap(out);
}

void angleDiffusion(Grid const& g, std::vector<double>& F, const std::vector<double>& sig_trg) {
    const int nyz = (g.Ny + 1) * (g.Nz + 1);
    const int n_angle = g.Nmu * g.Nom;
    std::vector<std::complex<double>> slice(n_angle);

    for (int ek = 0; ek < g.Ng; ++ek) {
        const double D = sig_trg[ek] / (4.0 * g.du * g.du);
        for (int pos = 0; pos < nyz; ++pos) {
            for (int a = 0; a < n_angle; ++a) {
                slice[a] = F[indexF(g, ek, pos, a)];
            }

            dft2(slice, g.Nmu, g.Nom, false);

            for (int idx = 0; idx < n_angle; ++idx) {
                const int i = idx % g.Nom;
                const int j = idx / g.Nom;
                double ku = (i < g.Nom / 2) ? i : i - g.Nom;
                double kv = (j < g.Nmu / 2) ? j : j - g.Nmu;
                ku *= 2.0 * PI / g.Nom;
                kv *= 2.0 * PI / g.Nmu;
                const double k2 = ku * ku + kv * kv;
                const double factor = (2.0 / g.dt + D * k2) / (2.0 / g.dt - D * k2);
                slice[idx] *= factor;
            }

            dft2(slice, g.Nmu, g.Nom, true);

            for (int a = 0; a < n_angle; ++a) {
                F[indexF(g, ek, pos, a)] = slice[a].real();
            }
        }
    }
}

void spatialTransport(Grid const& g, std::vector<double>& F, const std::vector<double>& Omend) {
    const int Ny1 = g.Ny + 1;
    const int Nz1 = g.Nz + 1;
    const int nyz = Ny1 * Nz1;
    const int n_angle = g.Nmu * g.Nom;
    std::vector<double> out(F.size());

    for (int ek = 0; ek < g.Ng; ++ek) {
        for (int j = 0; j < Nz1; ++j) {
            for (int i = 0; i < Ny1; ++i) {
                const int s = j * Ny1 + i;
                for (int ang = 0; ang < n_angle; ++ang) {
                    const double omega_y = Omend[2 * ang + 0];
                    const double omega_z = Omend[2 * ang + 1];
                    const size_t base = indexF(g, ek, s, ang);
                    const double center = F[base];

                    const int s_im1 = (i > 0) ? j * Ny1 + (i - 1) : s;
                    const int s_ip1 = (i < Ny1 - 1) ? j * Ny1 + (i + 1) : s;
                    const int s_jm1 = (j > 0) ? (j - 1) * Ny1 + i : s;
                    const int s_jp1 = (j < Nz1 - 1) ? (j + 1) * Ny1 + i : s;

                    double dFdy = 0.0;
                    if (omega_y > 0.0) {
                        dFdy = (center - F[indexF(g, ek, s_im1, ang)]) / g.dy;
                    } else if (omega_y < 0.0) {
                        dFdy = (F[indexF(g, ek, s_ip1, ang)] - center) / g.dy;
                    }

                    double dFdz = 0.0;
                    if (omega_z > 0.0) {
                        dFdz = (center - F[indexF(g, ek, s_jm1, ang)]) / g.dz;
                    } else if (omega_z < 0.0) {
                        dFdz = (F[indexF(g, ek, s_jp1, ang)] - center) / g.dz;
                    }

                    out[base] = center - g.dt * (omega_y * dFdy + omega_z * dFdz);
                }
            }
        }
    }

    F.swap(out);
}

void energyAttenuation(Grid const& g, std::vector<double>& F, std::vector<double>& f_F,
                       const std::vector<double>& sigma_c) {
    const int nyz = (g.Ny + 1) * (g.Nz + 1);
    const int n_angle = g.Nmu * g.Nom;
    for (int ek = 0; ek < g.Ng; ++ek) {
        const double decay = std::exp(-sigma_c[ek] * g.dt);
        for (int pos = 0; pos < nyz; ++pos) {
            for (int ang = 0; ang < n_angle; ++ang) {
                const size_t idx = indexF(g, ek, pos, ang);
                const double old = F[idx];
                const double next = old * decay;
                F[idx] = next;
                f_F[idx] = old - next;
            }
        }
    }
}

double computeDose(Grid const& g, const std::vector<double>& f_F) {
    double sum = 0.0;
    for (double value : f_F) {
        sum += value;
    }
    return sum * g.dg;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Args args = parseArgs(argc, argv);

        Grid g;
        g.Nx = args.nx;
        g.computeDeltas();

        std::vector<double> mu = loadColumnFile(args.data_path + "/data_cross.txt", g.Ng, 3);
        std::vector<double> sigma_c = loadColumnFile(args.data_path + "/cross_total.txt", g.Ng, 1);
        for (double& s : sigma_c) {
            s *= 0.9;
        }
        for (int i = 401; i < 444 && i < g.Ng; ++i) {
            sigma_c[i] -= 0.0002 * (0.5 + 0.5 * (i - 401));
        }

        std::vector<double> u(g.Nmu), v(g.Nom);
        for (int i = 0; i < g.Nmu; ++i) {
            u[i] = -0.5 + i * g.du;
        }
        for (int i = 0; i < g.Nom; ++i) {
            v[i] = -0.5 + i * g.dv;
        }

        const int n_angle = g.Nmu * g.Nom;
        const int nyz = (g.Ny + 1) * (g.Nz + 1);
        std::vector<double> Omend(static_cast<size_t>(n_angle) * 2);
        for (int j = 0; j < g.Nmu; ++j) {
            for (int i = 0; i < g.Nom; ++i) {
                const int idx = j * g.Nom + i;
                Omend[2 * idx + 0] = -v[g.Nmu - 1 - j];
                Omend[2 * idx + 1] = -u[i];
            }
        }

        std::vector<double> sig_trg(g.Ng);
        for (int ek = 0; ek < g.Ng; ++ek) {
            const double E = (ek + 0.5) * g.dg + 1.0;
            const double beta = std::sqrt(2.0 * E * 1.6e-19 / 1.673e-27) / (3.0 * 2.998e8);
            const double eta = std::pow(3.0, 2.0 / 3.0) * std::pow(1.0 / 137.0, 2) *
                               std::pow(9.10956e-31 / 1.673e-27, 2) / (beta * beta);
            const double mid = std::log((eta + 1.0) / eta) - 1.0 / (eta + 1.0);
            sig_trg[ek] = 2.0 * PI * 3.34 * 197.3 * 197.3 /
                          std::pow(1.0 / 137.0, 2) / 4.0 * 9.0 / (E * E) * mid / 1000.0;
        }

        std::vector<double> F(static_cast<size_t>(nyz) * g.Ng * n_angle, 0.0);
        std::vector<double> f_F(F.size(), 0.0);

        constexpr double a_1 = 1.0 / 2.0 / 0.1 / 0.1;
        constexpr double a_2 = 1.0 / 2.0 / 1e-6 / 1e-6;
        constexpr double sig_E = 1.0;
        for (int ang = 0; ang < n_angle; ++ang) {
            const double omega1 = 0.0;
            const double omega2 = 0.0;
            for (int j = 0; j <= g.Nz; ++j) {
                for (int i = 0; i <= g.Ny; ++i) {
                    const double y_pos = i * g.dy - 1.0;
                    const double z_pos = j * g.dz - 1.0;
                    const double f1 = std::exp(-(a_1 * y_pos * y_pos + a_2 * omega1 * omega1)) *
                                      std::exp(-(a_1 * z_pos * z_pos + a_2 * omega2 * omega2));
                    const int pos = j * (g.Ny + 1) + i;
                    for (int ek = 0; ek < g.Ng; ++ek) {
                        const double E = (ek + 0.5) * g.dg + 1.0;
                        const double f2 = 1.0 / std::sqrt(2.0 * PI) / sig_E *
                                          std::exp(-std::pow((E - 230.0) / sig_E / std::sqrt(2.0), 2));
                        F[indexF(g, ek, pos, ang)] = f1 * f2;
                    }
                }
            }
        }

        int steps = 0;
        double t = 0.0;
        double last_dose = 0.0;
        auto start = std::chrono::steady_clock::now();
        while (t < args.t_final) {
            angleDiffusion(g, F, sig_trg);
            spatialTransport(g, F, Omend);
            energyAttenuation(g, F, f_F, sigma_c);
            last_dose = computeDose(g, f_F);
            t += g.dt;
            ++steps;
        }
        auto end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();

        std::cout << "CPU serial BFPn prototype\n";
        std::cout << "Grid: " << g.Nx << "x" << g.Ny << "x" << g.Nz
                  << ", Energy: " << g.Ng << ", Angle: " << g.Nmu << "x" << g.Nom << "\n";
        std::cout << "Steps: " << steps << "\n";
        std::cout << "Last dose: " << last_dose << "\n";
        std::cout << "Elapsed seconds: " << elapsed << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
