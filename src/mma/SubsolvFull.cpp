////////////////////////////////////////////////////////////////////////////////
// Full primal-dual MMA subproblem solver -- see SubsolvFull.h for the contract.
//
// Every block below cites the `subsolv.m` lines it ports. The port keeps the
// reference's variable names and its algorithmic order:
//
//   1. primal-dual initialization                       (subsolv.m 42-58)
//   2. barrier parameter continuation                   (subsolv.m 59, 219)
//   3. complete residual assembly                       (subsolv.m 60-85, 186-207)
//   4. Newton system construction                       (subsolv.m 90-115)
//   5. Newton direction                                 (subsolv.m 116-148)
//   6. maximum feasible step                            (subsolv.m 152-161)
//   7. trial point                                      (subsolv.m 177-185)
//   8. residual evaluation at the trial                 (subsolv.m 186-208)
//   9. residual-decrease backtracking                   (subsolv.m 173-210)
//  10. next Newton iteration / next barrier level       (subsolv.m 211-219)
//
// The only representation-driven deviations are called out inline with
// `DEVIATION:` and collected in the m9 report.
////////////////////////////////////////////////////////////////////////////////

#include "SubsolvFull.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mma {
namespace {

/**
 * DEVIATION (documented, not arithmetic): the reference implements the Newton
 * system in two forms and picks by problem shape --
 *
 *   m < n   an (m+1)x(m+1) system          (subsolv.m 116-125)
 *   m >= n  an (n+1)x(n+1) system          (subsolv.m 126-141)
 *
 * For a topology-optimization subproblem m is the number of constraints (1-3)
 * and n is the design dimension (1e6+), so the first branch is the one that
 * runs. The second branch is still implemented -- the n = 1 and n = 2
 * equivalence fixtures take it -- but it needs a dense (n+1)x(n+1) factorisation
 * and is therefore refused above this limit instead of silently allocating
 * something enormous. No production configuration reaches it.
 */
constexpr int kDenseBranchLimit = 512;

/// Dense LU with partial pivoting, row-major k x k. Mirrors MATLAB's `A\b`
/// for the square systems the reference builds. Returns false if singular.
bool SolveDense(std::vector<double>& A, std::vector<double>& rhs, int k) {
    for (int col = 0; col < k; ++col) {
        int piv = col;
        double best = std::abs(A[col * k + col]);
        for (int r = col + 1; r < k; ++r) {
            const double v = std::abs(A[r * k + col]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (!(best > 0.0)) return false;
        if (piv != col) {
            for (int c = 0; c < k; ++c) {
                std::swap(A[col * k + c], A[piv * k + c]);
            }
            std::swap(rhs[col], rhs[piv]);
        }
        const double d = A[col * k + col];
        for (int r = col + 1; r < k; ++r) {
            const double f = A[r * k + col] / d;
            A[r * k + col] = f;
            for (int c = col + 1; c < k; ++c) {
                A[r * k + c] -= f * A[col * k + c];
            }
        }
    }
    for (int r = 1; r < k; ++r) {
        double acc = rhs[r];
        for (int c = 0; c < r; ++c) acc -= A[r * k + c] * rhs[c];
        rhs[r] = acc;
    }
    rhs[k - 1] /= A[(k - 1) * k + (k - 1)];
    for (int r = k - 2; r >= 0; --r) {
        double acc = rhs[r];
        for (int c = r + 1; c < k; ++c) acc -= A[r * k + c] * rhs[c];
        rhs[r] = acc / A[r * k + r];
    }
    return true;
}

double MaxAbs(const std::vector<double>& v) {
    double r = 0.0;
    for (double x : v) r = std::max(r, std::abs(x));
    return r;
}

void EmitTrace(const SubsolvSolveOptions& options, SubsolvTraceRecord record) {
    if (options.trace) options.trace(record);
}

bool TraceDomainOk(const SubsolvProblem& p, const SubsolvResult& q) {
    if (!std::isfinite(q.z) || !std::isfinite(q.zet) || q.z < 0.0 || q.zet < 0.0) {
        return false;
    }
    for (int j = 0; j < p.n; ++j) {
        if (!std::isfinite(q.x[j]) || !std::isfinite(q.xsi[j]) ||
            !std::isfinite(q.eta[j]) || q.x[j] < p.alfa[j] || q.x[j] > p.beta[j] ||
            q.xsi[j] < 0.0 || q.eta[j] < 0.0) return false;
    }
    for (int i = 0; i < p.m; ++i) {
        if (!std::isfinite(q.y[i]) || !std::isfinite(q.lam[i]) ||
            !std::isfinite(q.mu[i]) || !std::isfinite(q.s[i]) || q.y[i] < 0.0 ||
            q.lam[i] < 0.0 || q.mu[i] < 0.0 || q.s[i] < 0.0) return false;
    }
    return true;
}

double TraceStepNorm(const std::vector<double>& dx, const std::vector<double>& dy,
                     double dz, const std::vector<double>& dlam,
                     const std::vector<double>& dxsi, const std::vector<double>& deta,
                     const std::vector<double>& dmu, double dzet,
                     const std::vector<double>& ds, double scale) {
    double sum = dz * dz + dzet * dzet;
    for (const auto* v : {&dx, &dy, &dlam, &dxsi, &deta, &dmu, &ds}) {
        for (double x : *v) sum += x * x;
    }
    return std::abs(scale) * std::sqrt(sum);
}

} // namespace

double SubsolvResidual::worst_component() const {
    return std::max({rex, rey, rez, relam, rexsi, reeta, remu, rezet, res});
}

SubsolvResidual EvaluateSubsolvResidual(const SubsolvProblem& p,
                                        const SubsolvResult& q, double epsi) {
    const int n = p.n, m = p.m;
    SubsolvResidual r;

    // subsolv.m 62-71: derived quantities from the current state.
    std::vector<double> ux1(n), xl1(n), ux2(n), xl2(n), uxinv1(n), xlinv1(n);
    std::vector<double> plam(n), qlam(n), gvec(m), dpsidx(n);
    for (int j = 0; j < n; ++j) {
        ux1[j] = p.upp[j] - q.x[j];
        xl1[j] = q.x[j] - p.low[j];
        ux2[j] = ux1[j] * ux1[j];
        xl2[j] = xl1[j] * xl1[j];
        uxinv1[j] = 1.0 / ux1[j];
        xlinv1[j] = 1.0 / xl1[j];
        double pl = p.p0[j], ql = p.q0[j];
        for (int i = 0; i < m; ++i) {
            pl += p.P[i * n + j] * q.lam[i];
            ql += p.Q[i * n + j] * q.lam[i];
        }
        plam[j] = pl;
        qlam[j] = ql;
        dpsidx[j] = pl / ux2[j] - ql / xl2[j];
    }
    for (int i = 0; i < m; ++i) {
        double g = 0.0;
        for (int j = 0; j < n; ++j) {
            g += p.P[i * n + j] * uxinv1[j] + p.Q[i * n + j] * xlinv1[j];
        }
        gvec[i] = g;
    }

    // subsolv.m 72-83: the nine residual groups.
    std::vector<double> rex(n), rey(m), relam(m), rexsi(n), reeta(n), remu(m),
        res(m);
    double rez = p.a0 - q.zet;
    for (int i = 0; i < m; ++i) rez -= p.a[i] * q.lam[i];
    for (int j = 0; j < n; ++j) {
        rex[j] = dpsidx[j] - q.xsi[j] + q.eta[j];
        rexsi[j] = q.xsi[j] * (q.x[j] - p.alfa[j]) - epsi;
        reeta[j] = q.eta[j] * (p.beta[j] - q.x[j]) - epsi;
    }
    for (int i = 0; i < m; ++i) {
        rey[i] = p.c[i] + p.d[i] * q.y[i] - q.mu[i] - q.lam[i];
        double g = gvec[i] - p.a[i] * q.z - q.y[i] + q.s[i] - p.b[i];
        relam[i] = g;
        remu[i] = q.mu[i] * q.y[i] - epsi;
        res[i] = q.lam[i] * q.s[i] - epsi;
    }
    const double rezet = q.zet * q.z - epsi;

    // subsolv.m 84-85: the two norms.
    double s2 = rez * rez + rezet * rezet;
    for (double v : rex) s2 += v * v;
    for (double v : rey) s2 += v * v;
    for (double v : relam) s2 += v * v;
    for (double v : rexsi) s2 += v * v;
    for (double v : reeta) s2 += v * v;
    for (double v : remu) s2 += v * v;
    for (double v : res) s2 += v * v;

    r.rex = MaxAbs(rex);
    r.rey = MaxAbs(rey);
    r.rez = std::abs(rez);
    r.relam = MaxAbs(relam);
    r.rexsi = MaxAbs(rexsi);
    r.reeta = MaxAbs(reeta);
    r.remu = MaxAbs(remu);
    r.rezet = std::abs(rezet);
    r.res = MaxAbs(res);
    r.rex_values = std::move(rex);
    r.rey_values = std::move(rey);
    r.relam_values = std::move(relam);
    r.rexsi_values = std::move(rexsi);
    r.reeta_values = std::move(reeta);
    r.remu_values = std::move(remu);
    r.res_values = std::move(res);
    r.rez_value = rez;
    r.rezet_value = rezet;
    r.norm2 = std::sqrt(s2);
    r.max_norm = std::max({r.rex, r.rey, r.rez, r.relam, r.rexsi, r.reeta, r.remu,
                           r.rezet, r.res});
    return r;
}

SubsolvResult SolveSubsolvFull(const SubsolvProblem& p) {
    return SolveSubsolvFull(p, SubsolvSolveOptions{});
}

SubsolvResult SolveSubsolvFull(const SubsolvProblem& p,
                              const SubsolvSolveOptions& options) {
    const int n = p.n, m = p.m;
    const double epsimin = p.epsimin;
    const int soft_iteration_cap = SubsolvConstants::InnerIterationCap();
    const int hard_iteration_cap = options.inner_iteration_cap > 0
                                       ? options.inner_iteration_cap
                                       : SubsolvConstants::HardIterationCap();
    SubsolvResult out;
    bool numerical_failure = false;
    bool domain_failure = false;
    bool hard_cap_exhausted = false;

    // ---- 1. primal-dual initialization (subsolv.m 42-58) ----
    std::vector<double> x(n), y(m), lam(m), xsi(n), eta(n), mu(m), s(m);
    double z = 0.0, zet = 0.0;
    for (int j = 0; j < n; ++j) {
        x[j] = 0.5 * (p.alfa[j] + p.beta[j]);
        xsi[j] = std::max(1.0 / (x[j] - p.alfa[j]), 1.0);
        eta[j] = std::max(1.0 / (p.beta[j] - x[j]), 1.0);
    }
    for (int i = 0; i < m; ++i) {
        y[i] = 1.0;
        lam[i] = 1.0;
        mu[i] = std::max(1.0, 0.5 * p.c[i]);
        s[i] = 1.0;
    }
    z = 1.0;
    zet = 1.0;

    double epsi = SubsolvConstants::InitialBarrier();
    double residunorm = 0.0, residumax = 0.0;

    // ---- 2. barrier continuation (subsolv.m 59, 219) ----
    while (epsi > epsimin) {
        const double epsvecn = epsi;
        const double epsvecm = epsi;

        // ---- 3. complete residual assembly (subsolv.m 60-85) ----
        {
            SubsolvResult cur;
            cur.x = x; cur.y = y; cur.z = z; cur.lam = lam;
            cur.xsi = xsi; cur.eta = eta; cur.mu = mu; cur.zet = zet; cur.s = s;
            const SubsolvResidual r = EvaluateSubsolvResidual(p, cur, epsi);
            residunorm = r.norm2;
            residumax = r.max_norm;
        }
        out.barrier_levels += 1;
        out.epsi_final = epsi;
        int ittt = 0;
        bool stage_extended = false;

        {
            SubsolvResult cur;
            cur.x = x; cur.y = y; cur.z = z; cur.lam = lam;
            cur.xsi = xsi; cur.eta = eta; cur.mu = mu; cur.zet = zet; cur.s = s;
            const SubsolvResidual r = EvaluateSubsolvResidual(p, cur, epsi);
            SubsolvTraceRecord record;
            record.event = "barrier_start";
            record.barrier_level = out.barrier_levels;
            record.newton_iteration = out.inner_newton_iterations;
            record.epsi = epsi;
            record.rex = r.rex; record.rey = r.rey; record.rez = r.rez;
            record.relam = r.relam; record.rexsi = r.rexsi; record.reeta = r.reeta;
            record.remu = r.remu; record.rezet = r.rezet; record.res = r.res;
            record.max_norm = r.max_norm; record.norm2 = r.norm2;
            record.domain_ok = TraceDomainOk(p, cur);
            record.all_finite = record.domain_ok;
            EmitTrace(options, record);
        }

        // ---- 4-10. inner Newton loop (subsolv.m 87-214) ----
        while (residumax > SubsolvConstants::InnerResidualFactor() * epsi &&
               ittt < hard_iteration_cap) {
            // The historical first 200 steps are unchanged. If the stage is
            // still not converged, continue from this exact state without
            // reinitializing any primal-dual or Newton quantity.
            if (ittt >= soft_iteration_cap) stage_extended = true;
            ittt += 1;
            out.inner_newton_iterations += 1;

            // ---- 4. Newton system construction (subsolv.m 90-115) ----
            std::vector<double> ux2(n), xl2(n), ux3(n), xl3(n);
            std::vector<double> uxinv1(n), xlinv1(n), uxinv2(n), xlinv2(n);
            std::vector<double> plam(n), qlam(n), dpsidx(n);
            std::vector<double> GG(static_cast<std::size_t>(m) * n);
            for (int j = 0; j < n; ++j) {
                const double u = p.upp[j] - x[j], l = x[j] - p.low[j];
                ux2[j] = u * u; xl2[j] = l * l;
                ux3[j] = ux2[j] * u; xl3[j] = xl2[j] * l;
                uxinv1[j] = 1.0 / u; xlinv1[j] = 1.0 / l;
                uxinv2[j] = 1.0 / ux2[j]; xlinv2[j] = 1.0 / xl2[j];
                double pl = p.p0[j], ql = p.q0[j];
                for (int i = 0; i < m; ++i) {
                    pl += p.P[i * n + j] * lam[i];
                    ql += p.Q[i * n + j] * lam[i];
                }
                plam[j] = pl;
                qlam[j] = ql;
                dpsidx[j] = pl / ux2[j] - ql / xl2[j];
            }
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    // subsolv.m 103: GG = P*diag(uxinv2) - Q*diag(xlinv2)
                    GG[static_cast<std::size_t>(i) * n + j] =
                        p.P[i * n + j] * uxinv2[j] - p.Q[i * n + j] * xlinv2[j];
                }
            }

            std::vector<double> delx(n), diagx(n), dely(m), diagy(m);
            std::vector<double> dellam(m), diaglam(m), diaglamyi(m);
            for (int j = 0; j < n; ++j) {
                // subsolv.m 105
                delx[j] = dpsidx[j] - epsvecn / (x[j] - p.alfa[j]) +
                          epsvecn / (p.beta[j] - x[j]);
                // subsolv.m 109-110
                diagx[j] = 2.0 * (plam[j] / ux3[j] + qlam[j] / xl3[j]) +
                           xsi[j] / (x[j] - p.alfa[j]) +
                           eta[j] / (p.beta[j] - x[j]);
            }
            for (int i = 0; i < m; ++i) {
                dely[i] = p.c[i] + p.d[i] * y[i] - lam[i] - epsvecm / y[i]; // 106
                dellam[i] = 0.0;                                          // 108
                for (int j = 0; j < n; ++j) {
                    dellam[i] += p.P[i * n + j] * uxinv1[j] +
                                 p.Q[i * n + j] * xlinv1[j];
                }
                dellam[i] += -p.a[i] * z - y[i] - p.b[i] + epsvecm / lam[i];
                diagy[i] = p.d[i] + mu[i] / y[i];                     // 112
                diaglam[i] = s[i] / lam[i];                           // 114
                diaglamyi[i] = diaglam[i] + 1.0 / diagy[i];           // 115
            }
            // subsolv.m 107: delz = a0 - a'*lam - epsi/z
            double delz = p.a0 - epsi / z;
            for (int i = 0; i < m; ++i) delz -= p.a[i] * lam[i];

            // ---- 5. Newton direction (subsolv.m 116-148) ----
            std::vector<double> dx(n), dz_vec(1), dlam(m), dy(m), dxsi(n),
                deta(n), dmu(m);
            double dz = 0.0, dzet = 0.0;
            std::vector<double> ds(m);

            if (m < n) {
                // subsolv.m 117-125
                std::vector<double> blam(m), bb(m + 1);
                for (int i = 0; i < m; ++i) {
                    double acc = dellam[i] + dely[i] / diagy[i];
                    for (int j = 0; j < n; ++j) {
                        acc -= GG[static_cast<std::size_t>(i) * n + j] *
                               (delx[j] / diagx[j]);
                    }
                    blam[i] = acc;
                    bb[i] = acc;
                }
                bb[m] = delz;

                std::vector<double> AA(static_cast<std::size_t>(m + 1) * (m + 1),
                                       0.0);
                for (int i = 0; i < m; ++i) {
                    for (int k = 0; k < m; ++k) {
                        double acc = 0.0;
                        for (int j = 0; j < n; ++j) {
                            acc += GG[static_cast<std::size_t>(i) * n + j] *
                                   (1.0 / diagx[j]) *
                                   GG[static_cast<std::size_t>(k) * n + j];
                        }
                        AA[i * (m + 1) + k] =
                            (i == k ? diaglamyi[i] : 0.0) + acc;
                    }
                    AA[i * (m + 1) + m] = p.a[i];
                    AA[m * (m + 1) + i] = p.a[i];
                }
                AA[m * (m + 1) + m] = -zet / z;

                if (!SolveDense(AA, bb, m + 1)) {
                    SubsolvTraceRecord record;
                    record.event = "newton_step";
                    record.barrier_level = out.barrier_levels;
                    record.newton_iteration = out.inner_newton_iterations;
                    record.newton_iteration_in_barrier = ittt;
                    record.epsi = epsi;
                    record.linear_system_ok = false;
                    EmitTrace(options, record);
                    out.unsupported_branch = true;
                    numerical_failure = true;
                    break;
                }
                for (int i = 0; i < m; ++i) dlam[i] = bb[i];
                dz = bb[m];
                for (int j = 0; j < n; ++j) {
                    double acc = 0.0;
                    for (int i = 0; i < m; ++i) {
                        acc += GG[static_cast<std::size_t>(i) * n + j] * dlam[i];
                    }
                    dx[j] = -delx[j] / diagx[j] - acc / diagx[j];
                }
            } else {
                // subsolv.m 126-141
                if (n > kDenseBranchLimit) {
                    out.unsupported_branch = true;
                    break;
                }
                std::vector<double> diaglamyiinv(m), dellamyi(m);
                for (int i = 0; i < m; ++i) {
                    diaglamyiinv[i] = 1.0 / diaglamyi[i];
                    dellamyi[i] = dellam[i] + dely[i] / diagy[i];
                }
                std::vector<double> Axx(static_cast<std::size_t>(n) * n, 0.0);
                for (int j = 0; j < n; ++j) {
                    for (int k = 0; k < n; ++k) {
                        double acc = 0.0;
                        for (int i = 0; i < m; ++i) {
                            acc += GG[static_cast<std::size_t>(i) * n + j] *
                                   diaglamyiinv[i] *
                                   GG[static_cast<std::size_t>(i) * n + k];
                        }
                        Axx[j * n + k] = (j == k ? diagx[j] : 0.0) + acc;
                    }
                }
                double azz = zet / z;
                std::vector<double> axz(n, 0.0);
                for (int i = 0; i < m; ++i) azz += p.a[i] * p.a[i] * diaglamyiinv[i];
                for (int j = 0; j < n; ++j) {
                    double acc = 0.0;
                    for (int i = 0; i < m; ++i) {
                        acc += GG[static_cast<std::size_t>(i) * n + j] *
                               p.a[i] * diaglamyiinv[i];
                    }
                    axz[j] = -acc;
                }
                std::vector<double> bx(n), bb(n + 1);
                for (int j = 0; j < n; ++j) {
                    double acc = 0.0;
                    for (int i = 0; i < m; ++i) {
                        acc += GG[static_cast<std::size_t>(i) * n + j] *
                               dellamyi[i] * diaglamyiinv[i];
                    }
                    bx[j] = delx[j] + acc;
                    bb[j] = -bx[j];
                }
                double bz = delz;
                for (int i = 0; i < m; ++i) bz -= p.a[i] * dellamyi[i] * diaglamyiinv[i];
                bb[n] = -bz;

                std::vector<double> AA(static_cast<std::size_t>(n + 1) * (n + 1),
                                       0.0);
                for (int j = 0; j < n; ++j) {
                    for (int k = 0; k < n; ++k) AA[j * (n + 1) + k] = Axx[j * n + k];
                    AA[j * (n + 1) + n] = axz[j];
                    AA[n * (n + 1) + j] = axz[j];
                }
                AA[n * (n + 1) + n] = azz;
                if (!SolveDense(AA, bb, n + 1)) {
                    SubsolvTraceRecord record;
                    record.event = "newton_step";
                    record.barrier_level = out.barrier_levels;
                    record.newton_iteration = out.inner_newton_iterations;
                    record.newton_iteration_in_barrier = ittt;
                    record.epsi = epsi;
                    record.linear_system_ok = false;
                    EmitTrace(options, record);
                    out.unsupported_branch = true;
                    numerical_failure = true;
                    break;
                }
                for (int j = 0; j < n; ++j) dx[j] = bb[j];
                dz = bb[n];
                for (int i = 0; i < m; ++i) {
                    double acc = 0.0;
                    for (int j = 0; j < n; ++j) {
                        acc += GG[static_cast<std::size_t>(i) * n + j] * dx[j];
                    }
                    dlam[i] = acc * diaglamyiinv[i] - dz * p.a[i] * diaglamyiinv[i] +
                              dellamyi[i] * diaglamyiinv[i];
                }
            }
            dz_vec[0] = dz;

            // subsolv.m 143-148
            for (int i = 0; i < m; ++i) dy[i] = -dely[i] / diagy[i] + dlam[i] / diagy[i];
            for (int j = 0; j < n; ++j) {
                dxsi[j] = -xsi[j] + epsvecn / (x[j] - p.alfa[j]) -
                          (xsi[j] * dx[j]) / (x[j] - p.alfa[j]);
                deta[j] = -eta[j] + epsvecn / (p.beta[j] - x[j]) +
                          (eta[j] * dx[j]) / (p.beta[j] - x[j]);
            }
            for (int i = 0; i < m; ++i) {
                dmu[i] = -mu[i] + epsvecm / y[i] - (mu[i] * dy[i]) / y[i];
                ds[i] = -s[i] + epsvecm / lam[i] - (s[i] * dlam[i]) / lam[i];
            }
            dzet = -zet + epsi / z - zet * dz / z;

            // ---- 6. maximum feasible step (subsolv.m 149-161) ----
            // xx = [y; z; lam; xsi; eta; mu; zet; s], dxx = its direction.
            double stmxx = 0.0;
            const double k = SubsolvConstants::StepSafety();
            for (int i = 0; i < m; ++i) stmxx = std::max(stmxx, -k * dy[i] / y[i]);
            stmxx = std::max(stmxx, -k * dz / z);
            for (int i = 0; i < m; ++i) stmxx = std::max(stmxx, -k * dlam[i] / lam[i]);
            for (int j = 0; j < n; ++j) {
                stmxx = std::max(stmxx, -k * dxsi[j] / xsi[j]);
                stmxx = std::max(stmxx, -k * deta[j] / eta[j]);
            }
            for (int i = 0; i < m; ++i) stmxx = std::max(stmxx, -k * dmu[i] / mu[i]);
            stmxx = std::max(stmxx, -k * dzet / zet);
            for (int i = 0; i < m; ++i) stmxx = std::max(stmxx, -k * ds[i] / s[i]);

            double stmalbe = 0.0;
            for (int j = 0; j < n; ++j) {
                stmalbe = std::max(stmalbe, -k * dx[j] / (x[j] - p.alfa[j]));
                stmalbe = std::max(stmalbe, k * dx[j] / (p.beta[j] - x[j]));
            }
            const double stminv = std::max(std::max(stmalbe, stmxx), 1.0);
            double steg = 1.0 / stminv;

            // ---- 7-9. trial points and residual-decrease backtracking ----
            // (subsolv.m 163-213)
            const std::vector<double> xold = x, yold = y, lamold = lam,
                                      xsiold = xsi, etaold = eta, muold = mu,
                                      sold = s;
            const double zold = z, zetold = zet;

            int itto = 0;
            double resinew = 2.0 * residunorm;
            int reductions = 0;
            bool exhausted = false;
            while (resinew > residunorm &&
                   itto < SubsolvConstants::BacktrackingReductionLimit()) {
                itto += 1;
                for (int j = 0; j < n; ++j) {
                    x[j] = xold[j] + steg * dx[j];
                    xsi[j] = xsiold[j] + steg * dxsi[j];
                    eta[j] = etaold[j] + steg * deta[j];
                }
                for (int i = 0; i < m; ++i) {
                    y[i] = yold[i] + steg * dy[i];
                    lam[i] = lamold[i] + steg * dlam[i];
                    mu[i] = muold[i] + steg * dmu[i];
                    s[i] = sold[i] + steg * ds[i];
                }
                z = zold + steg * dz;
                zet = zetold + steg * dzet;

                SubsolvResult trial;
                trial.x = x; trial.y = y; trial.z = z; trial.lam = lam;
                trial.xsi = xsi; trial.eta = eta; trial.mu = mu;
                trial.zet = zet; trial.s = s;
                resinew = EvaluateSubsolvResidual(p, trial, epsi).norm2;
                steg = steg / 2.0;
                if (resinew > residunorm &&
                    itto >= SubsolvConstants::BacktrackingReductionLimit()) {
                    exhausted = true;
                }
            }
            if (itto > 0) {
                out.backtracking_iterations += 1;
                reductions = itto - 1;
                if (reductions == 0) out.full_step_iterations += 1;
                out.backtracking_reductions += reductions;
                out.backtracking_max_reductions =
                    std::max(out.backtracking_max_reductions, reductions);
                if (exhausted) out.backtracking_exhausted += 1;
            }

            // subsolv.m 211-213
            residunorm = resinew;
            SubsolvResult cur;
            cur.x = x; cur.y = y; cur.z = z; cur.lam = lam;
            cur.xsi = xsi; cur.eta = eta; cur.mu = mu; cur.zet = zet; cur.s = s;
            const SubsolvResidual rr = EvaluateSubsolvResidual(p, cur, epsi);
            residumax = rr.max_norm;
            SubsolvTraceRecord record;
            record.event = "newton_step";
            record.barrier_level = out.barrier_levels;
            record.newton_iteration = out.inner_newton_iterations;
            record.newton_iteration_in_barrier = ittt;
            record.epsi = epsi;
            record.rex = rr.rex; record.rey = rr.rey; record.rez = rr.rez;
            record.relam = rr.relam; record.rexsi = rr.rexsi; record.reeta = rr.reeta;
            record.remu = rr.remu; record.rezet = rr.rezet; record.res = rr.res;
            record.max_norm = rr.max_norm; record.norm2 = rr.norm2;
            record.newton_step_norm = TraceStepNorm(
                dx, dy, dz, dlam, dxsi, deta, dmu, dzet, ds, steg * 2.0);
            record.backtracking_trials = itto;
            record.backtracking_reductions = reductions;
            record.domain_ok = TraceDomainOk(p, cur);
            record.all_finite = record.domain_ok;
            EmitTrace(options, record);
            steg = 2.0 * steg;

            if (!std::isfinite(rr.max_norm) || !std::isfinite(rr.norm2)) {
                numerical_failure = true;
                break;
            }
        }

        out.max_newton_iterations_per_barrier =
            std::max(out.max_newton_iterations_per_barrier, ittt);
        if (stage_extended) {
            out.extended_barrier_levels += 1;
            out.extra_newton_iterations +=
                std::max(0, ittt - soft_iteration_cap);
        }

        // The soft cap is only a work-budget boundary. Exhaustion of the
        // absolute ceiling is an explicit failure, never normal completion.
        if (!numerical_failure && !domain_failure &&
            ittt >= hard_iteration_cap &&
            residumax > SubsolvConstants::InnerResidualFactor() * epsi) {
            out.capped_barrier_levels += 1;
            hard_cap_exhausted = true;
        }

        {
            SubsolvResult cur;
            cur.x = x; cur.y = y; cur.z = z; cur.lam = lam;
            cur.xsi = xsi; cur.eta = eta; cur.mu = mu; cur.zet = zet; cur.s = s;
            const SubsolvResidual r = EvaluateSubsolvResidual(p, cur, epsi);
            SubsolvTraceRecord record;
            record.event = "barrier_end";
            record.barrier_level = out.barrier_levels;
            record.newton_iteration = out.inner_newton_iterations;
            record.newton_iteration_in_barrier = ittt;
            record.epsi = epsi;
            record.rex = r.rex; record.rey = r.rey; record.rez = r.rez;
            record.relam = r.relam; record.rexsi = r.rexsi; record.reeta = r.reeta;
            record.remu = r.remu; record.rezet = r.rezet; record.res = r.res;
            record.max_norm = r.max_norm; record.norm2 = r.norm2;
            record.domain_ok = TraceDomainOk(p, cur);
            record.all_finite = record.domain_ok;
            EmitTrace(options, record);
        }

        // ---- 11. next barrier level (subsolv.m 219) ----
        epsi = epsi * SubsolvConstants::BarrierReduction();
        if (numerical_failure || domain_failure ||
            (hard_cap_exhausted && options.stop_on_hard_cap)) {
            break;
        }
    }

    // ---- return (subsolv.m 221-229) ----
    out.x = x; out.y = y; out.z = z; out.lam = lam;
    out.xsi = xsi; out.eta = eta; out.mu = mu; out.zet = zet; out.s = s;

    out.all_finite = std::isfinite(z) && std::isfinite(zet);
    for (int j = 0; j < n && out.all_finite; ++j) {
        out.all_finite = std::isfinite(x[j]) && std::isfinite(xsi[j]) &&
                         std::isfinite(eta[j]);
    }
    for (int i = 0; i < m && out.all_finite; ++i) {
        out.all_finite = std::isfinite(y[i]) && std::isfinite(lam[i]) &&
                         std::isfinite(mu[i]) && std::isfinite(s[i]);
    }

    bool in_domain = out.all_finite && z >= 0.0 && zet >= 0.0;
    for (int j = 0; j < n && in_domain; ++j) {
        in_domain = x[j] >= p.alfa[j] && x[j] <= p.beta[j];
    }
    for (int i = 0; i < m && in_domain; ++i) {
        in_domain = y[i] >= 0.0 && lam[i] >= 0.0 && mu[i] >= 0.0 && s[i] >= 0.0;
    }
    out.domain_ok = in_domain;

    out.residual = EvaluateSubsolvResidual(p, out, out.epsi_final);
    if (numerical_failure || out.unsupported_branch || !out.all_finite ||
        !std::isfinite(out.residual.max_norm)) {
        out.status = SubsolvSolveStatus::NumericalFailure;
    } else if (domain_failure || !out.domain_ok) {
        out.status = SubsolvSolveStatus::DomainFailure;
    } else if (hard_cap_exhausted) {
        out.status = SubsolvSolveStatus::HardCapExhausted;
    } else if (out.extended_barrier_levels > 0) {
        out.status = SubsolvSolveStatus::ConvergedAfterSoftCapExtension;
    } else {
        out.status = SubsolvSolveStatus::ConvergedWithinSoftCap;
    }
    return out;
}

} // namespace mma

namespace mma {
namespace {

constexpr char kReplayMagic[] = "MMARPLY1";

template <typename T>
void WritePod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed writing MMA replay fixture");
}

template <typename T>
T ReadPod(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated MMA replay fixture");
    return value;
}

void WriteVector(std::ostream& out, const std::vector<double>& values) {
    WritePod<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(double)));
        if (!out) throw std::runtime_error("failed writing MMA replay vector");
    }
}

std::vector<double> ReadVector(std::istream& in) {
    const std::uint64_t size = ReadPod<std::uint64_t>(in);
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(double))) {
        throw std::runtime_error("invalid MMA replay vector size");
    }
    std::vector<double> values(static_cast<std::size_t>(size));
    if (!values.empty()) {
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(double)));
        if (!in) throw std::runtime_error("truncated MMA replay vector");
    }
    return values;
}

void WriteResidual(std::ostream& out, const SubsolvResidual& r) {
    WriteVector(out, r.rex_values); WriteVector(out, r.rey_values);
    WriteVector(out, r.relam_values); WriteVector(out, r.rexsi_values);
    WriteVector(out, r.reeta_values); WriteVector(out, r.remu_values);
    WriteVector(out, r.res_values);
    WritePod(out, r.rez_value); WritePod(out, r.rezet_value);
    WritePod(out, r.rex); WritePod(out, r.rey); WritePod(out, r.rez);
    WritePod(out, r.relam); WritePod(out, r.rexsi); WritePod(out, r.reeta);
    WritePod(out, r.remu); WritePod(out, r.rezet); WritePod(out, r.res);
    WritePod(out, r.max_norm); WritePod(out, r.norm2);
}

SubsolvResidual ReadResidual(std::istream& in) {
    SubsolvResidual r;
    r.rex_values = ReadVector(in); r.rey_values = ReadVector(in);
    r.relam_values = ReadVector(in); r.rexsi_values = ReadVector(in);
    r.reeta_values = ReadVector(in); r.remu_values = ReadVector(in);
    r.res_values = ReadVector(in);
    r.rez_value = ReadPod<double>(in); r.rezet_value = ReadPod<double>(in);
    r.rex = ReadPod<double>(in); r.rey = ReadPod<double>(in);
    r.rez = ReadPod<double>(in); r.relam = ReadPod<double>(in);
    r.rexsi = ReadPod<double>(in); r.reeta = ReadPod<double>(in);
    r.remu = ReadPod<double>(in); r.rezet = ReadPod<double>(in);
    r.res = ReadPod<double>(in); r.max_norm = ReadPod<double>(in);
    r.norm2 = ReadPod<double>(in);
    return r;
}

void WriteProblem(std::ostream& out, const SubsolvProblem& p) {
    WritePod(out, p.n); WritePod(out, p.m); WritePod(out, p.epsimin);
    WriteVector(out, p.low); WriteVector(out, p.upp); WriteVector(out, p.alfa);
    WriteVector(out, p.beta); WriteVector(out, p.p0); WriteVector(out, p.q0);
    WriteVector(out, p.P); WriteVector(out, p.Q); WritePod(out, p.a0);
    WriteVector(out, p.a); WriteVector(out, p.b); WriteVector(out, p.c);
    WriteVector(out, p.d);
}

SubsolvProblem ReadProblem(std::istream& in) {
    SubsolvProblem p;
    p.n = ReadPod<int>(in); p.m = ReadPod<int>(in);
    p.epsimin = ReadPod<double>(in); p.low = ReadVector(in); p.upp = ReadVector(in);
    p.alfa = ReadVector(in); p.beta = ReadVector(in); p.p0 = ReadVector(in);
    p.q0 = ReadVector(in); p.P = ReadVector(in); p.Q = ReadVector(in);
    p.a0 = ReadPod<double>(in); p.a = ReadVector(in); p.b = ReadVector(in);
    p.c = ReadVector(in); p.d = ReadVector(in);
    return p;
}

void WriteResult(std::ostream& out, const SubsolvResult& r) {
    WriteVector(out, r.x); WriteVector(out, r.y); WritePod(out, r.z);
    WriteVector(out, r.lam); WriteVector(out, r.xsi); WriteVector(out, r.eta);
    WriteVector(out, r.mu); WritePod(out, r.zet); WriteVector(out, r.s);
    WriteResidual(out, r.residual); WritePod(out, r.epsi_final);
    WritePod(out, r.barrier_levels); WritePod(out, r.inner_newton_iterations);
    WritePod(out, r.capped_barrier_levels);
    WritePod(out, r.backtracking_iterations); WritePod(out, r.backtracking_reductions);
    WritePod(out, r.backtracking_max_reductions);
    WritePod(out, r.backtracking_exhausted); WritePod(out, r.full_step_iterations);
    WritePod(out, r.domain_ok); WritePod(out, r.all_finite);
    WritePod(out, r.unsupported_branch);
}

SubsolvResult ReadResult(std::istream& in) {
    SubsolvResult r;
    r.x = ReadVector(in); r.y = ReadVector(in); r.z = ReadPod<double>(in);
    r.lam = ReadVector(in); r.xsi = ReadVector(in); r.eta = ReadVector(in);
    r.mu = ReadVector(in); r.zet = ReadPod<double>(in); r.s = ReadVector(in);
    r.residual = ReadResidual(in); r.epsi_final = ReadPod<double>(in);
    r.barrier_levels = ReadPod<int>(in); r.inner_newton_iterations = ReadPod<int>(in);
    r.capped_barrier_levels = ReadPod<int>(in);
    r.backtracking_iterations = ReadPod<int>(in); r.backtracking_reductions = ReadPod<int>(in);
    r.backtracking_max_reductions = ReadPod<int>(in);
    r.backtracking_exhausted = ReadPod<int>(in); r.full_step_iterations = ReadPod<int>(in);
    r.domain_ok = ReadPod<bool>(in); r.all_finite = ReadPod<bool>(in);
    r.unsupported_branch = ReadPod<bool>(in);
    return r;
}

void RequireClose(double a, double b, double tolerance, const char* what) {
    if (a == b) return;
    if (tolerance > 0.0 && std::abs(a - b) <= tolerance) return;
    throw std::runtime_error(std::string("MMA replay mismatch: ") + what);
}

void RequireVector(const std::vector<double>& a, const std::vector<double>& b,
                  double tolerance, const char* what) {
    if (a.size() != b.size()) throw std::runtime_error(std::string("MMA replay size mismatch: ") + what);
    for (std::size_t i = 0; i < a.size(); ++i) RequireClose(a[i], b[i], tolerance, what);
}

void RequireDimension(const char* field, std::size_t actual,
                      std::size_t expected) {
    if (actual != expected) {
        throw std::runtime_error(std::string("INVALID_FIXTURE_DIMENSION: ") +
                                 field + " has size " +
                                 std::to_string(actual) + ", expected " +
                                 std::to_string(expected));
    }
}

void ValidateMmaReplayFixtureDimensions(const MmaReplayFixture& f) {
    if (f.n < 0 || f.m < 0) {
        throw std::runtime_error(
            "INVALID_FIXTURE_DIMENSION: negative header dimension");
    }
    const std::size_t n = static_cast<std::size_t>(f.n);
    const std::size_t m = static_cast<std::size_t>(f.m);
    const std::size_t nm = n * m;
    RequireDimension("xval", f.xval.size(), n);
    RequireDimension("xold1", f.xold1.size(), n);
    RequireDimension("xold2", f.xold2.size(), n);
    RequireDimension("xmin", f.xmin.size(), n);
    RequireDimension("xmax", f.xmax.size(), n);
    RequireDimension("low", f.low.size(), n);
    RequireDimension("upp", f.upp.size(), n);
    RequireDimension("dfdx", f.dfdx.size(), n);
    RequireDimension("gx", f.gx.size(), m);
    RequireDimension("dgdx", f.dgdx.size(), nm);
    RequireDimension("a", f.a.size(), m);
    RequireDimension("c", f.c.size(), m);
    RequireDimension("d", f.d.size(), m);

    if (f.problem.n != f.n || f.problem.m != f.m) {
        throw std::runtime_error(
            "INVALID_FIXTURE_DIMENSION: SubsolvProblem header disagrees with fixture header");
    }
    RequireDimension("problem.low", f.problem.low.size(), n);
    RequireDimension("problem.upp", f.problem.upp.size(), n);
    RequireDimension("problem.alfa", f.problem.alfa.size(), n);
    RequireDimension("problem.beta", f.problem.beta.size(), n);
    RequireDimension("problem.p0", f.problem.p0.size(), n);
    RequireDimension("problem.q0", f.problem.q0.size(), n);
    RequireDimension("problem.P", f.problem.P.size(), nm);
    RequireDimension("problem.Q", f.problem.Q.size(), nm);
    RequireDimension("problem.a", f.problem.a.size(), m);
    RequireDimension("problem.b", f.problem.b.size(), m);
    RequireDimension("problem.c", f.problem.c.size(), m);
    RequireDimension("problem.d", f.problem.d.size(), m);

    RequireDimension("result.x", f.result.x.size(), n);
    RequireDimension("result.y", f.result.y.size(), m);
    RequireDimension("result.lam", f.result.lam.size(), m);
    RequireDimension("result.xsi", f.result.xsi.size(), n);
    RequireDimension("result.eta", f.result.eta.size(), n);
    RequireDimension("result.mu", f.result.mu.size(), m);
    RequireDimension("result.s", f.result.s.size(), m);
    RequireDimension("residual.rex_values", f.result.residual.rex_values.size(), n);
    RequireDimension("residual.rey_values", f.result.residual.rey_values.size(), m);
    RequireDimension("residual.relam_values", f.result.residual.relam_values.size(), m);
    RequireDimension("residual.rexsi_values", f.result.residual.rexsi_values.size(), n);
    RequireDimension("residual.reeta_values", f.result.residual.reeta_values.size(), n);
    RequireDimension("residual.remu_values", f.result.residual.remu_values.size(), m);
    RequireDimension("residual.res_values", f.result.residual.res_values.size(), m);
}

void VerifyResult(const SubsolvResult& expected, const SubsolvResult& actual,
                  double tolerance) {
    RequireVector(expected.x, actual.x, tolerance, "x");
    RequireVector(expected.y, actual.y, tolerance, "y");
    RequireClose(expected.z, actual.z, tolerance, "z");
    RequireVector(expected.lam, actual.lam, tolerance, "lambda");
    RequireVector(expected.xsi, actual.xsi, tolerance, "xsi");
    RequireVector(expected.eta, actual.eta, tolerance, "eta");
    RequireVector(expected.mu, actual.mu, tolerance, "mu");
    RequireClose(expected.zet, actual.zet, tolerance, "zet");
    RequireVector(expected.s, actual.s, tolerance, "s");
    const auto& e = expected.residual; const auto& a = actual.residual;
    RequireVector(e.rex_values, a.rex_values, tolerance, "rex values");
    RequireVector(e.rey_values, a.rey_values, tolerance, "rey values");
    RequireVector(e.relam_values, a.relam_values, tolerance, "relam values");
    RequireVector(e.rexsi_values, a.rexsi_values, tolerance, "rexsi values");
    RequireVector(e.reeta_values, a.reeta_values, tolerance, "reeta values");
    RequireVector(e.remu_values, a.remu_values, tolerance, "remu values");
    RequireVector(e.res_values, a.res_values, tolerance, "res values");
    RequireClose(e.rez_value, a.rez_value, tolerance, "rez value");
    RequireClose(e.rezet_value, a.rezet_value, tolerance, "rezet value");
    RequireClose(e.max_norm, a.max_norm, tolerance, "max norm");
    RequireClose(e.norm2, a.norm2, tolerance, "norm2");
    if (expected.barrier_levels != actual.barrier_levels ||
        expected.inner_newton_iterations != actual.inner_newton_iterations ||
        expected.capped_barrier_levels != actual.capped_barrier_levels ||
        expected.domain_ok != actual.domain_ok || expected.all_finite != actual.all_finite ||
        expected.unsupported_branch != actual.unsupported_branch) {
        throw std::runtime_error("MMA replay diagnostic mismatch");
    }
}

} // namespace

void WriteMmaReplayFixture(const MmaReplayFixture& f, const std::string& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open MMA replay fixture for writing");
    out.write(kReplayMagic, sizeof(kReplayMagic) - 1);
    WritePod(out, f.format_version); WritePod(out, f.iteration);
    WritePod(out, f.update_number); WritePod(out, f.n); WritePod(out, f.m);
    WriteVector(out, f.xval); WriteVector(out, f.xold1); WriteVector(out, f.xold2);
    WriteVector(out, f.xmin); WriteVector(out, f.xmax); WriteVector(out, f.low);
    WriteVector(out, f.upp); WriteVector(out, f.dfdx); WriteVector(out, f.gx);
    WriteVector(out, f.dgdx); WriteVector(out, f.df0dx); WriteVector(out, f.fval);
    WriteVector(out, f.a); WriteVector(out, f.c); WriteVector(out, f.d);
    WritePod(out, f.a0); WritePod(out, f.f0val); WritePod(out, f.objective_values_supplied);
    WritePod(out, f.epsimin); WritePod(out, f.xmamieps); WritePod(out, f.raa0);
    WritePod(out, f.move); WritePod(out, f.albefa); WritePod(out, f.asyminit);
    WritePod(out, f.asymdec); WritePod(out, f.asyminc);
    WriteProblem(out, f.problem); WriteResult(out, f.result);
    WritePod(out, f.objective_value); WritePod(out, f.constraint_value_norm);
    WritePod(out, f.before_design_hash); WritePod(out, f.candidate_design_hash);
    WritePod(out, f.after_rejection_design_hash); WritePod(out, f.before_history_hash);
    WritePod(out, f.candidate_history_hash); WritePod(out, f.after_rollback_history_hash);
    WritePod(out, f.update_rejected);
}

MmaReplayFixture ReadMmaReplayFixture(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open MMA replay fixture for reading");
    char magic[sizeof(kReplayMagic) - 1]{};
    in.read(magic, sizeof(magic));
    if (!in || std::string(magic, sizeof(magic)) != std::string(kReplayMagic, sizeof(kReplayMagic) - 1))
        throw std::runtime_error("invalid MMA replay fixture magic");
    MmaReplayFixture f;
    f.format_version = ReadPod<std::uint32_t>(in);
    if (f.format_version != MmaReplayFixture::kFormatVersion)
        throw std::runtime_error("unsupported MMA replay fixture version");
    f.iteration = ReadPod<int>(in); f.update_number = ReadPod<int>(in);
    f.n = ReadPod<int>(in); f.m = ReadPod<int>(in);
    f.xval = ReadVector(in); f.xold1 = ReadVector(in); f.xold2 = ReadVector(in);
    f.xmin = ReadVector(in); f.xmax = ReadVector(in); f.low = ReadVector(in);
    f.upp = ReadVector(in); f.dfdx = ReadVector(in); f.gx = ReadVector(in);
    f.dgdx = ReadVector(in); f.df0dx = ReadVector(in); f.fval = ReadVector(in);
    f.a = ReadVector(in); f.c = ReadVector(in); f.d = ReadVector(in);
    f.a0 = ReadPod<double>(in); f.f0val = ReadPod<double>(in);
    f.objective_values_supplied = ReadPod<bool>(in);
    f.epsimin = ReadPod<double>(in); f.xmamieps = ReadPod<double>(in);
    f.raa0 = ReadPod<double>(in); f.move = ReadPod<double>(in);
    f.albefa = ReadPod<double>(in); f.asyminit = ReadPod<double>(in);
    f.asymdec = ReadPod<double>(in); f.asyminc = ReadPod<double>(in);
    f.problem = ReadProblem(in); f.result = ReadResult(in);
    f.objective_value = ReadPod<double>(in); f.constraint_value_norm = ReadPod<double>(in);
    f.before_design_hash = ReadPod<std::uint64_t>(in); f.candidate_design_hash = ReadPod<std::uint64_t>(in);
    f.after_rejection_design_hash = ReadPod<std::uint64_t>(in);
    f.before_history_hash = ReadPod<std::uint64_t>(in); f.candidate_history_hash = ReadPod<std::uint64_t>(in);
    f.after_rollback_history_hash = ReadPod<std::uint64_t>(in); f.update_rejected = ReadPod<bool>(in);
    ValidateMmaReplayFixtureDimensions(f);
    return f;
}

void VerifyMmaReplayFixture(const MmaReplayFixture& fixture, double tolerance) {
    ValidateMmaReplayFixtureDimensions(fixture);
    // Version-1 captures predate the soft-cap extension and intentionally
    // replay the historical 200-step boundary. This preserves compatibility
    // of the immutable replay artifact while the production call uses the
    // extended policy.
    SubsolvSolveOptions legacy_options;
    legacy_options.inner_iteration_cap = SubsolvConstants::InnerIterationCap();
    legacy_options.stop_on_hard_cap = false;
    const SubsolvResult actual =
        SolveSubsolvFull(fixture.problem, legacy_options);
    VerifyResult(fixture.result, actual, tolerance);
}

} // namespace mma
