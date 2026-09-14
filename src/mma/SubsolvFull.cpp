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
    r.norm2 = std::sqrt(s2);
    r.max_norm = std::max({r.rex, r.rey, r.rez, r.relam, r.rexsi, r.reeta, r.remu,
                           r.rezet, r.res});
    return r;
}

SubsolvResult SolveSubsolvFull(const SubsolvProblem& p) {
    const int n = p.n, m = p.m;
    const double epsimin = p.epsimin;
    SubsolvResult out;

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

        // ---- 4-10. inner Newton loop (subsolv.m 87-214) ----
        while (residumax > SubsolvConstants::InnerResidualFactor() * epsi &&
               ittt < SubsolvConstants::InnerIterationCap()) {
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
                    out.unsupported_branch = true;
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
                    out.unsupported_branch = true;
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
            steg = 2.0 * steg;
        }

        // A level that exhausted the inner cap is recorded, never used on its
        // own to reject the solve (m9 handoff section 9).
        if (ittt >= SubsolvConstants::InnerIterationCap() &&
            residumax > SubsolvConstants::InnerResidualFactor() * epsi) {
            out.capped_barrier_levels += 1;
        }

        // ---- 11. next barrier level (subsolv.m 219) ----
        epsi = epsi * SubsolvConstants::BarrierReduction();
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
    return out;
}

} // namespace mma
