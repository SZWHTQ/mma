////////////////////////////////////////////////////////////////////////////////
// Copyright © 2018 Jérémie Dumas
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////

#include "MMASolver.h"
#include <algorithm>
#include <cmath>

namespace mma {
////////////////////////////////////////////////////////////////////////////////
// PUBLIC
////////////////////////////////////////////////////////////////////////////////

MMASolver::MMASolver(int nn, int mm, double ai, double ci, double di)
    : n(nn), m(mm), iter(0), xmamieps(1.0e-5)
      //, epsimin(1e-7)
      ,
      epsimin(std::sqrt(n + m) * 1e-9), raa0(0.00001), move(0.5), albefa(0.1),
      asyminit(0.5) // 0.2;
      ,
      asymdec(0.7) // 0.65;
      ,
      asyminc(1.2) // 1.08;
      ,
      a(m, ai), c(m, ci), d(m, di), y(m), lam(m), mu(m), low(n),
      upp(n), alpha(n), beta(n), p0(n), q0(n), pij(n * m), qij(n * m), b(m),
      xold1(n), xold2(n), m_committed_xold1(n),
      m_committed_xold2(n), m_committed_low(n), m_committed_upp(n),
      m_input_x(n) {}

void MMASolver::SetAsymptotes(double init, double decrease, double increase) {

    // asymptotes initialization and increase/decrease
    asyminit = init;
    asymdec = decrease;
    asyminc = increase;
}

void MMASolver::Update(double* xval, const double* dfdx, const double* gx,
                       const double* dgdx, const double* xmin,
                       const double* xmax) {
    // The iteration state a solve is allowed to commit. Captured before GenSub
    // so that a rejected solve can be undone exactly.
    SnapshotIterationState(xval);

    // Generate the subproblem
    GenSub(xval, dfdx, gx, dgdx, xmin, xmax);

    // Update xolds
    xold2 = xold1;
    std::copy_n(xval, n, xold1.data());

    // Solve the dual with an interior point method
    SolveDIP(xval);

    // Solve the dual with a steepest ascent method
    // SolveDSA(xval);

    // Observe the returned dual point, then accept or reject it. A rejected
    // solve restores the iteration state captured above and leaves the caller's
    // design vector untouched, so a failed subproblem cannot advance the
    // asymptote history or propose a garbage design step.
    QualifyDualSolve(xval);
    if (m_dual.status != DualSolveStatus::Success) {
        RestoreIterationState(xval);
    }
}

void MMASolver::SnapshotIterationState(const double* xval) {
    m_dual.update_rejected = false;
    m_committed_iter = iter;
    m_committed_xold1 = xold1;
    m_committed_xold2 = xold2;
    m_committed_low = low;
    m_committed_upp = upp;
    m_input_x.assign(xval, xval + n);
}

void MMASolver::RestoreIterationState(double* xval) {
    iter = m_committed_iter;
    xold1 = m_committed_xold1;
    xold2 = m_committed_xold2;
    low = m_committed_low;
    upp = m_committed_upp;
    std::copy_n(m_input_x.begin(), n, xval);
    m_dual.update_rejected = true;
}

/**
 * Qualify the dual point SolveDIP returned. This is a pure observation: it reads
 * the solver state and fills m_dual, and changes nothing the arithmetic depends
 * on.
 */
void MMASolver::QualifyDualSolve(const double* x) {
    m_dual.lambda = lam;
    m_dual.mu = mu;
    m_dual.epsimin = epsimin;

    bool all_finite = std::isfinite(z);
    for (int j = 0; j < m && all_finite; ++j) {
        all_finite = std::isfinite(lam[j]) && std::isfinite(mu[j]) &&
                     std::isfinite(y[j]);
    }
    for (int i = 0; i < n && all_finite; ++i) {
        all_finite = std::isfinite(x[i]);
    }
    m_dual.all_finite = all_finite;

    // XYZofLAMBDA clamps the primal iterate into [alpha, beta], so this holds by
    // construction for any finite iterate; the check exists to catch a
    // non-finite or otherwise corrupted box, not to re-derive the clamp.
    bool within_box = true;
    for (int i = 0; i < n; ++i) {
        const double width = std::abs(beta[i] - alpha[i]);
        const double tol = 1.0e-12 * std::max(1.0, width);
        if (x[i] < alpha[i] - tol || x[i] > beta[i] + tol) {
            within_box = false;
        }
    }
    m_dual.design_within_subproblem_box = within_box;

    const bool finite_ok = all_finite && m_dual.all_finite;

    // ---- the m9 acceptance policy -----------------------------------
    //
    // Acceptance is decided by the quality of the returned point, not by how
    // the solver got there. `capped_barrier_levels` stays observable but is
    // NOT a failure clause on its own: m8 measured a fixture that saturates the
    // inner cap and still delivers a residual 3e4x inside the threshold while
    // converging to the analytic optimum (§9, G6).
    //
    // Structural clauses, any of which is fatal whatever the residual:
    //   * every returned state entry is finite;
    //   * the point lies inside its own domain (alfa <= x <= beta,
    //     y,z,lam,mu,s,zet >= 0);
    //   * the reference's Newton branch was taken as written.
    // Quality clause: the full primal-dual KKT/barrier residual, which is the
    // solver's own convergence measure, within `KktResidualToleranceFactor()`
    // times epsimin.
    m_dual.kkt_residual_tolerance = KktResidualToleranceFactor() * epsimin;
    const bool kkt_ok = m_dual.kkt.max_norm <= m_dual.kkt_residual_tolerance &&
                        std::isfinite(m_dual.kkt.max_norm);
    const bool structural_ok = finite_ok && within_box &&
                               m_dual.kkt_domain_ok &&
                               !m_dual.kkt_unsupported_branch;

    m_dual.status = (structural_ok && kkt_ok) ? DualSolveStatus::Success
                                              : DualSolveStatus::FailedToConverge;
}

////////////////////////////////////////////////////////////////////////////////
// PRIVATE
////////////////////////////////////////////////////////////////////////////////

/**
 * The solve path (m9): hand the generated subproblem to the full primal-dual
 * solver and adopt what it returns.
 *
 * GenSub is untouched and produces exactly the quantities `subsolv.m` takes.
 * The only conversion is the layout of P/Q: production stores them as
 * `pij[i*m + j]` (design index i, constraint index j), the reference as
 * `P(i,j)` with i the constraint, so the two are transposes of each other.
 */
void MMASolver::SolveDIP(double* x) {
    SubsolvProblem sp;
    sp.n = n;
    sp.m = m;
    sp.epsimin = epsimin;
    sp.low = low;
    sp.upp = upp;
    sp.alfa = alpha;
    sp.beta = beta;
    sp.p0 = p0;
    sp.q0 = q0;
    sp.P.assign(static_cast<std::size_t>(m) * n, 0.0);
    sp.Q.assign(static_cast<std::size_t>(m) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            sp.P[static_cast<std::size_t>(j) * n + i] = pij[i * m + j];
            sp.Q[static_cast<std::size_t>(j) * n + i] = qij[i * m + j];
        }
    }
    sp.a0 = 1.0;  // production's XYZofLAMBDA encodes a0 = 1
    sp.a = a;
    sp.b = b;
    sp.c = c;
    sp.d = d;

    const SubsolvResult r = SolveSubsolvFull(sp);

    if (r.x.size() == static_cast<std::size_t>(n)) {
        std::copy(r.x.begin(), r.x.end(), x);
    }
    lam = r.lam;
    mu = r.mu;
    y = r.y;
    z = r.z;

    // Counters and residuals, for QualifyDualSolve.
    m_dual.barrier_levels = r.barrier_levels;
    m_dual.capped_barrier_levels = r.capped_barrier_levels;
    m_dual.inner_newton_iterations = r.inner_newton_iterations;
    m_dual.epsi_final = r.epsi_final;
    m_dual.all_finite = r.all_finite;
    m_dual.kkt = r.residual;
    m_dual.kkt_domain_ok = r.domain_ok;
    m_dual.kkt_unsupported_branch = r.unsupported_branch;
    m_dual.full_step_iterations = r.full_step_iterations;
    m_dual.backtracking_iterations = r.backtracking_iterations;
    m_dual.backtracking_reductions = r.backtracking_reductions;
    m_dual.backtracking_max_reductions = r.backtracking_max_reductions;
    m_dual.backtracking_exhausted = r.backtracking_exhausted;
}

void MMASolver::GenSub(const double* xval, const double* dfdx, const double* gx,
                       const double* dgdx, const double* xmin,
                       const double* xmax) {
    // Forward the iterator
    iter++;

    // Set asymptotes
    if (iter < 3) {
#ifdef MMA_WITH_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < n; i++) {
            low[i] = xval[i] - asyminit * (xmax[i] - xmin[i]);
            upp[i] = xval[i] + asyminit * (xmax[i] - xmin[i]);
        }
    } else {
#ifdef MMA_WITH_OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < n; i++) {
            double zzz = (xval[i] - xold1[i]) * (xold1[i] - xold2[i]);
            double gamma;
            if (zzz < 0.0) {
                gamma = asymdec;
            } else if (zzz > 0.0) {
                gamma = asyminc;
            } else {
                gamma = 1.0;
            }
            low[i] = xval[i] - gamma * (xold1[i] - low[i]);
            upp[i] = xval[i] + gamma * (upp[i] - xold1[i]);

            double xmami = std::max(xmamieps, xmax[i] - xmin[i]);
            // double xmami = xmax[i] - xmin[i];
            low[i] = std::max(low[i], xval[i] - 100.0 * xmami);
            low[i] = std::min(low[i], xval[i] - 1.0e-5 * xmami);
            upp[i] = std::max(upp[i], xval[i] + 1.0e-5 * xmami);
            upp[i] = std::min(upp[i], xval[i] + 100.0 * xmami);

            double xmi = xmin[i] - 1.0e-6;
            double xma = xmax[i] + 1.0e-6;
            if (xval[i] < xmi) {
                low[i] = xval[i] - (xma - xval[i]) / 0.9;
                upp[i] = xval[i] + (xma - xval[i]) / 0.9;
            }
            if (xval[i] > xma) {
                low[i] = xval[i] - (xval[i] - xmi) / 0.9;
                upp[i] = xval[i] + (xval[i] - xmi) / 0.9;
            }
        }
    }

// Set bounds and the coefficients for the approximation
// double raa0 = 0.5*1e-6;
#ifdef MMA_WITH_OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; ++i) {
        // Compute bounds alpha and beta
        alpha[i] = std::max(xmin[i], low[i] + albefa * (xval[i] - low[i]));
        alpha[i] = std::max(alpha[i], xval[i] - move * (xmax[i] - xmin[i]));
        alpha[i] = std::min(alpha[i], xmax[i]);
        beta[i] = std::min(xmax[i], upp[i] - albefa * (upp[i] - xval[i]));
        beta[i] = std::min(beta[i], xval[i] + move * (xmax[i] - xmin[i]));
        beta[i] = std::max(beta[i], xmin[i]);

        // Objective function
        {
            double dfdxp = std::max(0.0, dfdx[i]);
            double dfdxm = std::max(0.0, -1.0 * dfdx[i]);
            double xmamiinv = 1.0 / std::max(xmamieps, xmax[i] - xmin[i]);
            double pq = 0.001 * std::abs(dfdx[i]) + raa0 * xmamiinv;
            p0[i] = std::pow(upp[i] - xval[i], 2.0) * (dfdxp + pq);
            q0[i] = std::pow(xval[i] - low[i], 2.0) * (dfdxm + pq);
        }

        // Constraints
        for (int j = 0; j < m; j++) {
            double dgdxp = std::max(0.0, dgdx[i * m + j]);
            double dgdxm = std::max(0.0, -1.0 * dgdx[i * m + j]);
            double xmamiinv = 1.0 / std::max(xmamieps, xmax[i] - xmin[i]);
            double pq = 0.001 * std::abs(dgdx[i * m + j]) + raa0 * xmamiinv;
            pij[i * m + j] = std::pow(upp[i] - xval[i], 2.0) * (dgdxp + pq);
            qij[i * m + j] = std::pow(xval[i] - low[i], 2.0) * (dgdxm + pq);
        }
    }

    // The constant for the constraints
    for (int j = 0; j < m; j++) {
        b[j] = -gx[j];
        for (int i = 0; i < n; i++) {
            b[j] += pij[i * m + j] / (upp[i] - xval[i]) +
                    qij[i * m + j] / (xval[i] - low[i]);
        }
    }
}

} // namespace mma
