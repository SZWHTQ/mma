#include <stdexcept>
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
#include <cstdint>
#include <cstring>

namespace mma {

namespace {

std::uint64_t HashBytes(std::uint64_t hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::uint64_t HashVector(std::uint64_t hash, const std::vector<double>& values) {
    const std::uint64_t size = values.size();
    hash = HashBytes(hash, &size, sizeof(size));
    return values.empty() ? hash : HashBytes(hash, values.data(), values.size() * sizeof(double));
}

std::uint64_t HashHistory(int iter, const std::vector<double>& xold1,
                         const std::vector<double>& xold2,
                         const std::vector<double>& low,
                         const std::vector<double>& upp) {
    std::uint64_t hash = 1469598103934665603ULL;
    hash = HashBytes(hash, &iter, sizeof(iter));
    hash = HashVector(hash, xold1); hash = HashVector(hash, xold2);
    hash = HashVector(hash, low); hash = HashVector(hash, upp);
    return hash;
}

} // namespace
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
                       const double* xmax, const double* move_scale) {
    if (move_scale) {
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(move_scale[i]) || move_scale[i] <= 0.0 ||
                move_scale[i] > 1.0) {
                throw std::invalid_argument(
                    "MMA move_scale entries must be finite and in (0, 1]");
            }
        }
    }

    // The iteration state a solve is allowed to commit. Captured before GenSub
    // so that a rejected solve can be undone exactly.
    SnapshotIterationState(xval);

    last_bounds.alpha_standard.resize(n);
    last_bounds.beta_standard.resize(n);

    // Generate the subproblem
    GenSub(xval, dfdx, gx, dgdx, xmin, xmax, move_scale);
    last_bounds.alpha = alpha;
    last_bounds.beta = beta;

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
    m_replay_capture.candidate_design_hash =
        HashBytes(1469598103934665603ULL, xval, sizeof(double) * n);
    m_replay_capture.candidate_history_hash =
        HashHistory(iter, xold1, xold2, low, upp);
    if (m_dual.status != DualSolveStatus::Success) {
        RestoreIterationState(xval);
    }
    m_replay_capture.after_rejection_design_hash =
        HashBytes(1469598103934665603ULL, xval, sizeof(double) * n);
    m_replay_capture.after_rollback_history_hash =
        HashHistory(iter, xold1, xold2, low, upp);
    m_replay_capture.update_rejected = m_dual.update_rejected;
    if (m_replay_callback) m_replay_callback(m_replay_capture);
}

void MMASolver::SnapshotIterationState(const double* xval) {
    m_dual.update_rejected = false;
    m_committed_iter = iter;
    m_committed_xold1 = xold1;
    m_committed_xold2 = xold2;
    m_committed_low = low;
    m_committed_upp = upp;
    m_input_x.assign(xval, xval + n);
    m_replay_capture = MmaReplayFixture{};
    m_replay_capture.iteration = iter;
    m_replay_capture.update_number = iter + 1;
    m_replay_capture.n = n;
    m_replay_capture.m = m;
    m_replay_capture.xval = m_input_x;
    m_replay_capture.xold1 = xold1;
    m_replay_capture.xold2 = xold2;
    m_replay_capture.low = low;
    m_replay_capture.upp = upp;
    m_replay_capture.a = a;
    m_replay_capture.c = c;
    m_replay_capture.d = d;
    m_replay_capture.a0 = 1.0;
    m_replay_capture.epsimin = epsimin;
    m_replay_capture.xmamieps = xmamieps;
    m_replay_capture.raa0 = raa0;
    m_replay_capture.move = move;
    m_replay_capture.albefa = albefa;
    m_replay_capture.asyminit = asyminit;
    m_replay_capture.asymdec = asymdec;
    m_replay_capture.asyminc = asyminc;
    m_replay_capture.before_design_hash =
        HashBytes(1469598103934665603ULL, xval, sizeof(double) * n);
    m_replay_capture.before_history_hash = HashHistory(iter, xold1, xold2, low, upp);
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

    // ---- the m17g acceptance policy ---------------------------------
    //
    // Acceptance is decided by the quality of the returned point and by the
    // full solver's explicit convergence status. A soft-cap crossing is
    // allowed only when the same stage continues and converges; hard-cap,
    // numerical, and domain outcomes are never accepted based on residual
    // proximity to the outer reliability band.
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
    const bool inner_status_ok =
        m_dual.subsolv_status == SubsolvSolveStatus::ConvergedWithinSoftCap ||
        m_dual.subsolv_status ==
            SubsolvSolveStatus::ConvergedAfterSoftCapExtension;
    const bool kkt_ok = inner_status_ok &&
                        m_dual.kkt.max_norm <= m_dual.kkt_residual_tolerance &&
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

    m_replay_capture.problem = sp;
    if (m_replay_pre_solve_callback) m_replay_pre_solve_callback(m_replay_capture);

    const SubsolvResult r = SolveSubsolvFull(sp);

    m_replay_capture.result = r;

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
    m_dual.subsolv_status = r.status;
    m_dual.extended_barrier_levels = r.extended_barrier_levels;
    m_dual.max_newton_iterations_per_barrier =
        r.max_newton_iterations_per_barrier;
    m_dual.extra_newton_iterations = r.extra_newton_iterations;
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
                       const double* xmax, const double* move_scale) {
    // Forward the iterator
    iter++;
    m_replay_capture.dfdx.assign(dfdx, dfdx + n);
    m_replay_capture.gx.assign(gx, gx + m);
    m_replay_capture.dgdx.assign(dgdx, dgdx + n * m);
    m_replay_capture.xmin.assign(xmin, xmin + n);
    m_replay_capture.xmax.assign(xmax, xmax + n);

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

        last_bounds.alpha_standard[i] = alpha[i];
        last_bounds.beta_standard[i] = beta[i];
        // Relative tightening of the already-formed asymmetric MMA interval.
        // Skip identity scales to preserve the original floating-point path.
        if (move_scale && move_scale[i] != 1.0) {
            alpha[i] = xval[i] - move_scale[i] * (xval[i] - alpha[i]);
            beta[i] = xval[i] + move_scale[i] * (beta[i] - xval[i]);
        }

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
