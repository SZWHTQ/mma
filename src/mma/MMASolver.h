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
//
// BETA VERSION  0.99
//
// MMA solver using a dual interior point method
//
// Original code by Niels Aage, February 2013
// Modified to use OpenMP by Jun Wu, April 2017
// Various modifications by Jérémie Dumas, June 2017
//
// The class solves a general non-linear programming problem
// on standard from, i.e. non-linear objective f, m non-linear
// inequality constraints g and box constraints on the n
// design variables xmin, xmax.
//
//        min_x^n f(x)
//        s.t. g_j(x) < 0,   j = 1,m
//        xmin < x_i < xmax, i = 1,n
//
// Each call to Update() sets up and solve the following
// convex subproblem:
//
//   min_x     sum(p0j./(U-x)+q0j./(x-L)) + a0*z + sum(c.*y + 0.5*d.*y.^2)
//
//   s.t.      sum(pij./(U-x)+qij./(x-L)) - ai*z - yi <= bi, i = 1,m
//             Lj < alphaj <=  xj <= betaj < Uj,  j = 1,n
//             yi >= 0, i = 1,m
//             z >= 0.
//
// NOTE: a0 == 1 in this implementation !!!!
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "SubsolvFull.h"

#include <vector>

namespace mma {

/**
 * Outcome of the dual interior-point solve performed by the last Update().
 *
 * `Success` and `FailedToConverge` are the two programmatically distinguishable
 * states a caller may branch on; a failure is never signalled by NaN, by a huge
 * value, or by a log message.
 */
enum class DualSolveStatus {
    NotRun,           ///< Update() has not been called yet.
    Success,          ///< The returned dual point passed qualification.
    FailedToConverge, ///< The returned dual point failed qualification.
};

/**
 * Read-only report of the last dual (interior-point) solve.
 *
 * Everything here is a *record of what SolveDIP did*; none of it feeds back into
 * the arithmetic. `lambda`/`mu` are copies of the returned dual variables, and
 * the residuals are the value of the solver's own KKT residual expression at the
 * returned point (see DualResidual) evaluated at `epsi_final`. Note that this is
 * not the same number as the last residual the inner Newton loop evaluated: the
 * inner loop never re-bases its residual between barrier levels, so a barrier
 * level whose inner loop is skipped leaves the complementarity residual parked at
 * the last *solved* level. The returned-point residual below is the honest
 * measure of what was delivered, and it is the quantity the qualification test
 * uses.
 */
struct DualSolveDiagnostics {
    DualSolveStatus status = DualSolveStatus::NotRun;

    /// Final dual variables of the returned point.
    std::vector<double> lambda;
    std::vector<double> mu;

    /// Last barrier level the outer loop entered, and the tolerance it targets.
    double epsi_final = 0.0;
    double epsimin = 0.0;

    /// Max-norm KKT residual of the returned point at `epsi_final`, split into
    /// its two implemented components. `dual_residual == max(stationarity,
    /// complementarity)` by construction.
    double dual_residual = 0.0;
    /// max_j |G_j + mu_j|  (dual stationarity / dual feasibility)
    double dual_stationarity_residual = 0.0;
    /// max_j |mu_j * lambda_j - epsi|  (the single complementarity relation the
    /// vendored solver keeps)
    double dual_complementarity_residual = 0.0;
    /// The threshold `dual_residual` was tested against.
    double dual_residual_tolerance = 0.0;

    /// Inner Newton work, summed over all barrier levels.
    int barrier_levels = 0;
    int capped_barrier_levels = 0;
    int inner_newton_iterations = 0;

    /// Structural checks of the returned state.
    bool all_finite = false;
    bool design_within_subproblem_box = false;

    // ==================================================================
    // Full primal-dual KKT/barrier residual of the returned point (m9).
    //
    // The reduced residual above only ever had two components (the
    // constraint stationarity and one complementarity relation) because the
    // reduced solver had no xsi/eta/zet/s state. With the full solver all
    // nine reference equation groups are available and are the primary
    // quality measure; `dual_residual` above is retained for continuity with
    // the Phase-1 contract.
    // ==================================================================
    SubsolvResidual kkt;

    /// The returned point lies inside its own domain: alfa <= x <= beta,
    /// y,z,lam,mu,s,zet >= 0. A point outside it is a structural failure
    /// whatever its residual.
    bool kkt_domain_ok = false;
    /// The reference's Newton-system branch could not be taken as written
    /// (see SubsolvFull.cpp); always a failure, never a silent fallback.
    bool kkt_unsupported_branch = false;
    /// Threshold `kkt.max_norm` is tested against, in units of epsimin.
    double kkt_residual_tolerance = 0.0;

    /// Inner Newton iterations that accepted the full feasible step, and those
    /// that needed backtracking.
    int full_step_iterations = 0;
    int backtracking_iterations = 0;
    int backtracking_reductions = 0;
    int backtracking_max_reductions = 0;
    int backtracking_exhausted = 0;

    /// True when the returned point failed qualification and the solver restored
    /// the iteration state it had before this Update() call, so that the caller's
    /// design vector is unchanged and the asymptote history is not advanced.
    bool update_rejected = false;
};

class MMASolver {

  public:
    MMASolver(int n, int m, double a = 0.0, double c = 1000.0, double d = 0.0);

    void SetAsymptotes(double init, double decrease, double increase);

    void ConstraintModification(bool conMod) {}

    void Update(double* xval, const double* dfdx, const double* gx,
                const double* dgdx, const double* xmin, const double* xmax);

    void Reset() { iter = 0; };

    // ==================================================================
    // Read-only observability of the dual solve. These accessors report;
    // they never modify MMA state.
    // ==================================================================

    /// Diagnostics of the dual solve performed by the most recent Update().
    const DualSolveDiagnostics& GetDualSolveDiagnostics() const noexcept {
        return m_dual;
    }
    DualSolveStatus GetDualSolveStatus() const noexcept {
        return m_dual.status;
    }
    /// True only when the most recent Update() produced a qualified dual point.
    bool LastDualSolveSucceeded() const noexcept {
        return m_dual.status == DualSolveStatus::Success;
    }
    /// True when the most recent Update() was rejected and rolled back.
    bool LastUpdateRejected() const noexcept { return m_dual.update_rejected; }

    /// A returned dual point is accepted while its max-norm KKT residual stays
    /// within this multiple of `epsimin`.
    ///
    /// This is a *qualification threshold* calibrated against measured
    /// production behaviour, not a claim that the solver reaches `epsimin`. The
    /// largest residual measured on any solve that did NOT saturate the inner
    /// Newton cap is 7066 * epsimin over the m7 matrix (n <= 32) and 1557 *
    /// epsimin on the first update of the n = 1803 multi-material assembly
    /// fixture; every solve that saturated the cap at four or more barrier
    /// levels reaches at least 2.9e9 * epsimin except for the mild ones, which
    /// the cap clause catches. The factor therefore sits one decade above every
    /// measured non-stalling solve.
    ///
    /// The ratio does not shrink with problem size, so the `sqrt(n+m)` growth of
    /// `epsimin` is not the right normalisation; a scale-relative measure of the
    /// stationarity residual is the Phase 2 direction. See
    /// docs/audits/mma-dual-reliability.md.
    static double DualResidualToleranceFactor() noexcept { return 1.0e4; }

    /// Acceptance threshold for the FULL primal-dual KKT/barrier residual of
    /// the returned point, in units of `epsimin`.
    ///
    /// The reference's own convergence target is `epsimin`, so a solve that
    /// converged should land near 1. This factor is the measured compatibility
    /// band of the full solver, calibrated in the m9 study against the archived
    /// reference oracle rather than chosen.
    static double KktResidualToleranceFactor() noexcept { return 1.0e4; }

    /// Retained for continuity with the Phase-1 contract and still reported in
    /// the failure message.
    ///
    /// NO LONGER A FAILURE CLAUSE. Under the reduced solver a cap-saturating
    /// level was the only in-solver evidence that the barrier path had stalled,
    /// because that loop had no residual-decrease acceptance test. The full
    /// solver does have one, and m8 measured a fixture that saturates the cap
    /// while delivering a residual 3e4x inside the threshold; acceptance is now
    /// decided by the full KKT residual (see KktResidualToleranceFactor).
    /// `capped_barrier_levels` remains observable and is reported.
    static int CappedBarrierLevelFailureThreshold() noexcept { return 4; }

  private:
    int n, m, iter;

    const double xmamieps;
    const double epsimin;

    const double raa0;
    const double move, albefa;
    double asyminit, asymdec, asyminc;

    std::vector<double> a, c, d;
    std::vector<double> y;
    double z;

    std::vector<double> lam, mu, s;
    std::vector<double> low, upp, alpha, beta, p0, q0, pij, qij, b, grad, hess;

    std::vector<double> xold1, xold2;

    /// Diagnostics of the last dual solve (see DualSolveDiagnostics).
    DualSolveDiagnostics m_dual;

    /// Iteration state that is committed by a solve and therefore has to be
    /// restored when a solve is rejected. Only `iter`, the two previous design
    /// vectors and the asymptotes carry over from one Update() to the next;
    /// everything else GenSub produces is rebuilt from scratch.
    int m_committed_iter = 0;
    std::vector<double> m_committed_xold1, m_committed_xold2;
    std::vector<double> m_committed_low, m_committed_upp;
    std::vector<double> m_input_x;

    /// Capture / restore the iteration state a solve is allowed to commit.
    void SnapshotIterationState(const double* xval);
    void RestoreIterationState(double* xval);

    /// Qualify the dual point SolveDIP just returned, filling m_dual.
    void QualifyDualSolve(const double* x);

    /// residual components of the returned point (see DualSolveDiagnostics)
    void DualResidualComponents(const double* x, double epsi, double* stationarity,
                                double* complementarity) const;

    void GenSub(const double* xval, const double* dfdx, const double* gx,
                const double* dgdx, const double* xmin, const double* xmax);

    void SolveDSA(double* x);

    /**
     * The solve path. Builds the subproblem in the reference's layout and
     * hands it to the full primal-dual solver (SubsolvFull.h), then adopts the
     * returned state.
     */
    void SolveDIP(double* x);

    /**
     * The former reduced-dual solve path, retained verbatim so that the two
     * solvers can still be compared on the same subproblem. NOT called by
     * Update(). It is scheduled for removal once the full solver is qualified
     * on the outer benchmarks (m9 handoff section 23, item 4).
     */
    void SolveDIPReduced(double* x);

    void XYZofLAMBDA(double* x);

    void DualGrad(double* x);
    void DualHess(double* x);
    void DualLineSearch();
    double DualResidual(double* x, double epsi);

    static void Factorize(double* K, int n);
    static void Solve(double* K, double* x, int n);
};
} // namespace mma