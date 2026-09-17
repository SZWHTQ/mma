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

#include <functional>
#include <utility>
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
 * the arithmetic. `lambda`/`mu` are copies of the returned dual variables. The
 * full primal-dual residual is evaluated at the returned point and is the
 * quantity used by the qualification test.
 */
struct DualSolveDiagnostics {
    DualSolveStatus status = DualSolveStatus::NotRun;
    /// Detailed status of the full primal-dual stage solver.
    SubsolvSolveStatus subsolv_status = SubsolvSolveStatus::NotRun;

    /// Final dual variables of the returned point.
    std::vector<double> lambda;
    std::vector<double> mu;

    /// Last barrier level the outer loop entered, and the tolerance it targets.
    double epsi_final = 0.0;
    double epsimin = 0.0;

    /// Inner Newton work, summed over all barrier levels.
    int barrier_levels = 0;
    int capped_barrier_levels = 0;
    int inner_newton_iterations = 0;
    int extended_barrier_levels = 0;
    int max_newton_iterations_per_barrier = 0;
    int extra_newton_iterations = 0;

    /// How the progress-aware termination policy ended each stage. These are
    /// the same counters `SubsolvResult` returns; without them a caller can see
    /// that a solve succeeded but not why it stopped.
    /// Barrier levels explicitly stopped after persistent lack of progress.
    int stagnated_barrier_levels = 0;
    /// Barrier levels stopped by the generous emergency work limit.
    int emergency_limited_barrier_levels = 0;
    /// Last measured relative reduction when a stage was classified stagnant.
    double stagnation_relative_reduction = 0.0;
    /// Newton iterations at which the policy's window test was performed.
    int progress_policy_tests = 0;
    /// Low-progress window comparisons observed (short-circuit limited).
    int low_progress_windows = 0;
    /// Smallest newest-window relative reduction seen; meaningful only when
    /// `progress_policy_tests > 0`.
    double minimum_window_relative_reduction = 0.0;

    /// Structural checks of the returned state.
    bool all_finite = false;
    bool design_within_subproblem_box = false;

    // ==================================================================
    // Full primal-dual KKT/barrier residual of the returned point (m9).
    //
    // All nine reference equation groups are represented and are the sole
    // numeric quality measure for this solver.
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
                const double* dgdx, const double* xmin, const double* xmax,
                const double* move_scale = nullptr);

    struct LocalBounds {
        std::vector<double> alpha_standard, beta_standard, alpha, beta;
    };
    const LocalBounds& LastLocalBounds() const noexcept { return last_bounds; }

    /// Install an optional observer for exact, immutable subproblem captures.
    /// The observer is diagnostic only and never participates in arithmetic.
    void SetReplayCaptureCallback(
        std::function<void(const MmaReplayFixture&)> callback) {
        m_replay_callback = std::move(callback);
    }

    /// Optional boundary observer invoked after the generated SubsolvProblem
    /// is complete and immediately before SolveSubsolvFull is entered.
    void SetReplayPreSolveCallback(
        std::function<void(const MmaReplayFixture&)> callback) {
        m_replay_pre_solve_callback = std::move(callback);
    }

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
    SubsolvSolveStatus GetSubsolvSolveStatus() const noexcept {
        return m_dual.subsolv_status;
    }
    /// True only when the most recent Update() produced a qualified dual point.
    bool LastDualSolveSucceeded() const noexcept {
        return m_dual.status == DualSolveStatus::Success;
    }
    /// True when the most recent Update() was rejected and rolled back.
    bool LastUpdateRejected() const noexcept { return m_dual.update_rejected; }

    /// Acceptance threshold for the FULL primal-dual KKT/barrier residual of
    /// the returned point, in units of `epsimin`.
    ///
    /// The reference's own convergence target is `epsimin`, so a solve that
    /// converged should land near 1. This factor is the measured compatibility
    /// band of the full solver, calibrated in the m9 study against the archived
    /// reference oracle rather than chosen.
    static double KktResidualToleranceFactor() noexcept { return 1.0e4; }

  private:
    LocalBounds last_bounds;
    int n, m, iter;

    const double xmamieps;
    const double epsimin;

    const double raa0;
    const double move, albefa;
    double asyminit, asymdec, asyminc;

    std::vector<double> a, c, d;
    std::vector<double> y;
    double z;

    std::vector<double> lam, mu;
    std::vector<double> low, upp, alpha, beta, p0, q0, pij, qij, b;

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

    std::function<void(const MmaReplayFixture&)> m_replay_callback;
    std::function<void(const MmaReplayFixture&)> m_replay_pre_solve_callback;
    MmaReplayFixture m_replay_capture;

    /// Capture / restore the iteration state a solve is allowed to commit.
    void SnapshotIterationState(const double* xval);
    void RestoreIterationState(double* xval);

    /// Qualify the dual point SolveDIP just returned, filling m_dual.
    void QualifyDualSolve(const double* x);

    void GenSub(const double* xval, const double* dfdx, const double* gx,
                const double* dgdx, const double* xmin, const double* xmax,
                const double* move_scale);

    /**
     * The solve path. Builds the subproblem in the reference's layout and
     * hands it to the full primal-dual solver (SubsolvFull.h), then adopts the
     * returned state.
     */
    void SolveDIP(double* x);

    void FinalizeReplayCapture(const SubsolvProblem& problem,
                               const SubsolvResult& result,
                               const double* xval, const double* dfdx,
                               const double* gx, const double* dgdx,
                               const double* xmin, const double* xmax);

};
} // namespace mma
