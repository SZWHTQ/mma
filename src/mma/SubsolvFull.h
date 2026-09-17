////////////////////////////////////////////////////////////////////////////////
// Full primal-dual MMA subproblem solver.
//
// A formula-preserving C++ port of the archived official Svanberg v1.5
// `subsolv.m` (GCMMA-MMA-code, Dec 2006). It replaces the reduced dual
// formulation that the vendored Dumas/Aage `MMASolver::SolveDIP` implements.
//
// WHY THIS EXISTS
// ---------------
// The reduced solver keeps only (lam, mu) as state and slaves x, y, z to lam.
// That system's Newton step is not the Newton step of the barrier system it is
// measuring, so the reference's residual-decrease backtracking cannot be
// transplanted onto it: measured on the m7 matrix, doing so refused the full
// feasible step on 80.5% of all inner iterations and created 3007 new
// cap-saturating solves (see /mnt/Local/Results/m8_mma_dual_repair). The
// reference globalization is correct precisely on the full system, so the full
// system is what this file implements.
//
// The variable names, dimensions and equations below follow `subsolv.m`
// verbatim; every block cites the reference lines it ports.
//
// THE SUBPROBLEM (subsolv.m lines 31-37)
// --------------------------------------
//   minimize   SUM_j [ p0_j/(upp_j - x_j) + q0_j/(x_j - low_j) ]
//              + a0*z + SUM_i [ c_i*y_i + 0.5*d_i*y_i^2 ]
//   subject to SUM_j [ p_ij/(upp_j - x_j) + q_ij/(x_j - low_j) ] - a_i*z - y_i
//              <= b_i
//              alfa_j <= x_j <= beta_j,  y_i >= 0,  z >= 0.
//
// DIMENSIONS
// ----------
//   n   design variables
//   m   constraints
//
// P and Q use the REFERENCE layout, which is the transpose of the production
// `pij`/`qij` arrays: P[i*n + j] is MATLAB's P(i+1, j+1) -- row i is the
// constraint, column j is the design variable. The caller converts once.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mma {

/** Read-only outcome of one full primal-dual subproblem solve. */
enum class SubsolvSolveStatus {
    NotRun,
    ConvergedWithinSoftCap,
    ConvergedAfterSoftCapExtension,
    StagnatedAfterSoftCapExtension,
    EmergencyWorkLimitExhausted,
    HardCapExhausted,
    NumericalFailure,
    DomainFailure,
};

/**
 * Full KKT / barrier residual of the subproblem, using the reference's own
 * nine equation groups (subsolv.m lines 72-85, evaluated at any point and
 * barrier parameter). Every field is the component-wise maximum of |.| over
 * that group's vector.
 */
struct SubsolvResidual {
    // Full equation vectors, retained before norm reduction for replay and
    // independent audit. The scalar fields below remain the production
    // compatibility interface and are computed from these vectors.
    std::vector<double> rex_values, rey_values, relam_values, rexsi_values,
        reeta_values, remu_values, res_values;
    double rez_value = 0.0;
    double rezet_value = 0.0;

    /// x stationarity: dpsidx - xsi + eta                          (subsolv 72)
    double rex = 0.0;
    /// y stationarity: c + d.*y - mu - lam                         (subsolv 73)
    double rey = 0.0;
    /// z stationarity: a0 - zet - a'*lam                           (subsolv 74)
    double rez = 0.0;
    /// primal constraints: gvec - a*z - y + s - b                  (subsolv 75)
    double relam = 0.0;
    /// lower-bound complementarity: xsi.*(x-alfa) - epsi           (subsolv 76)
    double rexsi = 0.0;
    /// upper-bound complementarity: eta.*(beta-x) - epsi           (subsolv 77)
    double reeta = 0.0;
    /// y/mu complementarity: mu.*y - epsi                          (subsolv 78)
    double remu = 0.0;
    /// z/zet complementarity: zet*z - epsi                         (subsolv 79)
    double rezet = 0.0;
    /// lambda/s complementarity: lam.*s - epsi                     (subsolv 80)
    double res = 0.0;

    /// max(abs(residu))  -- the inner loop's own stopping measure (subsolv 85)
    double max_norm = 0.0;
    /// sqrt(residu'*residu) -- the reference's backtracking measure (subsolv 84)
    double norm2 = 0.0;

    /// Largest of the nine component maxima.
    double worst_component() const;
};

/// The subproblem exactly as `subsolv.m` receives it.
struct SubsolvProblem {
    int n = 0;
    int m = 0;
    double epsimin = 1.0e-7;
    /// n
    std::vector<double> low, upp, alfa, beta;
    /// n
    std::vector<double> p0, q0;
    /// m*n, reference layout: P[i*n + j] == MATLAB P(i+1, j+1)
    std::vector<double> P, Q;
    double a0 = 1.0;
    /// m
    std::vector<double> a, b, c, d;
};

/** One behavior-neutral observation emitted by an explicitly traced solve. */
struct SubsolvTraceRecord {
    const char* event = ""; // barrier_start, newton_step, or barrier_end
    int barrier_level = 0;
    int newton_iteration = 0;
    int newton_iteration_in_barrier = 0;
    double epsi = 0.0;
    double rex = 0.0, rey = 0.0, rez = 0.0, relam = 0.0;
    double rexsi = 0.0, reeta = 0.0, remu = 0.0, rezet = 0.0, res = 0.0;
    double max_norm = 0.0;
    double norm2 = 0.0;
    double newton_step_norm = 0.0;
    int backtracking_trials = 0;
    int backtracking_reductions = 0;
    bool domain_ok = false;
    bool all_finite = false;
    bool linear_system_ok = true;
};

/** Policy decision after the normal per-stage work budget is exhausted. */
enum class SubsolvProgressDecision {
    Continue,
    Stagnated,
};

/**
 * Bookkeeping of ONE evaluation of the progress policy's window test.
 *
 * Every field records what `Decide` inspected, never what it decided, so
 * filling a report cannot move a solve. `windows_tested` is short-circuit
 * limited: `Decide` returns at the first window that shows adequate progress,
 * so under a `Continue` verdict it counts only the comparisons needed to reach
 * that window -- a lower bound on the windows the policy looked at, never an
 * upper one.
 */
struct SubsolvProgressReport {
    /// At least one window comparison was actually performed.
    bool tested = false;
    /// Window comparisons performed before the verdict was reached.
    int windows_tested = 0;
    /// Those comparisons whose relative reduction did not exceed the threshold.
    int low_progress_windows = 0;
    /// Relative reduction of the newest window at the last comparison.
    /// Meaningful only when `tested`.
    double newest_window_relative_reduction = 0.0;
};

/**
 * Scale-insensitive, solver-level progress monitor. The history is indexed by
 * Newton iteration within one barrier stage and contains the full residual
 * 2-norm, including the stage-start value at index zero.
 *
 * `report` is an optional pure-observation out-parameter; passing it changes
 * no decision and no arithmetic.
 */
struct SubsolvProgressPolicy {
    int window = 20;
    int consecutive_windows = 2;
    double minimum_relative_reduction = 1.0e-6;

    SubsolvProgressDecision Decide(const std::vector<double>& residual_history,
                                   int iteration, int soft_budget,
                                   SubsolvProgressReport* report = nullptr)
        const noexcept;
};

/** Explicitly opt-in diagnostic controls; defaults select the production policy. */
struct SubsolvSolveOptions {
    // Zero selects the production policy (200-iteration soft cap and the
    // hard safety cap). A positive value is an explicit diagnostic hard cap.
    int inner_iteration_cap = 0;
    // Diagnostic-only compatibility mode: retain later barrier records after
    // a deliberately low cap, but still return HardCapExhausted.
    bool stop_on_hard_cap = true;
    // Defaults enable the progress-aware production policy. Positive
    // inner_iteration_cap remains an explicit historical/diagnostic cap and
    // disables progress-aware termination for replay compatibility.
    SubsolvProgressPolicy progress_policy;
    std::function<void(const SubsolvTraceRecord&)> trace;
};

/// Everything `subsolv.m` returns, plus the diagnostics this project needs.
struct SubsolvResult {
    SubsolvSolveStatus status = SubsolvSolveStatus::NotRun;

    /// n
    std::vector<double> x;
    /// m
    std::vector<double> y;
    double z = 0.0;
    /// m
    std::vector<double> lam;
    /// n
    std::vector<double> xsi;
    /// n
    std::vector<double> eta;
    /// m
    std::vector<double> mu;
    double zet = 0.0;
    /// m
    std::vector<double> s;

    /// Full KKT residual at the RETURNED point, at `epsi_final`.
    SubsolvResidual residual;

    double epsi_final = 0.0;
    int barrier_levels = 0;
    int inner_newton_iterations = 0;
    /// Barrier levels that exhausted the absolute hard cap.
    int capped_barrier_levels = 0;
    /// Barrier levels that crossed the historical soft cap and were extended.
    int extended_barrier_levels = 0;
    /// Maximum Newton iterations used by any one barrier level.
    int max_newton_iterations_per_barrier = 0;
    /// Newton iterations performed after soft-cap saturation.
    int extra_newton_iterations = 0;
    /// Barrier levels explicitly stopped after persistent lack of progress.
    int stagnated_barrier_levels = 0;
    /// Barrier levels stopped by the generous emergency work limit.
    int emergency_limited_barrier_levels = 0;
    /// Last measured relative reduction when a stage was classified stagnant.
    double stagnation_relative_reduction = 0.0;

    /// Newton iterations at which the progress policy's window test was
    /// performed, i.e. where the stage had enough post-soft-cap history for the
    /// policy to judge. Zero means the policy never became decisive.
    int progress_policy_tests = 0;
    /// Low-progress window comparisons observed across those tests, summed
    /// over all stages. Short-circuit limited -- see SubsolvProgressReport.
    int low_progress_windows = 0;
    /// Smallest newest-window relative reduction seen at any comparison.
    /// Meaningful only when `progress_policy_tests > 0`.
    double minimum_window_relative_reduction = 0.0;

    int backtracking_iterations = 0;   ///< Newton iterations needing >=1 halving
    int backtracking_reductions = 0;   ///< halvings summed over the solve
    int backtracking_max_reductions = 0;
    int backtracking_exhausted = 0;    ///< line searches that gave up
    int full_step_iterations = 0;      ///< Newton iterations accepted at once

    /// Every returned state entry is finite and inside its own domain
    /// (alfa <= x <= beta, y >= 0, z >= 0, lam >= 0, mu >= 0, s >= 0,
    /// xsi > 0, eta > 0, zet >= 0).
    bool domain_ok = false;
    bool all_finite = false;

    /// Set when the `m < n` / `m >= n` branch could not be taken as the
    /// reference takes it (see the note on `kDenseBranchLimit` in the .cpp).
    bool unsupported_branch = false;
};

/**
 * Solve the subproblem with the full primal-dual interior-point method of
 * `subsolv.m`. Pure function of the problem: no global state, no I/O.
 */
SubsolvResult SolveSubsolvFull(const SubsolvProblem& problem);

/** Traced/extended offline solve. The equations and globalization are unchanged. */
SubsolvResult SolveSubsolvFull(const SubsolvProblem& problem,
                               const SubsolvSolveOptions& options);

/// The full KKT/barrier residual of a point, evaluated with the reference's
/// equations. Exposed so that a returned solution can be scored independently
/// of the solver that produced it.
SubsolvResidual EvaluateSubsolvResidual(const SubsolvProblem& problem,
                                        const SubsolvResult& point,
                                        double epsi);

/**
 * Versioned, self-contained capture of one exact production MMA call.
 *
 * The binary representation is deliberately separate from the scientific
 * results/checkpoint formats: it is a replay artifact for the subproblem
 * boundary, not a restart format.
 */
struct MmaReplayFixture {
    static constexpr std::uint32_t kFormatVersion = 1;

    std::uint32_t format_version = kFormatVersion;
    int iteration = 0;
    int update_number = 0;
    int n = 0;
    int m = 0;

    // Exact public Update inputs and persistent MMA state.
    std::vector<double> xval, xold1, xold2, xmin, xmax;
    std::vector<double> low, upp;
    std::vector<double> dfdx, gx, dgdx;
    std::vector<double> df0dx, fval;
    std::vector<double> a, c, d;
    double a0 = 1.0;
    double f0val = 0.0;
    bool objective_values_supplied = false;
    double epsimin = 0.0;
    double xmamieps = 0.0;
    double raa0 = 0.0;
    double move = 0.0;
    double albefa = 0.0;
    double asyminit = 0.0;
    double asymdec = 0.0;
    double asyminc = 0.0;

    // Exact generated input to subsolv, including derived bounds.
    SubsolvProblem problem;

    // Exact returned primal-dual state and diagnostics.
    SubsolvResult result;
    double objective_value = 0.0; // recorded only when supplied by caller
    double constraint_value_norm = 0.0;

    // Transaction hashes: FNV-1a over canonical little-endian scalar bytes.
    std::uint64_t before_design_hash = 0;
    std::uint64_t candidate_design_hash = 0;
    std::uint64_t after_rejection_design_hash = 0;
    std::uint64_t before_history_hash = 0;
    std::uint64_t candidate_history_hash = 0;
    std::uint64_t after_rollback_history_hash = 0;
    bool update_rejected = false;
};

void WriteMmaReplayFixture(const MmaReplayFixture& fixture,
                           const std::string& path);
MmaReplayFixture ReadMmaReplayFixture(const std::string& path);

/** Solve only the frozen subproblem and compare its returned state/residual. */
void VerifyMmaReplayFixture(const MmaReplayFixture& fixture,
                            double tolerance = 0.0);

/// Reference constants that are part of `subsolv.m`'s control logic.
struct SubsolvConstants {
    /// Historical per-stage work budget, `ittt < 200` (subsolv.m line 87).
    static int InnerIterationCap() noexcept { return 200; }
    /// Historical absolute cap retained for explicit diagnostic modes.
    static int HardIterationCap() noexcept { return 1100; }
    /// Runaway guard for the progress-aware production policy. This is not a
    /// normal convergence/failure criterion.
    static int EmergencyWorkLimit() noexcept { return 10000; }
    /// Backtracking budget, `itto < 50` (subsolv.m line 175).
    static int BacktrackingReductionLimit() noexcept { return 50; }
    /// Barrier reduction, `epsi = 0.1*epsi` (subsolv.m line 219).
    static double BarrierReduction() noexcept { return 0.1; }
    /// Initial barrier, `epsi = 1` (subsolv.m line 44).
    static double InitialBarrier() noexcept { return 1.0; }
    /// Inner-loop acceptance factor, `residumax > 0.9*epsi` (subsolv.m line 87).
    static double InnerResidualFactor() noexcept { return 0.9; }
    /// Step-limiting safety factor, `-1.01*dxx./xx` (subsolv.m line 152).
    static double StepSafety() noexcept { return 1.01; }
};

} // namespace mma
